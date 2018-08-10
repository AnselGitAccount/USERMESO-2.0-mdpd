/* ----------------------------------------------------------------------
     LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
     http://lammps.sandia.gov, Sandia National Laboratories
     Steve Plimpton, sjplimp@sandia.gov

     Copyright (2003) Sandia Corporation.   Under the terms of Contract
     DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
     certain rights in this software.   This software is distributed under
     the GNU General Public License.

     See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "domain.h"
#include "mpi.h"
#include "stdio.h"
#include "string.h"
#include "force.h"
#include "update.h"
#include "error.h"

#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "engine_meso.h"
#include "fix_density_profile_meso.h"
#include "comm_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MesoFixDensityProf::MesoFixDensityProf(LAMMPS *lmp, int narg, char **arg):
    Fix(lmp, narg, arg),
    MesoPointers(lmp),
    dev_count    ( lmp, "MesoFixDensityProf::dev_count" )
{
    along     = -1;
    n_bin     =  0;
    every     =  0;
    window    =  0;
    last_dump_time = -1;
    count_frames = 0;

    for( int i = 0 ; i < narg ; i++ )
    {
        if ( !strcmp( arg[i], "output" ) )
        {
            if ( ++i >= narg ) error->all(FLERR,"Incomplete fix density/meso command after 'output'");
            output = arg[i];
        }
        if ( !strcmp( arg[i], "along" ) )
        {
            if ( ++i >= narg ) error->all(FLERR,"Incomplete fix density/meso command after 'along'");
            if ( arg[i][0] == 'x' ) along = 0;
            if ( arg[i][0] == 'y' ) along = 1;
            if ( arg[i][0] == 'z' ) along = 2;
        }
        else if ( !strcmp( arg[i], "nbin" ) )
        {
            if ( ++i >= narg ) error->all(FLERR,"Incomplete fix density/meso command after 'nbin'");
            n_bin = atoi( arg[i] );
        }
        else if ( !strcmp( arg[i], "every" ) )
        {
            if ( ++i >= narg ) error->all(FLERR,"Incomplete fix density/meso command after 'every'");
            every = atoi( arg[i] );
        }
        else if ( !strcmp( arg[i], "window" ) )
        {
            if ( ++i >= narg ) error->all(FLERR,"Incomplete fix density/meso command after 'window'");
            window = atoi( arg[i] );
        }
    }

    if ( output == "" || along == -1 || n_bin == 0 || every == 0 || window == 0 ) {
        error->all(FLERR,"Incomplete fix vprof command: insufficient arguments");
    }
    if ( window * 2 >= every ) {
        error->warning(FLERR,"fix vprof: window size larger than sampling length");
    }

    dev_count.grow( n_bin );
}

MesoFixDensityProf::~MesoFixDensityProf()
{
    bigint step = update->ntimestep;
    bigint n = ( step + every/2 ) / every;
    bigint m = n * every;
    if ( step - m >= -window && step - m < window ) {
        if ( last_dump_time < step ) {
            dump( m );
            count_frames = 0;
        }
    }
}

int MesoFixDensityProf::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_INTEGRATE;
    return mask;
}

void MesoFixDensityProf::setup(int vflag)
{
    bin_size = ( domain->boxhi[along] - domain->boxlo[along] ) / n_bin;
    dev_count    .set( 0 , meso_device->stream() );
}

void MesoFixDensityProf::post_integrate()
{
    compute();
}

// 32-lane bitonic sort for local reduction
// hopefully spatial locality will help in this case
__global__ void gpu_density_profile(
    r64* __restrict coord,
    uint*__restrict count,
    int* __restrict mask,
    const int groupbit,
    const r64 box_low,
    const r64 bin_inv,
    const int n_bin,
    const int n_all )
{
    extern __shared__ char __shmem__[];
    uint *local_count = (uint*) &__shmem__[0];

    for(int i = threadIdx.x ; i < n_bin ; i += blockDim.x )
        local_count[i] = 0;

    __syncthreads();

    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_all; i += gridDim.x * blockDim.x )
    {
    	if (mask[i] & groupbit) {
    		r64 x = coord[i];
    		int bin = floor( ( x - box_low ) * bin_inv );
    		if ( bin >= 0 && bin < n_bin )
    			atomicInc( local_count + bin, 0xFFFFFFFFU );
    	}
    }

    __syncthreads();

    for(int i = threadIdx.x ; i < n_bin ; i += blockDim.x)
        atomic_add( count + i, local_count[i] );

}

void MesoFixDensityProf::compute()
{
    static GridConfig grid_cfg;
    if ( !grid_cfg.x )
    {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_density_profile, 0, cudaFuncCachePreferShared );
        cudaFuncSetCacheConfig( gpu_density_profile, cudaFuncCachePreferShared );
    }

    bigint step = update->ntimestep;
    bigint n = ( step + every/2 ) / every;
    bigint m = n * every;

    if ( step - m >= -window && step - m < window )
    {
        r64 *dev_coord;
        if ( along == 0 ) dev_coord = meso_atom->dev_coord(0);
        if ( along == 1 ) dev_coord = meso_atom->dev_coord(1);
        if ( along == 2 ) dev_coord = meso_atom->dev_coord(2);

        gpu_density_profile <<< grid_cfg.x, grid_cfg.y, dev_count.n_byte(), meso_device->stream() >>>(
            dev_coord,
            dev_count,
            meso_atom->dev_mask,
            groupbit,
            domain->boxlo[along],
            1.0/bin_size,
            n_bin,
            atom->nlocal );

        count_frames++;
    }
    else if ( step - m == window )
    {
        dump( m );
        count_frames = 0;
    }
}

void MesoFixDensityProf::dump( bigint tstamp )
{
    last_dump_time = update->ntimestep;

    // dump result
    std::vector<uint> count( n_bin, 0 );
    dev_count.download( &count[0], count.size(), meso_device->stream() );
    meso_device->sync_device();
    dev_count.set( 0, meso_device->stream() );

    std::vector<uint> count_master( n_bin, 0  );

    MPI_Reduce( count.data(), count_master.data(), n_bin, MPI_UNSIGNED, MPI_SUM, 0, world );

    if ( comm->me == 0 )
    {
        std::ofstream fout;
        char fn[256];
        sprintf( fn, "%s.%09d", output.c_str(), tstamp );
        fout.open( fn );
        fout<< std::setprecision(15);
        for( int i = 0 ; i < n_bin ; i++ )
            fout << (i+0.5) * bin_size << '\t'
                 << count_master[i]/float(count_frames) << std::endl;
        fout.close();
    }
}

