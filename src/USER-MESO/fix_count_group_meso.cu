
#include "domain.h"
#include "mpi.h"
#include "stdio.h"
#include "string.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "group.h"

#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "engine_meso.h"
#include "fix_count_group_meso.h"
#include "comm_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
/*
 * Count number of particles for each group.
 * Output to file.
 */
MesoFixCountGroup::MesoFixCountGroup(LAMMPS *lmp, int narg, char **arg):
    Fix(lmp, narg, arg),
    MesoPointers(lmp),
    dev_count( lmp, "MesoFixCountGroup::dev_count" ),
    dev_groupbits( lmp, "MesoFixCountGroup::dev_groupbits" )
{
	every = 0;

    for( int i = 0 ; i < narg ; i++ )
    {
        if ( !strcmp( arg[i], "output" ) )
        {
            if ( ++i >= narg ) error->all(FLERR,"Incomplete fix MesoFixCountGroup command after 'output'");
            output = arg[i];
            continue;
        }
        else if ( !strcmp( arg[i], "every" ) )
        {
            if ( ++i >= narg ) error->all(FLERR,"Incomplete fix MesoFixCountGroup command after 'every'");
            every = atoi( arg[i] );
            continue;
        }
        else if ( !strcmp( arg[i], "gid" ) )  // group-id
        {
            if ( ++i >= narg ) error->all(FLERR,"Incomplete fix MesoFixCountGroup command after 'groupid'");
            int tgroup;
        	if ( (tgroup = group->find( arg[i] ) ) == -1 )
        		error->all( FLERR, "<MESO> Undefined group id in fix MesoFixCountGroup" );
        	target_groupbits.push_back( group->bitmask[tgroup] );
            continue;
        }
    }

    if ( output == "" | every == 0 ) {
        error->all(FLERR,"Incomplete fix countgroup/meso command: insufficient arguments");
    }

    dump_time 	= 1;
	me 			= comm->me;   			// my rank
	nprocs 		= comm->nprocs;			// world size
	ngroups 	= target_groupbits.size();
    dev_count    .grow( ngroups );
    dev_groupbits.grow( ngroups*nprocs );

    fout.open( output.c_str() );
    fout << std::setprecision(15);
    fout << "Step\tRank0\tCount(Group...)\tRank1\tCount(Group...)\t...\n";
}

MesoFixCountGroup::~MesoFixCountGroup()
{
    fout.close();
}

int MesoFixCountGroup::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_INTEGRATE;
    return mask;
}

void MesoFixCountGroup::setup(int vflag)
{
    dev_count.set( 0 , meso_device->stream() );
    dev_groupbits.upload( target_groupbits.data(), target_groupbits.size(), meso_device->stream() );
}

void MesoFixCountGroup::post_integrate()
{
    compute();
}

// 32-lane bitonic sort for local reduction
// hopefully spatial locality will help in this case
__global__ void gpu_velocity_profile_histogram(
    int* __restrict mask,
    int* __restrict groupbits,
    uint*__restrict count,
    const int ngroups,
    const int n_local )
{
    extern __shared__ char __shmem__[];
    uint *local_count = (uint*) &__shmem__[0];

    for(int i = threadIdx.x ; i < ngroups ; i += blockDim.x )
        local_count[i] = 0;

    __syncthreads();

    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_local; i += gridDim.x * blockDim.x )
    {
    	for( int j=0; j<ngroups; j++ ) {
    		if( mask[i] & groupbits[j] )
    			atomicInc( local_count + j, 0xFFFFFFFFU );
    	}
    }
    __syncthreads();

    for(int i = threadIdx.x ; i < ngroups ; i += blockDim.x) {
        atomic_add( count + i, local_count[i] );
    }
}

void MesoFixCountGroup::compute()
{
    static GridConfig grid_cfg;
    if ( !grid_cfg.x )
    {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_velocity_profile_histogram, 0, cudaFuncCachePreferShared );
        cudaFuncSetCacheConfig( gpu_velocity_profile_histogram, cudaFuncCachePreferShared );
    }

    if ( dump_time == update->ntimestep )
    {
        gpu_velocity_profile_histogram<<< grid_cfg.x, grid_cfg.y, dev_count.n_byte(), meso_device->stream() >>>(
            meso_atom->dev_mask,
            dev_groupbits,
            dev_count,
            ngroups,
            atom->nlocal );

        this->dump( dump_time );
        dump_time += every;
    }
}

void MesoFixCountGroup::dump( bigint tstamp )
{
    // dump result
    std::vector<uint> count( ngroups, 0 );
    dev_count.download( &count[0], count.size(), meso_device->stream() );
    meso_device->sync_device();
    dev_count.set( 0, meso_device->stream() );

    // MPI_Gather from all ranks
    std::vector<uint> count_all( ngroups*nprocs, 0 );
    MPI_Gather( count.data(), ngroups, MPI_UNSIGNED, count_all.data(), ngroups, MPI_UNSIGNED, 0, world );


    // write to file
    if ( me == 0 )
    {
    	fout << std::setw(8) << update->ntimestep << ' ';
        for( int i = 0 ; i < count_all.size() ; i++ ) {
        	if (i%ngroups == 0) fout << i/ngroups << ' ';  // write rank-id
            fout << count_all[i] << ' ';
        }
        fout << std::endl;
    }
}

