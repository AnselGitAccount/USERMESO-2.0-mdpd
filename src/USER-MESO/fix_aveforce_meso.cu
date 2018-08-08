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


#include "mpi.h"
#include "stdio.h"
#include "string.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "domain.h"

#include "atom_meso.h"
#include "comm_meso.h"
#include "atom_vec_meso.h"
#include "engine_meso.h"
#include "fix_aveforce_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MesoFixAveForce::MesoFixAveForce( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    MesoPointers( lmp ),
    foriginal( lmp, "MesoFixAveForce::foriginal" ),
    foriginal_all( lmp, "MesoFixAveForce::foriginal_all" )
{
    if( narg < 6 ) error->all( FLERR, "Illegal fix aveforce/meso command" );

    xstyle = ystyle = zstyle = 0;
    xvalue = yvalue = zvalue = 0;

    if (!strcmp(arg[3],"NULL")) xvalue = 0;	else { xvalue = atof( arg[3] ); xstyle = 1; }
    if (!strcmp(arg[4],"NULL")) yvalue = 0;	else { yvalue = atof( arg[4] ); ystyle = 1; }
    if (!strcmp(arg[5],"NULL")) zvalue = 0;	else { zvalue = atof( arg[5] ); zstyle = 1; }

    foriginal.grow( 4 );
    foriginal_all.grow( 4 );
}

int MesoFixAveForce::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_FORCE;
    return mask;
}

void MesoFixAveForce::init()
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixAveForce. %s %d\n", __FILE__, __LINE__ );
    }
}

void MesoFixAveForce::setup( int vflag )
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixAveForce. %s %d\n", __FILE__, __LINE__ );
    }
    post_force( vflag );
}

__global__ void gpu_fix_aveforce_sum_force(
    r64* __restrict force_x,
    r64* __restrict force_y,
    r64* __restrict force_z,
    int* __restrict mask,
    const int groupbit,
    const int n,
    r64* fsum )
{
	r64 fx=0, fy=0, fz=0;
	if (threadIdx.x==0) fsum[0] = fsum[1] = fsum[2] = fsum[3] = 0;
	__syncthreads();
	int count=0;
	for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
		if (mask[i] & groupbit) {
			fx = force_x[i];
			fy = force_y[i];
			fz = force_z[i];
			count = 1;
		}
    }

	r64 fxsum     = __warp_sum( fx );
	r64 fysum     = __warp_sum( fy );
	r64 fzsum     = __warp_sum( fz );
	int countwarp = __warp_sum( count );
	if ( __laneid()==0 ) {
		atomic_add( fsum  , fxsum );
		atomic_add( fsum+1, fysum );
		atomic_add( fsum+2, fzsum );
		atomic_add( fsum+3, countwarp );
//		printf("fxsum %g %g %g countwarp %d ",fxsum, fysum, fzsum,countwarp);
	}
}


__global__ void gpu_fix_aveforce_set_force(
    r64* __restrict force_x,
    r64* __restrict force_y,
    r64* __restrict force_z,
    int* __restrict mask,
    const int groupbit,
    const int n,
    const int xstyle,
    const int ystyle,
    const int zstyle,
    const r64 xvalue,
    const r64 yvalue,
    const r64 zvalue,
    const r64* fsum )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
        	if( xstyle ) force_x[i] = fsum[0]/fsum[3] + xvalue;
            if( ystyle ) force_y[i] = fsum[1]/fsum[3] + yvalue;
            if( zstyle ) force_z[i] = fsum[2]/fsum[3] + zvalue;
        }
    }
}

void MesoFixAveForce::post_force( int vflag )
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_aveforce_sum_force, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_aveforce_sum_force, cudaFuncCachePreferL1 );
    }

    foriginal.set(0);
    foriginal_all.set(0);

    // sum forces on participating atoms
    gpu_fix_aveforce_sum_force <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        meso_atom->dev_force(0),
        meso_atom->dev_force(1),
        meso_atom->dev_force(2),
        meso_atom->dev_mask,
        groupbit,
        atom->nlocal,
        foriginal);

    // average the force on participating atoms
    // add in requested amount
    r64 foriginal_host[4];
    r64 foriginal_all_host[4];
    foriginal.download( &foriginal_host[0], 4, meso_device->stream() );
    meso_device->stream().sync();
    MPI_Allreduce(foriginal_host,foriginal_all_host,4,MPI_DOUBLE,MPI_SUM,world);

//    printf("foriginal_host %g %g %g %g \n",foriginal_host[0],foriginal_host[1],foriginal_host[2],foriginal_host[3]);
//    printf("foriginal_all_host %g %g %g %g \n\n",foriginal_all_host[0],foriginal_all_host[1],foriginal_all_host[2],foriginal_all_host[3]);

    // set force of all participating atoms to same value
    // only for active dimensions
    if (foriginal_all_host[3] == 0) error->all(FLERR,"<MESO> fix_aveforce_meso; no particle selected.");
    foriginal_all.upload( &foriginal_all_host[0], 4, meso_device->stream() );
    gpu_fix_aveforce_set_force <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        meso_atom->dev_force(0),
        meso_atom->dev_force(1),
        meso_atom->dev_force(2),
        meso_atom->dev_mask,
        groupbit,
        atom->nlocal,
        xstyle, ystyle, zstyle,
        xvalue, yvalue, zvalue,
        foriginal_all);

}
