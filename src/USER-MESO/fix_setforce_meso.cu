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
#include "fix_setforce_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MesoFixSetForce::MesoFixSetForce( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    MesoPointers( lmp )
{
    if( narg < 6 ) error->all( FLERR, "Illegal fix setforce/meso command" );

    fx_flag = fy_flag = fz_flag = 0;

 	if (!strcmp(arg[3],"NULL")) fx = 0;	else { fx = atof( arg[3] ); fx_flag = 1; }
	if (!strcmp(arg[4],"NULL")) fy = 0;	else { fy = atof( arg[4] ); fy_flag = 1; }
	if (!strcmp(arg[5],"NULL")) fz = 0;	else { fz = atof( arg[5] ); fz_flag = 1; }
}

int MesoFixSetForce::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_FORCE;
    return mask;
}

void MesoFixSetForce::init()
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixSetForce. %s %d\n", __FILE__, __LINE__ );
    }
}

void MesoFixSetForce::setup( int vflag )
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixSetForce. %s %d\n", __FILE__, __LINE__ );
    }
    post_force( vflag );
}

__global__ void gpu_fix_set_force(
    r64* __restrict force_x,
    r64* __restrict force_y,
    r64* __restrict force_z,
    int* __restrict mask,
    const r64 fx,
    const r64 fy,
    const r64 fz,
    const int fx_flag,
	const int fy_flag,
	const int fz_flag,
    const int groupbit,
    const int n )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
        	if( fx_flag ) force_x[i] = fx;
            if( fy_flag ) force_y[i] = fy;
            if( fy_flag ) force_z[i] = fz;
        }
    }
}

void MesoFixSetForce::post_force( int vflag )
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_set_force, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_set_force, cudaFuncCachePreferL1 );
    }

    gpu_fix_set_force <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        meso_atom->dev_force(0),
        meso_atom->dev_force(1),
        meso_atom->dev_force(2),
        meso_atom->dev_mask,
        fx, fy, fz,
        fx_flag, fy_flag, fz_flag,
        groupbit,
        atom->nlocal );
}
