

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
#include "fix_addconf_tdpd_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MesoFixAddConfTDPD::MesoFixAddConfTDPD( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    index ( 100 ),
    MesoPointers( lmp ),
    cf( lmp, "add constant flux within a location range")
{
    int parg = 3;
    if (!strcmp( arg[parg], "x" )) index=0;
    else if (!strcmp( arg[parg], "y")) index=1;
    else if (!strcmp( arg[parg], "z")) index=2;

    r64 argone = atof( arg[++parg] );
    r64 argtwo = atof( arg[++parg] );
    lo = min(argone, argtwo);
    hi = max(argone, argtwo);

    n_species = atoi( arg[++parg] );
    std::vector<r32> readin;
    readin.resize( n_species );
    for (int k=0; k<n_species; k++)
        readin[k] = atof( arg[++parg] );

    if( narg != (7+n_species) ) error->all( FLERR, "Illegal fix addconf/tdpd/meso command" );

    cf.grow( n_species );
    cf.upload( &readin[0], readin.size(), meso_device->stream() );
    //printf("Add concentration flux:  lo=%g, hi=%g, cf=%g \n",lo,hi,cf);
}

int MesoFixAddConfTDPD::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_FORCE;
    return mask;
}

void MesoFixAddConfTDPD::init()
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixAddConfTDPD. %s %d\n", __FILE__, __LINE__ );
    }
}

void MesoFixAddConfTDPD::setup( int vflag )
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixAddConfTDPD. %s %d\n", __FILE__, __LINE__ );
    }
    post_force( vflag );
}

__global__ void gpu_fix_add_conf_tdpd(
    r64* __restrict coord_x,
    r64* __restrict coord_y,
    r64* __restrict coord_z,
    r32** __restrict conf,
    int* __restrict mask,
    const int index,
    const r64 lo,
    const r64 hi,
    const r32* cf,
    int n_species,
    const int groupbit,
    const int n )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
            for (int k=0; k<n_species; k++) {
                if (index==0 && coord_x[i]>lo && coord_x[i]<hi) conf[k][i] += cf[k];
                if (index==1 && coord_y[i]>lo && coord_y[i]<hi) conf[k][i] += cf[k];
                if (index==2 && coord_z[i]>lo && coord_z[i]<hi) conf[k][i] += cf[k];
            }
        }
    }
}

void MesoFixAddConfTDPD::post_force( int vflag )
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_add_conf_tdpd, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_add_conf_tdpd, cudaFuncCachePreferL1 );
    }

    gpu_fix_add_conf_tdpd <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        meso_atom->dev_coord(0), meso_atom->dev_coord(1), meso_atom->dev_coord(2),
        meso_atom->dev_CONF.ptrs(),
        meso_atom->dev_mask,
        index, lo, hi,
        cf,
        n_species,
        groupbit,
        atom->nlocal );
}
