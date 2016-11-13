#include "mpi.h"
#include "stdio.h"
#include "string.h"
#include "force.h"
#include "update.h"
#include "respa.h"
#include "error.h"

#include "atom_vec_meso.h"
#include "fix_nve_tdpd_meso.h"
#include "engine_meso.h"
#include "atom_meso.h"
#include "comm_meso.h"

using namespace LAMMPS_NS;

FixNVETDPDMeso::FixNVETDPDMeso( LAMMPS *lmp, int narg, char **arg ) :
    Fix( lmp, narg, arg ),
    MesoPointers( lmp ),
    dtv( 0 )
{
    time_integrate = 1;

    if ( atom->CONC && atom->CONF ) {
        if ( narg < 3 ) error->all( FLERR, "Invalid nve/meso command, usage: Id group style" );
    }
}

void FixNVETDPDMeso::init()
{
    dtv = update->dt;

#ifdef LMP_MESO_LOG_L1
    if( strcmp( update->integrate_style, "respa" ) == 0 )
        fprintf( stderr, "<MESO> respa style directive detected, RESPA not supported in CUDA at present.\n" );
#endif
}

int FixNVETDPDMeso::setmask()
{
    int mask = 0;
    mask |= FixConst::INITIAL_INTEGRATE;
    mask |= FixConst::FINAL_INTEGRATE;
    return mask;
}


__global__ void gpu_fix_NVE_tdpd_integrate(
    r32** __restrict CONC,
    r32** __restrict CONF,
    uint n_species,
    int* __restrict mask,
    r64  dtv,
    const int  groupbit,
    const int  n_atom )
{
    for( int i = blockDim.x * blockIdx.x + threadIdx.x ; i < n_atom ; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
            for (int k=0; k<n_species; k++) {
                CONC[k][i] += CONF[k][i] * dtv;
                CONC[k][i] = CONC[k][i] > 0 ? CONC[k][i] : 0.0f;
            }
        }
    }
}

void FixNVETDPDMeso::initial_integrate( __attribute__( ( unused ) ) int vflag )
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_NVE_tdpd_integrate, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_NVE_tdpd_integrate, cudaFuncCachePreferL1 );
    }

    gpu_fix_NVE_tdpd_integrate <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        meso_atom->dev_CONC.ptrs(),
        meso_atom->dev_CONF.ptrs(),
        (*(meso_atom->dev_CONC)).d(),
        meso_atom->dev_mask,
        dtv,
        groupbit,
        atom->nlocal );
}


void FixNVETDPDMeso::final_integrate()
{
}

void FixNVETDPDMeso::reset_dt()
{
    dtv = update->dt;
}


