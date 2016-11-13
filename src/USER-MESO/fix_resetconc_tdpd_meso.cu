
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
#include "fix_resetconc_tdpd_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MesoFixResetConcTDPD::MesoFixResetConcTDPD( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    MesoPointers( lmp ),
    c( lmp, "reset concentration values")
{
    begin = -1;
    nextreset = begin;
    n_species = -1;
    std::vector<r32> readin;

    for( int i = 0; i < narg; i++ )
    {
        if ( !strcmp( arg[i], "begin") )
        {
            if ( ++i >= narg ) error->all(FLERR,"Incomplete resetconc command after 'begin'");
            begin = atoi(arg[i]);
            nextreset = begin;
        }
        else if ( !strcmp( arg[i], "interval") )
        {
            if ( ++i >= narg ) error->all(FLERR,"Incomplete resetconc command after 'interval'");
            interval = atoi(arg[i]);
        }
        else if ( !strcmp( arg[i], "n_species") )
        {
            if ( ++i >= narg ) error->all(FLERR,"Incomplete resetconc command after 'n_species'");
            n_species = atoi(arg[i]);
        }
        else if ( !strcmp( arg[i], "conc") )
        {
            if ( ++i >= narg ) error->all(FLERR,"Incomplete resetconc command after 'conc'");
            readin.resize( n_species );
            for ( int k=0; k<n_species; k++ ) readin[k] = atof(arg[i++]);
            i--;
        }
    }


    if ( begin == -1 || n_species == -1 || nextreset == -1) error->all(FLERR,"Insufficient resetconc command");

    c.grow( n_species );
    c.upload( &readin[0], readin.size(), meso_device->stream() );

}

MesoFixResetConcTDPD::~MesoFixResetConcTDPD() {
}

int MesoFixResetConcTDPD::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_FORCE;
    return mask;
}

void MesoFixResetConcTDPD::init()
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixResetConcTDPD. %s %d\n", __FILE__, __LINE__ );
    }
}

void MesoFixResetConcTDPD::setup( int vflag )
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixResetConcTDPD. %s %d\n", __FILE__, __LINE__ );
    }
    post_force( vflag );
}

__global__ void gpu_fix_reset_conc_tdpd(
    r32** __restrict conc,
    int* __restrict mask,
    const r32* c,
    const int n_species,
    const int groupbit,
    const int n )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
            for (int k=0; k<n_species; k++) {
                conc[k][i] = c[k];
            }
        }
    }
}

void MesoFixResetConcTDPD::post_force( int vflag )
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_reset_conc_tdpd, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_reset_conc_tdpd, cudaFuncCachePreferL1 );
    }

    currstep = update->ntimestep;       // current timestep
    if ( nextreset == currstep ) {
        nextreset += interval;
        gpu_fix_reset_conc_tdpd <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
            meso_atom->dev_CONC.ptrs(),
            meso_atom->dev_mask,
            c,
            n_species,
            groupbit,
            atom->nlocal );
    }

}
