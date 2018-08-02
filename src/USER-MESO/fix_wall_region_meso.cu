#include "mpi.h"
#include "stdio.h"
#include "string.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "neighbor.h"
#include "domain.h"
#include "region.h"
#include "region_block.h"

#include "atom_meso.h"
#include "comm_meso.h"
#include "atom_vec_meso.h"
#include "engine_meso.h"
#include "neighbor_meso.h"
#include "fix_wall_region_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MesoFixWallRegion::MesoFixWallRegion( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    MesoPointers( lmp )
{
    if( narg < 4 ) error->all( FLERR, "Illegal fix MesoFixWallRegion command" );


    iregion = domain->find_region(arg[3]);
    if (iregion == -1)
      error->all(FLERR,"Region ID for fix wall/region does not exist");
    int n = strlen(arg[3]) + 1;
    idregion = new char[n];
    strcpy(idregion,arg[3]);

    // For now, only support block-region
    if (!strcmp("Block",arg[4])) { regstyle = 0; }

    nevery = 1;
}

MesoFixWallRegion::~MesoFixWallRegion()
{
	delete [] idregion;
}

int MesoFixWallRegion::setmask()
{
    int mask = 0;
    mask |= FixConst::PRE_EXCHANGE;
    mask |= FixConst::END_OF_STEP;
    return mask;
}

void MesoFixWallRegion::init()
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixWallRegion. %s %d\n", __FILE__, __LINE__ );
    }
}

void MesoFixWallRegion::setup( int vflag )
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixWallRegion. %s %d\n", __FILE__, __LINE__ );
    }
    post_force( vflag );
}

// bounce-forward
__global__ void gpu_fix_wall_region_block(
    r64* __restrict coord_x,
    r64* __restrict coord_y,
    r64* __restrict coord_z,
    r64* __restrict veloc_x,
    r64* __restrict veloc_y,
    r64* __restrict veloc_z,
    int* __restrict mask,
    const double3 reghi,
    const double3 reglo,
    const int groupbit,
    const int n_all )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_all; i += gridDim.x * blockDim.x ) {
    	if( mask[i] & groupbit ) {
    		if( coord_x[i] <= reglo.x ) {
    			veloc_x[i] = fabs( veloc_x[i] );
    			coord_x[i] = 2. * reglo.x - coord_x[i];
    		} else if( coord_x[i] >= reghi.x ) {
    			veloc_x[i] = -fabs( veloc_x[i] );
    			coord_x[i] = 2. * reghi.x - coord_x[i];
    		}

    		if( coord_y[i] <= reglo.y ) {
    			veloc_y[i] = fabs( veloc_y[i] );
    			coord_y[i] = 2. * reglo.y - coord_y[i];
    		} else if( coord_y[i] >= reghi.y ) {
    			veloc_y[i] = -fabs( veloc_y[i] );
    			coord_y[i] = 2. * reghi.y - coord_y[i];
    		}

    		if( coord_z[i] <= reglo.z ) {
    			veloc_z[i] = fabs( veloc_z[i] );
    			coord_z[i] = 2. * reglo.z - coord_z[i];
    		} else if( coord_z[i] >= reghi.z ) {
    			veloc_z[i] = -fabs( veloc_z[i] );
    			coord_z[i] = 2. * reghi.z - coord_z[i];
    		}
    		//if ( coord_z[i] <= boxlo.z || coord_z[i] >= boxhi.z ) {
    		//  printf("particle %d still out of box after bouncing back, z0 = %lf\n", z0);
    		//}

    	}
    }
}

void MesoFixWallRegion::bounce_back()
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_wall_region_block, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_wall_region_block, cudaFuncCachePreferL1 );
    }

   	Region* region = domain->regions[iregion];

    if (regstyle == 0) {   	// If Block regions
    	RegBlock* regblock = static_cast<RegBlock*>(region);
    	double3 hi = make_double3( regblock->xhi, regblock->yhi, regblock->zhi );
    	double3 lo = make_double3( regblock->xlo, regblock->ylo, regblock->zlo );

    	gpu_fix_wall_region_block <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
    			meso_atom->dev_coord(0), meso_atom->dev_coord(1), meso_atom->dev_coord(2),
    			meso_atom->dev_veloc(0), meso_atom->dev_veloc(1), meso_atom->dev_veloc(2),
    			meso_atom->dev_mask,
    			hi, lo,
    			groupbit,
    			atom->nlocal );
    }
}

void MesoFixWallRegion::end_of_step()
{
    bounce_back();
}

void MesoFixWallRegion::pre_exchange()
{
    bounce_back();
}

