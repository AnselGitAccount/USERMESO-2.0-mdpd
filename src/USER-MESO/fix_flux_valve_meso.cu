
#include "force.h"
#include "update.h"
#include "neigh_list.h"
#include "error.h"
#include "group.h"

#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "comm_meso.h"
#include "engine_meso.h"
#include "neighbor_meso.h"
#include "neigh_list_meso.h"
#include "fix_flux_valve_meso.h"

using namespace LAMMPS_NS;

FixFluxValve::FixFluxValve( LAMMPS *lmp, int narg, char **arg ) :
    Fix( lmp, narg, arg ),
    MesoPointers( lmp )
{
    drag   = 0;
    for(int d=0; d<3; ++d) {
        vo[d] = 0;
        hi[d] = std::numeric_limits<r64>::infinity();
        lo[d] = -hi[d];
    }

    for(int i = 0 ; i < narg ; i++) {
        if (!strcmp(arg[i],"box")) {
            for(int d = 0; d < 3; ++d) lo[d] = atof( arg[++i] );
            for(int d = 0; d < 3; ++d) hi[d] = atof( arg[++i] );
        }
        if (!strcmp(arg[i],"v0")) {
            for(int d = 0; d < 3; ++d) vo[d] = atof( arg[++i] );
            continue;
        }
        if (!strcmp(arg[i],"drag")) {
            drag = atof( arg[++i] );
            continue;
        }
    }

    if ( ( vo[0] == 0 && vo[1] == 0 && vo[2] == 0 ) || drag == 0 ) {
        std::stringstream msg;
        msg << "<MESO> Fix valve/meso usage: " << arg[2]
            << " <box double x 6: box low and high> <v0 double x 3: target velocity> <drag double>"
            << std::endl;
        error->all( __FILE__, __LINE__, msg.str().c_str() );
    }
}

int FixFluxValve::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_FORCE;
    return mask;
}

__global__ void gpu_control_flux(
    r64* __restrict coord_x,   r64* __restrict coord_y,   r64* __restrict coord_z,
    r64* __restrict veloc_x,   r64* __restrict veloc_y,   r64* __restrict veloc_z,
    r64* __restrict force_x,   r64* __restrict force_y,   r64* __restrict force_z,
    int* __restrict mask,
    const r64 vox, const r64 voy, const r64 voz,
    const r64 drag,
    const r64 lox, const r64 loy, const r64 loz,
    const r64 hix, const r64 hiy, const r64 hiz,
    const int groupbit,
    const int n )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
        if ( mask[i] & groupbit ) {
            if ( ( coord_x[i] >= lox && coord_x[i] <= hix ) &&
                    ( coord_y[i] >= loy && coord_y[i] <= hiy ) &&
                    ( coord_z[i] >= loz && coord_z[i] <= hiz ) )
            {
                if ( vox ) force_x[i] += drag * ( vox - veloc_x[i] );
                if ( voy ) force_y[i] += drag * ( voy - veloc_y[i] );
                if ( voz ) force_z[i] += drag * ( voz - veloc_z[i] );
            }
        }
    }
}

void FixFluxValve::post_force(int evflag)
{
    static GridConfig grid_cfg;
    if (!grid_cfg.x) {
        grid_cfg = meso_device->configure_kernel( gpu_control_flux );
    }

    gpu_control_flux<<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        meso_atom->dev_coord(0), meso_atom->dev_coord(1), meso_atom->dev_coord(2),
        meso_atom->dev_veloc(0), meso_atom->dev_veloc(1), meso_atom->dev_veloc(2),
        meso_atom->dev_force(0), meso_atom->dev_force(1), meso_atom->dev_force(2),
        meso_atom->dev_mask,
        vo[0], vo[1], vo[2], drag,
        lo[0], lo[1], lo[2],
        hi[0], hi[1], hi[2],
        groupbit, atom->nlocal );
}
