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
#include "pair_mdpd_meso.h"
#include "fix_aboundary_rho_meso.h"

using namespace LAMMPS_NS;

FixArbitraryBoundaryRho::FixArbitraryBoundaryRho( LAMMPS *lmp, int narg, char **arg ) :
    Fix( lmp, narg, arg ),
    MesoPointers( lmp )
{
    cut_rho       = 0.;

    for(int i = 0 ; i < narg ; i++) {
        if (!strcmp(arg[i],"cut_rho")) {
            cut_rho = atof( arg[++i] );
            continue;
        }
    }

    r64 rc_3 = cut_rho*cut_rho*cut_rho;
    r64 rc_7 = rc_3*rc_3*cut_rho;
    rho_factor = 105.0 / (16.0 * M_PI * rc_7);


    if( cut_rho==0 )
    	error->all( FLERR, "Fix parameters are not set");
}

int FixArbitraryBoundaryRho::setmask()
{
    int mask = 0;
    mask |= FixConst::PRE_FORCE;
    return mask;
}

__global__ void gpu_compute_rho(
    texobj tex_coord, texobj tex_mass,
    r64* __restrict rho,  int* __restrict mask,
    int* __restrict pair_count,
    int* __restrict pair_table,
    const int pair_padding,
    const int n,
    const int groupbit,
    const r64 cut_rho,
    const r64 rho_factor
    )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
    	if( mask[i] & groupbit ) {
    		f3u  coord1 = tex1Dfetch<float4>( tex_coord, i ); // must fetch from texture, because shifted by domain local center
    		int  n_pair = pair_count[i];
    		int *p_pair = pair_table + ( i - __laneid() ) * pair_padding + __laneid();

    		r64 rho_i = 0.;
    		for( int p = 0; p < n_pair; p++ ) {
    			int j   = __lds( p_pair );
    			p_pair += pair_padding;
    			if( ( p & (WARPSZ - 1) ) == WARPSZ - 1 ) p_pair -= WARPSZ * pair_padding - WARPSZ;

    			f3u coord2   = tex1Dfetch<float4>( tex_coord, j );
    			r32 dx       = coord1.x - coord2.x;
    			r32 dy       = coord1.y - coord2.y;
    			r32 dz       = coord1.z - coord2.z;
    			r32 rsq      = dx * dx + dy * dy + dz * dz;
    			r32 rinv     = rsqrt( rsq );
    			r32 r        = rsq * rinv;
    			r32 h        = cut_rho;


    			// Calculate Many Body Density rho; Lucy Kernel;
    			if ( rsq < h*h && rsq >= EPSILON_SQ ) {
    				r32 wf = (h + 3.0 * r) * (h - r) * (h - r) * (h - r) * rho_factor;
    				rho_i += tex1Dfetch<r64>( tex_mass, i ) * wf;
    			}
    		}

    		rho[i] = rho_i;
//			printf("%d %09.7g \n", i, rho[i] );
    	}
    }
}

void FixArbitraryBoundaryRho::pre_force( __attribute__( ( unused ) ) int vflag )
{
    MesoNeighList *dlist = meso_neighbor->lists_device[ force->pair->list->index ];
    if( !dlist ) error->all( FLERR, "<MESO> fix aboundary/rho/meso must be used together with a pair from USER-MESO" );

    meso_atom->meso_avec->dp2sp_merged( 0, 0, atom->nlocal+atom->nghost, true );

    static GridConfig grid_cfg;
    if (!grid_cfg.x) grid_cfg = meso_device->configure_kernel( gpu_compute_rho );

    gpu_compute_rho <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
            meso_atom->tex_coord_merged,
            meso_atom->tex_mass,
            meso_atom->dev_rho,
            meso_atom->dev_mask,
            dlist->dev_pair_count_core,
            dlist->dev_pair_table,
            dlist->n_col,
            atom->nlocal,
            groupbit,
            cut_rho,
            rho_factor
    );

    // Update Ghost Particles
    std::vector<CUDAEvent> events;
    CUDAEvent prev_work = meso_device->event( "FixArbitraryBoundaryRho::gpu_compute_rho" );
    prev_work.record( meso_device->stream() );
    CUDAStream::all_waiton( prev_work );
    events = meso_atom->meso_avec->transfer( AtomAttribute::BORDER | AtomAttribute::RHO, CUDACOPY_G2C );

    if( events.size()!=0 ) {
    	// if there are some particles to transfer,
		events.back().sync();
		meso_comm->forward_comm_fix( this );

		//  there is no need to sync with any previous GPU events
		//  because this transfer only depends on data on the CPU
		//  which is surly to be ready when post_comm is called
		events = meso_atom->meso_avec->transfer( AtomAttribute::GHOST | AtomAttribute::RHO, CUDACOPY_C2G );
		meso_device->stream().waiton( events.back() );
    } else {
    	// if there are nothing to transfer, then do nothing.
    }

}

void FixArbitraryBoundaryRho::setup_pre_force( int vflag )
{
    pre_force( vflag );
}


int FixArbitraryBoundaryRho::pack_comm( int n, int *list, double *buf, int pbc_flag, int *pbc )
{
    int m = 0;
    for( int i = 0; i < n; i++ ) {
        buf[m++] = atom->rho[ list[i] ];
    }
    return 1;
}

void FixArbitraryBoundaryRho::unpack_comm( int n, int first, double *buf )
{
    int m = 0;
    int last = first + n;
    for( int i = first; i < last; i++ ) {
        atom->rho[i] = buf[m++];
    }
}

