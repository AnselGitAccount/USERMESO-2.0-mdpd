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
    cut_rho     = 0.;
    cut_phi		= 0.;
    phi_c		= 0.;
    rho_factor	= 0.;
    phi_factor  = 0.;
    dw_factor   = 0.;

    /* Default is to ONLY compute rho.
     * If wall/mobile groups are supplies, ALSO do arbitrary boundary. */

    flag_compute_arb_bc = 0;

    for(int i = 0 ; i < narg ; i++) {
        if (!strcmp(arg[i],"cut_rho")) {
            cut_rho = atof( arg[++i] );
            continue;
        }
        if (!strcmp(arg[i],"cut_phi")) {
        	cut_phi = atof( arg[++i] );
        	continue;
        }
        if (!strcmp(arg[i],"phi_c")) {
        	phi_c = atof( arg[++i] );
        	continue;
        }
        if (!strcmp(arg[i],"rho_wall")) {
        	rho_wall = atof( arg[++i] );
        	continue;
        }
        if (!strcmp(arg[i],"wall")) {
        	if ( (wall_group = group->find( arg[++i] ) ) == -1 )
        		error->all( FLERR, "<MESO> Undefined wall group id in fix aboundary/rho/meso" );
            wall_groupbit = group->bitmask[ wall_group ];
            flag_compute_arb_bc = 1;
            continue;
        }
        if (!strcmp(arg[i],"mobile")) {
        	if ( (mobile_group = group->find( arg[++i] ) ) == -1 )
        		error->all( FLERR, "<MESO> Undefined fluid group id in fix aboundary/rho/meso" );
            mobile_groupbit = group->bitmask[ mobile_group ];
            flag_compute_arb_bc = 1;
            continue;
        }
    }

    /*  groupbit MUST BE all particles (fluid + wall).
     *  mobile_groupbit is fluid.
     *  wall_groupbit is wall.
     */

    if ( mobile_groupbit & wall_groupbit ) {
        error->warning( FLERR, "<MESO> Fix aboundary/rho/meso mobile and wall group overlap", true );
    }

    r64 rc_7 = cut_rho*cut_rho*cut_rho*cut_rho*cut_rho*cut_rho*cut_rho;
//    rho_factor = 105.0 / (16.0 * M_PI * rc_7); // For Lucy Kernel
    rho_factor = 15.0 / (2.0 * M_PI * cut_rho*cut_rho*cut_rho*cut_rho*cut_rho);

    rc_7 = cut_phi*cut_phi*cut_phi*cut_phi*cut_phi*cut_phi*cut_phi;
    phi_factor = 105.0 / (16.0 * M_PI * rho_wall * rc_7);
    dw_factor = -315.0 / ( 4.0 * M_PI * rho_wall * rc_7);


    // If ONLY compute rho.
    if( !flag_compute_arb_bc && cut_rho==0 )
    	error->all( FLERR, "<MESO> Fix aboundary/rho/meso parameters are not set properly.");

    // If do arbitrary boundary AS WELL.
    if( flag_compute_arb_bc && (cut_rho==0 | cut_phi==0 | phi_c==0 | rho_factor==0 | phi_factor==0 | dw_factor==0) )
    	error->all( FLERR, "<MESO> Fix aboundary/rho/meso parameters are not set properly.");
}

void FixArbitraryBoundaryRho::init() {
	dtv = update->dt;
	dtf = 0.5 * update->dt * force->ftm2v;
}

int FixArbitraryBoundaryRho::setmask()
{
    int mask = 0;
    mask |= FixConst::PRE_FORCE;            					// gpu_compute_rho
    if(flag_compute_arb_bc) mask |= FixConst::POST_INTEGRATE;   // gpu_aboundary
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
	// for all particles (fluid + wall).
	for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
        if ( !(mask[i] & groupbit) ) continue;

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
			r32 h        = cut_rho;


			// Calculate Many Body Density rho;
			if ( rsq < h*h && rsq >= EPSILON_SQ ) {
				r32 r  = rsq * rinv;
//				r32 wf = (h + 3.0 * r) * (h - r) * (h - r) * (h - r) * rho_factor;  // Lucy Kernel
				r32 wf = (h - r) * (h - r) * rho_factor;
				rho_i += tex1Dfetch<r64>( tex_mass, i ) * wf;
			}
		}

		rho[i] = rho_i;
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


__global__ void gpu_aboundary(
    r64* mass, int* __restrict mask,
    r64* __restrict coord_x,   r64* __restrict coord_y,   r64* __restrict coord_z,
    r64* __restrict veloc_x,   r64* __restrict veloc_y,   r64* __restrict veloc_z,
    r64* __restrict phi,
    int* __restrict pair_count, int* __restrict pair_table,
    const int pair_padding,
    const int mobile_bit,
    const int wall_bit,
    const int n,
    const r64 cut_phi,
    const r64 phi_factor,
    const r64 dw_factor,
    const r64 phi_c,
    const r64 dtf,
    const r64 dtv )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
        if ( !(mask[i] & mobile_bit) ) continue;  // if not fluid, skip the rest.

        const float3 coord1 = make_float3( coord_x[i], coord_y[i], coord_z[i] );
        const float3 v      = make_float3( veloc_x[i], veloc_y[i], veloc_z[i] );

        int n_pair       = pair_count[i];
        int *p_pair      = pair_table + ( i - __laneid() ) * pair_padding + __laneid();
        r64 phi_i        = 0.f;
        float3 nw        = make_float3( 0.f, 0.f, 0.f );

        for ( int p = 0; p < n_pair; p++ ) {
            int j   = __lds( p_pair );
            p_pair += pair_padding;
            if( ( p & (WARPSZ - 1) ) == WARPSZ - 1 ) p_pair -= WARPSZ * pair_padding - WARPSZ;

            const float3 coord2 = make_float3( coord_x[j], coord_y[j], coord_z[j] );
            const float3 dcoord = coord1 - coord2;
            r32 rsq      		= normsq(dcoord);

            // If j is a wall particle.
            if ( (mask[j] & wall_bit) && (rsq < cut_phi*cut_phi) )  {
    			// compute boundary volume fraction phi & wall-normal.
                r32 rinv     = rsqrt( rsq );
                r32 r        = MAX(rsq * rinv, EPSILON);
                r32 d        = cut_phi - r;
    			phi_i       += (cut_phi + 3.*r) * d * d * d * phi_factor;
    			nw           = nw - (r * d * d * dw_factor) * dcoord * rinv;
            }
        }
        phi[i] = phi_i;  // only for fluid particle.

        // if the particle across wall-plane: bounce-back
        if ( phi_i > phi_c ) {
            nw = normalize( nw );  // Now, nw is the unit vector.

            // calculate the position prior to Initial_Integrate -> orig_x.
            const r32 dtfm      = dtf / mass[i];
            const float3 orig_x = coord1 - v * dtfm;
            const r32 tmp  		= MAX( 0.f, dot( v, nw ) );	// remove the normal components

            // Update based on bounce-back condition.
            veloc_x[i]  = -v.x + 2 * tmp * nw.x;
            veloc_y[i]  = -v.y + 2 * tmp * nw.y;
            veloc_z[i]  = -v.z + 2 * tmp * nw.z;
            coord_x[i]  = orig_x.x + dtv * veloc_x[i];
            coord_y[i]  = orig_x.y + dtv * veloc_y[i];
            coord_z[i]  = orig_x.z + dtv * veloc_z[i];
        }
    }
}


void FixArbitraryBoundaryRho::post_integrate() {

	MesoNeighList *dlist = meso_neighbor->lists_device[ force->pair->list->index ];
	if( !dlist ) error->all( FLERR, "<MESO> fix aboundary/rho/meso must be used together with a pair from USER-MESO" );

	meso_atom->meso_avec->dp2sp_merged( 0, 0, atom->nlocal+atom->nghost, true );

	static GridConfig grid_cfg;
	if (!grid_cfg.x) grid_cfg = meso_device->configure_kernel( gpu_aboundary );

	gpu_aboundary <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
			meso_atom->dev_mass, meso_atom->dev_mask,
	        meso_atom->dev_coord(0), meso_atom->dev_coord(1), meso_atom->dev_coord(2),
	        meso_atom->dev_veloc(0), meso_atom->dev_veloc(1), meso_atom->dev_veloc(2),
            meso_atom->dev_phi,
	        dlist->dev_pair_count_core, dlist->dev_pair_table,
	        dlist->n_col,
	        mobile_groupbit,
	        wall_groupbit,
	        atom->nlocal,
	        cut_phi,
	        phi_factor,
	        dw_factor,
	        phi_c,
	        dtf,
	        dtv
	);
}


int FixArbitraryBoundaryRho::pack_comm( int n, int *list, double *buf, int pbc_flag, int *pbc )
{
	// No phi, because only rho is updated in pre_force().
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

