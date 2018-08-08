#include "mpi.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "atom_vec.h"
#include "update.h"
#include "force.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"
#include "modify.h"
#include "fix.h"
#include "group.h"

#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "comm_meso.h"
#include "neighbor_meso.h"
#include "neigh_list_meso.h"
#include "pair_mdpd_meso.h"

using namespace LAMMPS_NS;
using namespace MDPD_COEFFICIENTS;

MesoPairMDPD::MesoPairMDPD( LAMMPS *lmp ) : Pair( lmp ), MesoPointers( lmp ),
    dev_coefficients( lmp, "MesoPairMDPD::dev_coefficients" )
{
    split_flag  = 0;     // very important; for the sake of using correct rho which is calculated in pre_force().
    coeff_ready = false;
    random = NULL;
}

MesoPairMDPD::~MesoPairMDPD()
{
    if( allocated ) {
        memory->destroy( setflag );
        memory->destroy( cutsq );
        memory->destroy( cut );
        memory->destroy( cut_r );
        memory->destroy( A_att );
        memory->destroy( B_rep );
        memory->destroy( gamma );
        memory->destroy( sigma );
    }
}

void MesoPairMDPD::allocate()
{
    allocated = 1;
    int n = atom->ntypes;

    memory->create( setflag, n + 1, n + 1, "pair:setflag" );
    memory->create( cutsq,   n + 1, n + 1, "pair:cutsq" );
    memory->create( cut,     n + 1, n + 1, "pair:cut" );
    memory->create( cut_r,   n + 1, n + 1, "pair:cut_r" );
    memory->create( A_att,   n + 1, n + 1, "pair:A_att" );
    memory->create( B_rep,   n + 1, n + 1, "pair:B_rep" );
    memory->create( gamma,   n + 1, n + 1, "pair:gamma" );
    memory->create( sigma,   n + 1, n + 1, "pair:sigma" );

    for( int i = 1; i <= n; i++ )
        for( int j = i; j <= n; j++ )
            setflag[i][j] = 0;

    dev_coefficients.grow( n * n * n_coeff );
}

void MesoPairMDPD::prepare_coeff()
{
    if( coeff_ready ) return;
    if( !allocated ) allocate();

    int n = atom->ntypes;
    static std::vector<r64> coeff_table;
    coeff_table.resize( n * n * n_coeff );
    for( int i = 1; i <= n; i++ ) {
        for( int j = 1; j <= n; j++ ) {
            int cid = ( i - 1 ) * n + ( j - 1 );
            coeff_table[ cid * n_coeff + p_cut   ] = cut[i][j];
            coeff_table[ cid * n_coeff + p_cut_r ] = cut_r[i][j];
            coeff_table[ cid * n_coeff + p_A_att ] = A_att[i][j];
            coeff_table[ cid * n_coeff + p_B_rep ] = B_rep[i][j];
            coeff_table[ cid * n_coeff + p_gamma ] = gamma[i][j];
            coeff_table[ cid * n_coeff + p_sigma ] = sigma[i][j];
        }
    }
    dev_coefficients.upload( &coeff_table[0], coeff_table.size(), meso_device->stream() );
    coeff_ready = true;
}

template<int evflag>
__global__ void gpu_mdpd(
    texobj tex_coord, texobj tex_veloc, texobj tex_rho, texobj tex_phi,
    int* __restrict mask,
    r64* __restrict force_x,   r64* __restrict force_y,   r64* __restrict force_z,
    r64* __restrict virial_xx, r64* __restrict virial_yy, r64* __restrict virial_zz,
    r64* __restrict virial_xy, r64* __restrict virial_xz, r64* __restrict virial_yz,
    int* __restrict pair_count, int* __restrict pair_table,
    r64* __restrict e_pair,
    r64* __restrict coefficients,
    const r64 dt_inv_sqrt,
    const int pair_padding,
    const int n_type,
    const int p_beg,
    const int p_end,
    const int flag_arb_bc, const int mobile_bit, const int wall_bit )
{
    extern __shared__ r64 coeffs[];
    for( int p = threadIdx.x; p < n_type * n_type * n_coeff; p += blockDim.x )
        coeffs[p] = coefficients[p];
    __syncthreads();

    for( int iter = blockIdx.x * blockDim.x + threadIdx.x; ; iter += gridDim.x * blockDim.x ) {
        int i = ( p_beg & WARPALIGN ) + iter;
        if( i >= p_end ) break;
        if( i >= p_beg ) {
            f3u  coord1 = tex1Dfetch<float4>( tex_coord, i );
            f3u  veloc1 = tex1Dfetch<float4>( tex_veloc, i );
            r64  rho_i  = tex1Dfetch<r64>( tex_rho, i );
            r64  phi_i  = tex1Dfetch<r64>( tex_phi, i );
            int  mask_i = mask[i];
            int  n_pair = pair_count[i];
            int *p_pair = pair_table + ( i - __laneid() ) * pair_padding + __laneid();
            r64 fx   = 0., fy   = 0., fz   = 0.;
            r64 vrxx = 0., vryy = 0., vrzz = 0.;
            r64 vrxy = 0., vrxz = 0., vryz = 0.;
            r64 energy = 0.;


            for( int p = 0; p < n_pair; p++ ) {
                int j   = __lds( p_pair );
                p_pair += pair_padding;
                if( ( p & 31 ) == 31 ) p_pair -= 32 * pair_padding - 32;
                f3u coord2   	= tex1Dfetch<float4>( tex_coord, j );
                r64 dx       	= coord1.x - coord2.x;
                r64 dy       	= coord1.y - coord2.y;
                r64 dz       	= coord1.z - coord2.z;
                r64 rsq      	= dx * dx + dy * dy + dz * dz;
                r64 *coeff_ij 	= coeffs + ( coord1.i * n_type + coord2.i ) * n_coeff;
                r64 cut      	= coeff_ij[p_cut];

                // printf("%g %g %g %g %g %g\n",coeff_ij[p_cut],coeff_ij[p_cut_r],coeff_ij[p_gamma],coeff_ij[p_sigma],coeff_ij[p_A_att],coeff_ij[p_B_rep]);


                if( rsq < cut*cut && rsq >= EPSILON_SQ ) {
                    f3u veloc2   = tex1Dfetch<float4>( tex_veloc, j );
                    r64 rinv     = rsqrt( rsq );
                    r64 r        = rsq * rinv;
                    if (r < EPSILON) continue;
                    r64 dvx      = veloc1.x - veloc2.x;
                    r64 dvy      = veloc1.y - veloc2.y;
                    r64 dvz      = veloc1.z - veloc2.z;
                    r64 dot      = dx * dvx + dy * dvy + dz * dvz;
                    r64 wc       = 1.0 - r / cut;
                    r64 wc_r     = MAX(1.0 - r / coeff_ij[p_cut_r] , 0.0) ;
                    r64 wr       = wc;

                    r64 rho_j    = tex1Dfetch<r64>( tex_rho, j );
                    r64 phi_j    = tex1Dfetch<r64>( tex_phi, j );
                    int mask_j   = mask[j];
                    r64 rn       = gaussian_TEA<4>( veloc1.i > veloc2.i, veloc1.i, veloc2.i );
                    r64 A_att    = coeff_ij[p_A_att];
                    r64 B_rep    = coeff_ij[p_B_rep];
                    r64 gamma_ij = coeff_ij[p_gamma];
                    r64 sigma_ij = coeff_ij[p_sigma];

                    /* This part is for arbitrary boundary */
                    /* phi, mobile_bit, wall_bit are only called in this block */
                    if( flag_arb_bc && (mask_i & mobile_bit) && (mask_j & wall_bit) ) { // one of them is mobile, and the other has to be wall.
                		r64 ratio;
                		r64 phi = MIN(phi_i, 0.5);
                		r64 rcw = cut;
                		r64 rcw_inv = 1./rcw;
                		r64 h 		= 1 - __powd( 2.088*phi*phi*phi + 1.478*phi, 0.25);
                		h     		= MAX(h, 0.025);
                		h    	   *= rcw;
                		ratio       = 1.0 + 0.187*(rcw/h - 1.0) - 0.093*(1.0-h*rcw_inv)*(1.0-h*rcw_inv)*(1.0-h*rcw_inv);

                        sigma_ij *= sqrtf(ratio);
                        gamma_ij *= ratio;

//                        printf("i j %d %d phi[i] %g ratio %g \n",i,j,phi,ratio);
                    }


                    r64 fpair    = ( A_att * wc + B_rep * (rho_i+rho_j) * wc_r )
                    		     - ( gamma_ij * wr * wr * dot * rinv )
                    		     + ( sigma_ij * wr * rn * dt_inv_sqrt );
                    fpair       *= rinv;

                    fx += dx * fpair;
                    fy += dy * fpair;
                    fz += dz * fpair;

//                    printf("rho1 %09.7g rho2 %09.7g, xyz %g %g %g fpair %g rinv %g dxdydz %g %g %g \n",rho_i,rho_j,coord1.x,coord1.y,coord1.z,fpair,rinv,dx,dy,dz);
//                    printf("gamma_ij=%g, sigma_ij=%g, wc=%g, wc_r=%g \n",gamma_ij,sigma_ij,wc,wc_r);
//                    printf("dissi %g \n",( gamma_ij * wr * wr * dot * rinv ));

                    if( evflag ) {
                    	energy += 0.5 * A_att * coeff_ij[p_cut] * wr * wr + 0.5 * B_rep * coeff_ij[p_cut_r] * (rho_i + rho_j) * wc_r * wc_r;
                    	vrxx += dx * dx * fpair;
                    	vryy += dy * dy * fpair;
                    	vrzz += dz * dz * fpair;
                    	vrxy += dx * dy * fpair;
                    	vrxz += dx * dz * fpair;
                    	vryz += dy * dz * fpair;
                    }

                }
            }

            force_x[i] += fx;
            force_y[i] += fy;
            force_z[i] += fz;
            if( evflag ) {
                e_pair[i] = energy * 0.5;
                virial_xx[i] += vrxx * 0.5;
                virial_yy[i] += vryy * 0.5;
                virial_zz[i] += vrzz * 0.5;
                virial_xy[i] += vrxy * 0.5;
                virial_xz[i] += vrxz * 0.5;
                virial_yz[i] += vryz * 0.5;
            }
        }
    }
}

void MesoPairMDPD::compute_kernel( int eflag, int vflag, int p_beg, int p_end )
{
    if( !coeff_ready ) prepare_coeff();
    MesoNeighList *dlist = meso_neighbor->lists_device[ list->index ];

    int shared_mem_size = atom->ntypes * atom->ntypes * n_coeff * sizeof( r64 );

    if( eflag || vflag ) {
        // evaluate force, energy and virial
        static GridConfig grid_cfg = meso_device->configure_kernel( gpu_mdpd<1>, shared_mem_size );
        gpu_mdpd<1> <<< grid_cfg.x, grid_cfg.y, shared_mem_size, meso_device->stream() >>> (
            meso_atom->tex_coord_merged, meso_atom->tex_veloc_merged,
            meso_atom->tex_rho, meso_atom->tex_phi, meso_atom->dev_mask,
            meso_atom->dev_force (0), meso_atom->dev_force (1), meso_atom->dev_force (2),
            meso_atom->dev_virial(0), meso_atom->dev_virial(1), meso_atom->dev_virial(2),
            meso_atom->dev_virial(3), meso_atom->dev_virial(4), meso_atom->dev_virial(5),
            dlist->dev_pair_count_core, dlist->dev_pair_table,
            meso_atom->dev_e_pair, dev_coefficients,
            1.0 / sqrt( update->dt ), dlist->n_col,
            atom->ntypes, p_beg, p_end,
            flag_arb_bc, mobile_groupbit, wall_groupbit );
    } else {
        // evaluate force only
        static GridConfig grid_cfg = meso_device->configure_kernel( gpu_mdpd<0>, shared_mem_size );
        gpu_mdpd<0> <<< grid_cfg.x, grid_cfg.y, shared_mem_size, meso_device->stream() >>> (
            meso_atom->tex_coord_merged, meso_atom->tex_veloc_merged,
            meso_atom->tex_rho, meso_atom->tex_phi, meso_atom->dev_mask,
            meso_atom->dev_force (0), meso_atom->dev_force (1), meso_atom->dev_force (2),
            meso_atom->dev_virial(0), meso_atom->dev_virial(1), meso_atom->dev_virial(2),
            meso_atom->dev_virial(3), meso_atom->dev_virial(4), meso_atom->dev_virial(5),
            dlist->dev_pair_count_core, dlist->dev_pair_table,
            meso_atom->dev_e_pair, dev_coefficients,
            1.0 / sqrt( update->dt ), dlist->n_col,
            atom->ntypes, p_beg, p_end,
            flag_arb_bc, mobile_groupbit, wall_groupbit );
    }
}

void MesoPairMDPD::compute_bulk( int eflag, int vflag )
{
    int p_beg, p_end, c_beg, c_end;
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::BULK, p_beg, p_end );
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::LOCAL, c_beg, c_end );
    meso_atom->meso_avec->dp2sp_merged( seed_now(), c_beg, c_end, true ); // convert coordinates data to r32
    compute_kernel( eflag, vflag, p_beg, p_end );
}

void MesoPairMDPD::compute_border( int eflag, int vflag )
{
    int p_beg, p_end, c_beg, c_end;
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::BORDER, p_beg, p_end );
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::GHOST, c_beg, c_end );
    meso_atom->meso_avec->dp2sp_merged( seed_now(), c_beg, c_end, true ); // convert coordinates data to r32
    compute_kernel( eflag, vflag, p_beg, p_end );
}

void MesoPairMDPD::compute( int eflag, int vflag )
{
    int p_beg, p_end, c_beg, c_end;
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::LOCAL, p_beg, p_end );
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::ALL, c_beg, c_end );
    meso_atom->meso_avec->dp2sp_merged( seed_now(), c_beg, c_end, true ); // convert coordinates data to r32
    compute_kernel( eflag, vflag, p_beg, p_end );
}

uint MesoPairMDPD::seed_now() {
    return premix_TEA<64>( seed, update->ntimestep );
}

void MesoPairMDPD::settings( int narg, char **arg )
{
    if( narg < 3 ) error->all( FLERR, "Illegal pair_style command:\n temperature cut_global seed fluid_group wall_group, "
    		"or temperature cut_global seed " );

    temperature = atof( arg[0] );
    cut_global = atof( arg[1] );
    seed = atoi( arg[2] );
    flag_arb_bc = 0;
    if ( narg > 3) {
    	flag_arb_bc = 1;
    	if ( (mobile_group = group->find( arg[3] ) ) == -1 )
    		error->all( FLERR, "<MESO> Undefined fluid group id in pairstyle mdpd/meso" );
    	mobile_groupbit = group->bitmask[ mobile_group ];
    	if ( (wall_group = group->find( arg[4] ) ) == -1 )
    		error->all( FLERR, "<MESO> Undefined wall group id in pairstyle mdpd/meso" );
    	wall_groupbit = group->bitmask[ wall_group ];
    }

    if( random ) delete random;
    random = new RanMars( lmp, seed % 899999999 + 1 );

    // reset cutoffs that have been explicitly set
    if( allocated ) {
        for( int i = 1; i <= atom->ntypes; i++ )
            for( int j = i + 1; j <= atom->ntypes; j++ )
                if( setflag[i][j] )
                    cut[i][j] = cut_global; // cut_inv[i][j] = 1.0 / cut_global;
    }
}

void MesoPairMDPD::coeff( int narg, char **arg )
{
    if( narg != 7 ) error->all( FLERR, "Incorrect args for mdpd pair coefficients: \n itype jtype A B gamma cutA cutB" );
    if( !allocated ) allocate();

    int iarg = 0;
    int ilo, ihi, jlo, jhi;
    force->bounds( arg[iarg++], atom->ntypes, ilo, ihi );
    force->bounds( arg[iarg++], atom->ntypes, jlo, jhi );

    double a0_one 		= atof( arg[iarg++] );
    double b0_one		= atof( arg[iarg++] );
    double gamma_one 	= atof( arg[iarg++] );
    double cut_one 		= atof( arg[iarg++] );
    double cut_two 		= atof( arg[iarg++] );

    if( cut_one < cut_two ) error->all( FLERR, "Incorrect args for pair coefficients:\n cutA should be larger than cutB." );

    int count = 0;
    for( int i = ilo; i <= ihi; i++ ) {
        for( int j = MAX( jlo, i ); j <= jhi; j++ ) {
            A_att[i][j]    	= a0_one;
            B_rep[i][j]    	= b0_one;
            gamma[i][j] 	= gamma_one;
            cut[i][j]   	= cut_one;
            cut_r[i][j] 	= cut_two;
            setflag[i][j]   = 1;
            count++;
        }
    }

    coeff_ready = false;

    if( count == 0 )
        error->all( FLERR, "Incorrect args for pair coefficients" );
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void MesoPairMDPD::init_style()
{
    int i = neighbor->request( this );
    neighbor->requests[i]->cudable = 1;
    neighbor->requests[i]->newton  = 2;
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double MesoPairMDPD::init_one( int i, int j )
{
    if( setflag[i][j] == 0 )
        error->all( FLERR, "All pair coeffs are not set" );

    sigma[i][j] = sqrt(2.0*force->boltz*temperature*gamma[i][j]);

    cut[j][i]     	= cut[i][j];
    cut_r[j][i] 	= cut_r[i][j];
    A_att[j][i]    	= A_att[i][j];
    B_rep[j][i]    	= B_rep[i][j];
    gamma[j][i]   	= gamma[i][j];
    sigma[j][i]		= sigma[i][j];

    return cut[i][j];
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void MesoPairMDPD::write_restart( FILE *fp )
{
    write_restart_settings( fp );

    for( int i = 1; i <= atom->ntypes; i++ ) {
        for( int j = i; j <= atom->ntypes; j++ ) {
            fwrite( &setflag[i][j], sizeof( int ), 1, fp );
            if( setflag[i][j] ) {
                fwrite( &A_att[i][j], sizeof( double ), 1, fp );
                fwrite( &B_rep[i][j], sizeof( double ), 1, fp );
                fwrite( &gamma[i][j], sizeof( double ), 1, fp );
                fwrite( &cut[i][j], sizeof( double ), 1, fp );
                fwrite( &cut_r[i][j], sizeof( double ), 1, fp );
            }
        }
    }
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void MesoPairMDPD::read_restart( FILE *fp )
{
    read_restart_settings( fp );

    allocate();

    int i, j;
    int me = comm->me;
    for( i = 1; i <= atom->ntypes; i++ ) {
        for( j = i; j <= atom->ntypes; j++ ) {
            if( me == 0 )
                fread( &setflag[i][j], sizeof( int ), 1, fp );
            MPI_Bcast( &setflag[i][j], 1, MPI_INT, 0, world );
            if( setflag[i][j] ) {
                if( me == 0 ) {
                    fread( &A_att[i][j], sizeof( double ), 1, fp );
                    fread( &B_rep[i][j], sizeof( double ), 1, fp );
                    fread( &gamma[i][j], sizeof( double ), 1, fp );
                    fread( &cut[i][j], sizeof( double ), 1, fp );
                    fread( &cut_r[i][j], sizeof( double ), 1, fp );
                }
                MPI_Bcast( &A_att[i][j], 1, MPI_DOUBLE, 0, world );
                MPI_Bcast( &B_rep[i][j], 1, MPI_DOUBLE, 0, world );
                MPI_Bcast( &gamma[i][j], 1, MPI_DOUBLE, 0, world );
                MPI_Bcast( &cut[i][j], 1, MPI_DOUBLE, 0, world );
                MPI_Bcast( &cut_r[i][j], 1, MPI_DOUBLE, 0, world );
            }
        }
    }
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void MesoPairMDPD::write_restart_settings( FILE *fp )
{
	fwrite( &temperature, sizeof( double ), 1, fp );
    fwrite( &cut_global, sizeof( double ), 1, fp );
    fwrite( &seed, sizeof( int ), 1, fp );
    fwrite( &mobile_groupbit, sizeof(int), 1, fp );
    fwrite( &mix_flag, sizeof( int ), 1, fp );
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void MesoPairMDPD::read_restart_settings( FILE *fp )
{
    if( comm->me == 0 ) {
    	fread( &temperature, sizeof( double ), 1, fp );
        fread( &cut_global, sizeof( double ), 1, fp );
        fread( &seed, sizeof( int ), 1, fp );
        fread( &mobile_groupbit, sizeof( int ), 1, fp );
        fread( &wall_groupbit, sizeof( int ), 1, fp );
        fread( &mix_flag, sizeof( int ), 1, fp );
    }
    MPI_Bcast( &temperature, 1, MPI_DOUBLE, 0, world );
    MPI_Bcast( &cut_global, 1, MPI_DOUBLE, 0, world );
    MPI_Bcast( &seed, 1, MPI_INT, 0, world );
    MPI_Bcast( &mobile_groupbit, 1, MPI_INT, 0, world );
    MPI_Bcast( &wall_groupbit, 1, MPI_INT, 0, world );
    MPI_Bcast( &mix_flag, 1, MPI_INT, 0, world );

    if( random ) delete random;
    random = new RanMars( lmp, seed % 899999999 + 1 );
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void MesoPairMDPD::write_data( FILE *fp )
{
	for (int i = 1; i <= atom->ntypes; i++)
		fprintf(fp,"%d %g %g %g\n",i,A_att[i][i],B_rep[i][i],gamma[i][i]);
}


/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void MesoPairMDPD::read_data( FILE *fp )
{
	for (int i = 1; i <= atom->ntypes; i++)
		for (int j = i; j <= atom->ntypes; j++)
			fprintf(fp,"%d %d %g %g %g %g %g\n",i,j,A_att[i][j],B_rep[i][j],gamma[i][j],cut[i][j],cut_r[i][j]);
}


/* ---------------------------------------------------------------------- */

//double MesoPairMDPD::single( int i, int j, int itype, int jtype, double rsq,
//                             double factor_coul, double factor_dpd, double &fforce )
//{
//    double r, rinv, wr, phi;
//
//    r = sqrt( rsq );
//    if( r < EPSILON ) {
//        fforce = 0.0;
//        return 0.5 * a0[itype][jtype] * cut[itype][jtype];
//    }
//
//    rinv = 1.0 / r;
//
//    wr = 1.0 - r * cut_inv[itype][jtype];
//    fforce = a0[itype][jtype] * wr * factor_dpd * rinv;
//
//    phi = 0.5 * a0[itype][jtype] * cut[itype][jtype] * wr * wr;
//    return factor_dpd * phi;
//}
