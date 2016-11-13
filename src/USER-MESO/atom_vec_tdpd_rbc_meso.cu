#include "stdlib.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"
#include "memory.h"
#include "error.h"
#include "bond.h"
#include "force.h"
#include "update.h"

#include "meso.h"
#include "atom_meso.h"
#include "engine_meso.h"
#include "domain_meso.h"
#include "neighbor_meso.h"
#include "atom_vec_tdpd_rbc_meso.h"
#include "input.h"
#include "variable.h"
#include "comm.h"

using namespace LAMMPS_NS;

#define DELTA 10000

AtomVecTDPDRBC::AtomVecTDPDRBC( LAMMPS *lmp ) :
    AtomVecRBC( lmp ),
    AtomVecDPDMolecular( lmp ),
    dev_e_bond( lmp, "AtomVecTDPDRBC::dev_e_bond" ),
    dev_nbond( lmp, "AtomVecTDPDRBC::dev_nbond" ),
    dev_bond( lmp, "AtomVecTDPDRBC::dev_bond" ),
    dev_bond_mapped( lmp, "AtomVecTDPDRBC::dev_bond_mapped" ),
    dev_bond_r0( lmp, "AtomVecTDPDRBC::dev_bond_r0" ),
    dev_e_angle( lmp, "AtomVecTDPDRBC::dev_e_angle" ),
    dev_nangle( lmp, "AtomVecTDPDRBC::dev_nangle" ),
    dev_angle( lmp, "AtomVecTDPDRBC::dev_angle" ),
    dev_angle_mapped( lmp, "AtomVecTDPDRBC::dev_angle_mapped" ),
    dev_angle_a0( lmp, "AtomVecTDPDRBC::dev_angle_a0" ),
    dev_e_dihed( lmp, "AtomVecTDPDRBC::dev_e_dihed" ),
    dev_ndihed( lmp, "AtomVecTDPDRBC::dev_ndihed" ),
    dev_dihed_type( lmp, "AtomVecTDPDRBC::dev_dihed_type" ),
    dev_dihed( lmp, "AtomVecTDPDRBC::dev_diheds" ),
    dev_dihed_mapped( lmp, "AtomVecTDPDRBC::dev_diheds_mapped" ),
    dev_nbond_pinned( lmp, "AtomVecTDPDRBC::dev_nbond_pinned" ),
    dev_bond_atom_pinned( lmp, "AtomVecTDPDRBC::dev_bond_atom_pinned" ),
    dev_bond_type_pinned( lmp, "AtomVecTDPDRBC::dev_bond_type_pinned" ),
    dev_bond_r0_pinned( lmp, "AtomVecTDPDRBC::dev_bond_r0_pinned" ),
    dev_nangle_pinned( lmp, "AtomVecTDPDRBC::dev_nangle_pinned" ),
    dev_angle_atom1_pinned( lmp, "AtomVecTDPDRBC::dev_angle_atom1_pinned" ),
    dev_angle_atom2_pinned( lmp, "AtomVecTDPDRBC::dev_angle_atom2_pinned" ),
    dev_angle_atom3_pinned( lmp, "AtomVecTDPDRBC::dev_angle_atom3_pinned" ),
    dev_angle_type_pinned( lmp, "AtomVecTDPDRBC::dev_angle_type_pinned" ),
    dev_angle_a0_pinned( lmp, "AtomVecTDPDRBC::dev_angle_a0_pinned" ),
    dev_ndihed_pinned( lmp, "AtomVecTDPDRBC::dev_ndihed_pinned" ),
    dev_dihed_atom1_pinned( lmp, "AtomVecTDPDRBC::dev_dihed_atom1_pinned" ),
    dev_dihed_atom2_pinned( lmp, "AtomVecTDPDRBC::dev_dihed_atom2_pinned" ),
    dev_dihed_atom3_pinned( lmp, "AtomVecTDPDRBC::dev_dihed_atom3_pinned" ),
    dev_dihed_atom4_pinned( lmp, "AtomVecTDPDRBC::dev_dihed_atom4_pinned" ),
    dev_dihed_type_pinned( lmp, "AtomVecTDPDRBC::dev_dihed_type_pinned" ),
    dev_CONC( lmp, "AtomVecTDPDRBC::dev_CONC", 1 ),
    dev_CONF( lmp, "AtomVecTDPDRBC::dev_CONF", 1 ),
    dev_CONC_pinned( lmp, "AtomVecTDPDRBC::dev_CONC_pinned" ),
    dev_CONF_pinned( lmp, "AtomVecTDPDRBC::dev_CONF_pinned" ),
    n_species( 1 )
{
    int n_species_exist = input->variable->find((char*)("__restartnspecies__"));

    if (lmp->input->narg > 1) n_species = atoi(lmp->input->arg[1]);
    else if ( n_species_exist != -1 ) {
        n_species = input->variable->compute_equal(n_species_exist);
    }
    else error->all( FLERR, "Incorrect number of args for atom_style tdpd/rbc/meso");  // get number of species from input file.


    dev_CONC.set_d(n_species);              // change the dimension here.
    dev_CONF.set_d(n_species);

    cudable       = 1;
    comm_x_only    = 0;
    comm_f_only    = 0;
    mass_type      = 1;
    size_forward   = 4 + n_species;  // xyz + n_species
    size_border    = 7 + 1 + n_species;  // (xyz + type + tag + mask + molecule) + image + n_species
    size_velocity  = 3;
    size_data_atom = 6 + n_species;  // xyz + type + tag + n_species
    size_data_vel  = 4;
    xcol_data      = 4;

    pre_sort     = AtomAttribute::LOCAL  | AtomAttribute::COORD;
    post_sort    = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL | AtomAttribute::MOLE  |
                   AtomAttribute::EXCL   | AtomAttribute::BOND      | AtomAttribute::ANGLE |
                   AtomAttribute::DIHED  | AtomAttribute::CONCENT;
    pre_border   = AtomAttribute::BORDER | AtomAttribute::ESSENTIAL | AtomAttribute::MOLE  | AtomAttribute::CONCENT;
    post_border  = AtomAttribute::GHOST  | AtomAttribute::ESSENTIAL | AtomAttribute::MOLE  | AtomAttribute::CONCENT;
    pre_comm     = AtomAttribute::BORDER | AtomAttribute::COORD     | AtomAttribute::VELOC | AtomAttribute::CONCENT;
    post_comm    = AtomAttribute::GHOST  | AtomAttribute::COORD     | AtomAttribute::VELOC | AtomAttribute::CONCENT;
    pre_exchange = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL | AtomAttribute::MOLE  |
                   AtomAttribute::EXCL   | AtomAttribute::BOND      | AtomAttribute::ANGLE |
                   AtomAttribute::DIHED  | AtomAttribute::CONCENT;
    pre_output   = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL | AtomAttribute::FORCE |
                   AtomAttribute::MOLE   | AtomAttribute::EXCL      | AtomAttribute::BOND  |
                   AtomAttribute::ANGLE  | AtomAttribute::DIHED     | AtomAttribute::CONCENT;

    CONC = CONF = NULL;         // initializes to NULL so that when first time grow_CPU runs, n_species is taken care of.

}

void AtomVecTDPDRBC::grow( int n )
{
    unpin_host_array();
    if( n == 0 ) n = max( nmax + growth_inc, ( int )( nmax * growth_mul ) );
    grow_cpu( n );
    grow_device( n );
    pin_host_array();
}

void AtomVecTDPDRBC::grow_reset()
{
    AtomVecRBC::grow_reset();
    grow_exclusion();
}

void AtomVecTDPDRBC::grow_cpu( int n )
{
    CONC = memory->grow_soa( atom->CONC, n_species, n, nmax, "atom:CONC" );         // reverse, this part must be placed before AtomVecDPDAtomic::grow_cpu( n)
    CONF = memory->grow_soa( atom->CONF, n_species, n, nmax, "atom:CONF" );

    AtomVecRBC::grow( n );
}

void AtomVecTDPDRBC::grow_device( int nmax_new )
{
    AtomVecDPDMolecular::grow_device( nmax_new );

    meso_atom->dev_e_bond       = dev_e_bond     .grow( nmax_new, false, true );
    meso_atom->dev_nbond        = dev_nbond      .grow( nmax_new );
    meso_atom->dev_bond         = dev_bond       .grow( nmax_new , atom->bond_per_atom );
    meso_atom->dev_bond_mapped  = dev_bond_mapped.grow( nmax_new , atom->bond_per_atom );
    meso_atom->dev_bond_r0      = dev_bond_r0    .grow( nmax_new , atom->bond_per_atom );

    meso_atom->dev_e_angle      = dev_e_angle      .grow( nmax_new, false, true );
    meso_atom->dev_nangle       = dev_nangle       .grow( nmax_new );
    meso_atom->dev_angle        = dev_angle       .grow( nmax_new , atom->angle_per_atom );
    meso_atom->dev_angle_mapped = dev_angle_mapped.grow( nmax_new , atom->angle_per_atom );
    meso_atom->dev_angle_a0     = dev_angle_a0    .grow( nmax_new , atom->angle_per_atom );

    meso_atom->dev_e_dihed       = dev_e_dihed      .grow( nmax_new, false, true );
    meso_atom->dev_ndihed        = dev_ndihed       .grow( nmax_new );
    meso_atom->dev_dihed_type    = dev_dihed_type   .grow( nmax_new , atom->dihedral_per_atom );
    meso_atom->dev_dihed         = dev_dihed       .grow( nmax_new , atom->dihedral_per_atom );
    meso_atom->dev_dihed_mapped  = dev_dihed_mapped.grow( nmax_new , atom->dihedral_per_atom );

    // gpu global memory
    meso_atom->dev_CONC = dev_CONC.grow( nmax_new );
    meso_atom->dev_CONF = dev_CONF.grow( nmax_new );

    grow_exclusion();
}

void AtomVecTDPDRBC::copy( int i, int j, int delflag ) {
    int k;

    tag[j] = tag[i];
    type[j] = type[i];
    mask[j] = mask[i];
    image[j] = image[i];
    x[j][0] = x[i][0];
    x[j][1] = x[i][1];
    x[j][2] = x[i][2];
    v[j][0] = v[i][0];
    v[j][1] = v[i][1];
    v[j][2] = v[i][2];

    molecule[j] = molecule[i];

    num_bond[j] = num_bond[i];
    for ( k = 0; k < num_bond[j]; k++ ) {
        bond_type[j][k] = bond_type[i][k];
        bond_atom[j][k] = bond_atom[i][k];
        bond_r0[j][k] = bond_r0[i][k];
    }

    num_angle[j] = num_angle[i];
    for ( k = 0; k < num_angle[j]; k++ ) {
        angle_type[j][k] = angle_type[i][k];
        angle_atom1[j][k] = angle_atom1[i][k];
        angle_atom2[j][k] = angle_atom2[i][k];
        angle_atom3[j][k] = angle_atom3[i][k];
        angle_a0[j][k] = angle_a0[i][k];
    }

    num_dihedral[j] = num_dihedral[i];
    for ( k = 0; k < num_dihedral[j]; k++ ) {
        dihedral_type[j][k] = dihedral_type[i][k];
        dihedral_atom1[j][k] = dihedral_atom1[i][k];
        dihedral_atom2[j][k] = dihedral_atom2[i][k];
        dihedral_atom3[j][k] = dihedral_atom3[i][k];
        dihedral_atom4[j][k] = dihedral_atom4[i][k];
    }

    nspecial[j][0] = nspecial[i][0];
    nspecial[j][1] = nspecial[i][1];
    nspecial[j][2] = nspecial[i][2];
    for ( k = 0; k < nspecial[j][2]; k++ ) special[j][k] = special[i][k];

    if ( atom->nextra_grow )
        for ( int iextra = 0; iextra < atom->nextra_grow; iextra++ )
            modify->fix[atom->extra_grow[iextra]]->copy_arrays( i, j, delflag );

    for (uint k=0; k<n_species; k++) {
        CONC[k][j] = CONC[k][i];
    }

}


void AtomVecTDPDRBC::pin_host_array()
{
    AtomVecDPDMolecular::pin_host_array();

    if( atom->bond_per_atom ) {
        if( atom->num_bond  )  dev_nbond_pinned    .map_host( atom->nmax, atom->num_bond );
        if( atom->bond_atom )  dev_bond_atom_pinned.map_host( atom->nmax * atom->bond_per_atom, &( atom->bond_atom[0][0] ) );
        if( atom->bond_type )  dev_bond_type_pinned.map_host( atom->nmax * atom->bond_per_atom, &( atom->bond_type[0][0] ) );
        if( atom->bond_r0   )  dev_bond_r0_pinned  .map_host( atom->nmax * atom->bond_per_atom, &( atom->bond_r0  [0][0] ) );
    }
    if( atom->angle_per_atom ) {
        if( atom->num_angle   ) dev_nangle_pinned     .map_host( atom->nmax, atom->num_angle );
        if( atom->angle_type  ) dev_angle_type_pinned .map_host( atom->nmax * atom->angle_per_atom, &( atom->angle_type[0][0] ) );
        if( atom->angle_atom1 ) dev_angle_atom1_pinned.map_host( atom->nmax * atom->angle_per_atom, &( atom->angle_atom1[0][0] ) );
        if( atom->angle_atom2 ) dev_angle_atom2_pinned.map_host( atom->nmax * atom->angle_per_atom, &( atom->angle_atom2[0][0] ) );
        if( atom->angle_atom3 ) dev_angle_atom3_pinned.map_host( atom->nmax * atom->angle_per_atom, &( atom->angle_atom3[0][0] ) );
        if( atom->angle_a0    ) dev_angle_a0_pinned   .map_host( atom->nmax * atom->angle_per_atom, &( atom->angle_a0   [0][0] ) );
    }
    if( atom->dihedral_per_atom ) {
        if( atom->num_dihedral   ) dev_ndihed_pinned     .map_host( atom->nmax, atom->num_dihedral );
        if( atom->dihedral_type  ) dev_dihed_type_pinned .map_host( atom->nmax * atom->dihedral_per_atom, &( atom->dihedral_type[0][0] ) );
        if( atom->dihedral_atom1 ) dev_dihed_atom1_pinned.map_host( atom->nmax * atom->dihedral_per_atom, &( atom->dihedral_atom1[0][0] ) );
        if( atom->dihedral_atom2 ) dev_dihed_atom2_pinned.map_host( atom->nmax * atom->dihedral_per_atom, &( atom->dihedral_atom2[0][0] ) );
        if( atom->dihedral_atom3 ) dev_dihed_atom3_pinned.map_host( atom->nmax * atom->dihedral_per_atom, &( atom->dihedral_atom3[0][0] ) );
        if( atom->dihedral_atom4 ) dev_dihed_atom4_pinned.map_host( atom->nmax * atom->dihedral_per_atom, &( atom->dihedral_atom4[0][0] ) );
    }

    if( atom->CONC ) dev_CONC_pinned.map_host( n_species * atom->nmax, &( atom->CONC[0][0] ) );
    if( atom->CONF ) dev_CONF_pinned.map_host( n_species * atom->nmax, &( atom->CONF[0][0] ) );
}

void AtomVecTDPDRBC::unpin_host_array()
{
    AtomVecDPDMolecular::unpin_host_array();

    dev_nbond_pinned    .unmap_host( atom->num_bond );
    dev_bond_atom_pinned.unmap_host( atom->bond_atom ? & ( atom->bond_atom[0][0] ) : NULL );
    dev_bond_type_pinned.unmap_host( atom->bond_type ? & ( atom->bond_type[0][0] ) : NULL );
    dev_bond_r0_pinned  .unmap_host( atom->bond_r0   ? & ( atom->bond_r0  [0][0] ) : NULL );
    dev_nangle_pinned     .unmap_host( atom->num_angle );
    dev_angle_type_pinned .unmap_host( atom->angle_type  ? & ( atom->angle_type[0][0] ) : NULL );
    dev_angle_atom1_pinned.unmap_host( atom->angle_atom1 ? & ( atom->angle_atom1[0][0] ) : NULL );
    dev_angle_atom2_pinned.unmap_host( atom->angle_atom2 ? & ( atom->angle_atom2[0][0] ) : NULL );
    dev_angle_atom3_pinned.unmap_host( atom->angle_atom3 ? & ( atom->angle_atom3[0][0] ) : NULL );
    dev_angle_a0_pinned   .unmap_host( atom->angle_a0    ? & ( atom->angle_a0   [0][0] ) : NULL );
    dev_ndihed_pinned     .unmap_host( atom->num_dihedral );
    dev_dihed_type_pinned .unmap_host( atom->dihedral_type  ? & ( atom->dihedral_type[0][0] ) : NULL );
    dev_dihed_atom1_pinned.unmap_host( atom->dihedral_atom1 ? & ( atom->dihedral_atom1[0][0] ) : NULL );
    dev_dihed_atom2_pinned.unmap_host( atom->dihedral_atom2 ? & ( atom->dihedral_atom2[0][0] ) : NULL );
    dev_dihed_atom3_pinned.unmap_host( atom->dihedral_atom3 ? & ( atom->dihedral_atom3[0][0] ) : NULL );
    dev_dihed_atom4_pinned.unmap_host( atom->dihedral_atom4 ? & ( atom->dihedral_atom4[0][0] ) : NULL );

    dev_CONC_pinned.unmap_host( atom->CONC ? atom->CONC[0] : NULL ); ///////////// is this right?
    dev_CONF_pinned.unmap_host( atom->CONF ? atom->CONF[0] : NULL );
}


__global__ void gpu_merge_xvt(
    r64* __restrict coord_x, r64* __restrict coord_y, r64* __restrict coord_z,
    r64* __restrict veloc_x, r64* __restrict veloc_y, r64* __restrict veloc_z,
    int* __restrict type, int* __restrict tag, r64* __restrict mass,
    float4* __restrict coord_merged,
    float4* __restrict veloc_merged,
    const r64 cx, const r64 cy, const r64 cz,
    const int seed1,
    const int seed2,
    const int p_beg,
    const int p_end )
{
    for( int i  = p_beg + blockDim.x * blockIdx.x + threadIdx.x; i < p_end; i += gridDim.x * blockDim.x ) {
        float4 coord;
        coord.x = coord_x[i] - cx;
        coord.y = coord_y[i] - cy;
        coord.z = coord_z[i] - cz;
        coord.w = __int_as_float( type[i] - 1 );
        coord_merged[i] = coord;

        float4 veloc;
        veloc.x = veloc_x[i];
        veloc.y = veloc_y[i];
        veloc.z = veloc_z[i];
        veloc.w = __uint_as_float( premix_TEA<32>( __brev( tag[i] ), seed1 ) );
        veloc_merged[i] = veloc;
    }
}

void AtomVecTDPDRBC::dp2sp_merged( int seed, int p_beg, int p_end, bool offset )
{
    r64 cx = 0., cy = 0., cz = 0.;
    if( offset ) {
        cx = 0.5 * ( meso_domain->subhi[0] + meso_domain->sublo[0] );
        cy = 0.5 * ( meso_domain->subhi[1] + meso_domain->sublo[1] );
        cz = 0.5 * ( meso_domain->subhi[2] + meso_domain->sublo[2] );
    }

    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_merge_xvt, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_merge_xvt, cudaFuncCachePreferL1 );
    }

    gpu_merge_xvt <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        dev_coord(0), dev_coord(1), dev_coord(2),
        dev_veloc(0), dev_veloc(1), dev_veloc(2),
        dev_type,
        dev_tag,
        dev_mass,
        dev_coord_merged,
        dev_veloc_merged,
        cx, cy, cz,
        seed,
        seed*1664525U+1013904223U,
        p_beg, p_end );
}


// DIR = 0 : CPU -> GPU
// DIR = 1 : GPU -> CPU
template<int DIR>
__global__ void gpu_copy_bond(
    int*  __restrict host_n_bond,
    int*  __restrict host_bond_atom,
    int*  __restrict host_bond_type,
    r64*  __restrict host_bond_r0,
    int*  __restrict n_bond,
    int2* __restrict bond,
    r64*  __restrict bond_r0,
    int*  __restrict perm_table,
    const int padding_host,
    const int padding_device,
    const int p_beg,
    const int n
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if( tid < n ) {
        int i = p_beg + tid;
        if( DIR == 0 ) {
            if( perm_table ) i = perm_table[ i ];
            int n = host_n_bond[ i ];
            for( int p = 0 ; p < n ; p++ ) {
                int2 b;
                b.x = host_bond_atom[ i * padding_host + p ];
                b.y = host_bond_type[ i * padding_host + p ];
                bond[ tid + p * padding_device ] = b;
                bond_r0[ tid + p * padding_device ] = host_bond_r0[ i * padding_host + p ];
            }
            n_bond[ tid ] = n;
        } else {
            int n = n_bond[ i ];
            for( int p = 0 ; p < n ; p++ ) {
                int2 b = bond[ i + p * padding_device ];
                host_bond_atom[ i * padding_host + p ] = b.x;
                host_bond_type[ i * padding_host + p ] = b.y;
                host_bond_r0[ i * padding_host + p ] = bond_r0[ i + p * padding_device ];
            }
            host_n_bond[ i ] = n;
        }
    }
}

// DIR = 0 : CPU -> GPU
// DIR = 1 : GPU -> CPU
template<int DIR>
__global__ void gpu_copy_angle(
    int*  __restrict host_n_angle,
    int*  __restrict host_angle_atom1,
    int*  __restrict host_angle_atom2,
    int*  __restrict host_angle_atom3,
    int*  __restrict host_angle_type,
    r64*  __restrict host_angle_a0,
    int*  __restrict n_angle,
    int4* __restrict angle,
    r64*  __restrict angle_a0,
    int*  __restrict perm_table,
    const int padding_host,
    const int padding_device,
    const int padding_device_a0,
    const int p_beg,
    const int n
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if( tid < n ) {
        int i = p_beg + tid;
        if( DIR == 0 ) {
            if( perm_table ) i = perm_table[ i ];
            int n = host_n_angle[ i ];
            for( int p = 0 ; p < n ; p++ ) {
                int4 a;
                a.x = host_angle_atom1[ i * padding_host + p ];
                a.y = host_angle_atom2[ i * padding_host + p ];
                a.z = host_angle_atom3[ i * padding_host + p ];
                a.w = host_angle_type [ i * padding_host + p ];
                angle[ tid + p * padding_device ] = a;
                angle_a0[ tid + p * padding_device_a0 ] = host_angle_a0[ i * padding_host + p ];
            }
            n_angle[ tid ] = n;
        } else {
            int n = n_angle[ i ];
            for( int p = 0 ; p < n ; p++ ) {
                int4 a = angle[ i + p * padding_device ];
                host_angle_atom1[ i * padding_host + p ] = a.x;
                host_angle_atom2[ i * padding_host + p ] = a.y;
                host_angle_atom3[ i * padding_host + p ] = a.z;
                host_angle_type [ i * padding_host + p ] = a.w;
                host_angle_a0[ i * padding_host + p ] = angle_a0[ i + p * padding_device_a0 ];
            }
            host_n_angle[ i ] = n;
        }
    }
}

// DIR = 0 : CPU -> GPU
// DIR = 1 : GPU -> CPU
template<int DIR>
__global__ void gpu_copy_dihed(
    int*  __restrict host_n_dihed,
    int*  __restrict host_dihed_atom1,
    int*  __restrict host_dihed_atom2,
    int*  __restrict host_dihed_atom3,
    int*  __restrict host_dihed_atom4,
    int*  __restrict host_dihed_type,
    int*  __restrict n_dihed,
    int*  __restrict dihed_type,
    int4* __restrict diheds,
    int*  __restrict perm_table,
    const int padding_host,
    const int padding_device,
    const int padding_device_type,
    const int p_beg,
    const int n
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if( tid < n ) {
        int i = p_beg + tid;
        if( DIR == 0 ) {
            if( perm_table ) i = perm_table[ i ];
            int n = host_n_dihed[ i ];
            for( int p = 0 ; p < n ; p++ ) {
                int4 dihed;
                dihed.x = host_dihed_atom1[ i * padding_host + p ];
                dihed.y = host_dihed_atom2[ i * padding_host + p ];
                dihed.z = host_dihed_atom3[ i * padding_host + p ];
                dihed.w = host_dihed_atom4[ i * padding_host + p ];
                diheds[ tid + p * padding_device ] = dihed;
                dihed_type[ tid + p * padding_device_type ] = host_dihed_type[ i * padding_host + p ];
            }
            n_dihed[ tid ] = n;
        } else {
            int n = n_dihed[ i ];
            for( int p = 0 ; p < n ; p++ ) {
                int4 dihed = diheds[ i + p * padding_device ];
                host_dihed_atom1[ i * padding_host + p ] = dihed.x;
                host_dihed_atom2[ i * padding_host + p ] = dihed.y;
                host_dihed_atom3[ i * padding_host + p ] = dihed.z;
                host_dihed_atom4[ i * padding_host + p ] = dihed.w;
                host_dihed_type [ i * padding_host + p ] = dihed_type[ i + p * padding_device_type ];
            }
            host_n_dihed[ i ] = n;
        }
    }
}

CUDAEvent AtomVecTDPDRBC::transfer_bond( TransferDirection direction, int* permute_from, int p_beg, int n_transfer, CUDAStream stream, int action )
{
    CUDAEvent e;
    if( direction == CUDACOPY_C2G ) {
        if( action & ACTION_COPY ) dev_nbond_pinned.upload( n_transfer, stream, p_beg );
        if( action & ACTION_COPY ) dev_bond_atom_pinned.upload( n_transfer * atom->bond_per_atom, stream, p_beg * atom->bond_per_atom );
        if( action & ACTION_COPY ) dev_bond_type_pinned.upload( n_transfer * atom->bond_per_atom, stream, p_beg * atom->bond_per_atom );
        if( action & ACTION_COPY ) dev_bond_r0_pinned  .upload( n_transfer * atom->bond_per_atom, stream, p_beg * atom->bond_per_atom );
        int threads_per_block = meso_device->query_block_size( gpu_copy_bond<0> );
        if( action & ACTION_PERM )
            gpu_copy_bond<0> <<< n_block( n_transfer, threads_per_block ), threads_per_block, 0, stream >>>(
                dev_nbond_pinned.buf_d(),
                dev_bond_atom_pinned.buf_d(),
                dev_bond_type_pinned.buf_d(),
                dev_bond_r0_pinned.buf_d(),
                dev_nbond,
                dev_bond,
                dev_bond_r0,
                permute_from,
                atom->bond_per_atom,
                dev_bond.pitch_elem(),
                p_beg,
                n_transfer );
        e = meso_device->event( "AtomVecDPDBond::transfer_bond::C2G" );
        e.record( stream );
    } else {
        int threads_per_block = meso_device->query_block_size( gpu_copy_bond<1> );
        gpu_copy_bond<1> <<< n_block( n_transfer, threads_per_block ), threads_per_block, 0, stream >>>(
            dev_nbond_pinned.buf_d(),
            dev_bond_atom_pinned.buf_d(),
            dev_bond_type_pinned.buf_d(),
            dev_bond_r0_pinned.buf_d(),
            dev_nbond,
            dev_bond,
            dev_bond_r0,
            NULL,
            atom->bond_per_atom,
            dev_bond.pitch_elem(),
            p_beg,
            n_transfer );
        dev_nbond_pinned.download( n_transfer, stream, p_beg );
        dev_bond_atom_pinned.download( n_transfer * atom->bond_per_atom, stream, p_beg * atom->bond_per_atom );
        dev_bond_type_pinned.download( n_transfer * atom->bond_per_atom, stream, p_beg * atom->bond_per_atom );
        dev_bond_r0_pinned  .download( n_transfer * atom->bond_per_atom, stream, p_beg * atom->bond_per_atom );
        e = meso_device->event( "AtomVecDPDBond::transfer_bond::G2C" );
        e.record( stream );
    }
    return e;
}

CUDAEvent AtomVecTDPDRBC::transfer_angle( TransferDirection direction, int* permute_to, int p_beg, int n_transfer, CUDAStream stream, int action )
{
    CUDAEvent e;
    if( direction == CUDACOPY_C2G ) {
        if (action & ACTION_COPY) dev_nangle_pinned.upload(n_transfer, stream, p_beg);
        if (action & ACTION_COPY) dev_angle_atom1_pinned.upload(n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom);
        if (action & ACTION_COPY) dev_angle_atom2_pinned.upload(n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom);
        if (action & ACTION_COPY) dev_angle_atom3_pinned.upload(n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom);
        if (action & ACTION_COPY) dev_angle_type_pinned.upload(n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom);
        if (action & ACTION_COPY) dev_angle_a0_pinned.upload(n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom);
        int threads_per_block = meso_device->query_block_size(gpu_copy_angle<0>);
        if (action & ACTION_PERM)
            gpu_copy_angle<0> <<< n_block(n_transfer,threads_per_block), threads_per_block, 0, stream >>>(
                dev_nangle_pinned.buf_d(),
                dev_angle_atom1_pinned.buf_d(),
                dev_angle_atom2_pinned.buf_d(),
                dev_angle_atom3_pinned.buf_d(),
                dev_angle_type_pinned.buf_d(),
                dev_angle_a0_pinned.buf_d(),
                dev_nangle,
                dev_angle,
                dev_angle_a0,
                meso_atom->dev_permute_from,
                atom->angle_per_atom,
                dev_angle.pitch_elem(),
                dev_angle_a0.pitch_elem(),
                p_beg,
                n_transfer );
        e = meso_device->event("AtomVecTDPDRBC::angle::C2G");
        e.record(stream);
    } else {
        int threads_per_block = meso_device->query_block_size( gpu_copy_angle<1> );
        gpu_copy_angle<1> <<< n_block( n_transfer, threads_per_block ), threads_per_block, 0, stream >>>(
            dev_nangle_pinned.buf_d(),
            dev_angle_atom1_pinned.buf_d(),
            dev_angle_atom2_pinned.buf_d(),
            dev_angle_atom3_pinned.buf_d(),
            dev_angle_type_pinned.buf_d(),
            dev_angle_a0_pinned.buf_d(),
            dev_nangle,
            dev_angle,
            dev_angle_a0,
            NULL,
            atom->angle_per_atom,
            dev_angle.pitch_elem(),
            dev_angle_a0.pitch_elem(),
            p_beg,
            n_transfer );
        dev_nangle_pinned.download( n_transfer, stream, p_beg );
        dev_angle_atom1_pinned.download( n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom );
        dev_angle_atom2_pinned.download( n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom );
        dev_angle_atom3_pinned.download( n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom );
        dev_angle_type_pinned.download( n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom );
        dev_angle_a0_pinned.download( n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom );
        e = meso_device->event( "AtomVecTDPDRBC::angle::G2C" );
        e.record( stream );
    }
    return e;
}

CUDAEvent AtomVecTDPDRBC::transfer_dihed( TransferDirection direction, int* permute_to, int p_beg, int n_transfer, CUDAStream stream, int action )
{
    CUDAEvent e;
    if( direction == CUDACOPY_C2G ) {
        if (action & ACTION_COPY) dev_ndihed_pinned.upload(n_transfer, stream, p_beg);
        if (action & ACTION_COPY) dev_dihed_atom1_pinned.upload(n_transfer * atom->dihedral_per_atom, stream, p_beg * atom->dihedral_per_atom);
        if (action & ACTION_COPY) dev_dihed_atom2_pinned.upload(n_transfer * atom->dihedral_per_atom, stream, p_beg * atom->dihedral_per_atom);
        if (action & ACTION_COPY) dev_dihed_atom3_pinned.upload(n_transfer * atom->dihedral_per_atom, stream, p_beg * atom->dihedral_per_atom);
        if (action & ACTION_COPY) dev_dihed_atom4_pinned.upload(n_transfer * atom->dihedral_per_atom, stream, p_beg * atom->dihedral_per_atom);
        if (action & ACTION_COPY) dev_dihed_type_pinned.upload(n_transfer * atom->dihedral_per_atom, stream, p_beg * atom->dihedral_per_atom);
        int threads_per_block = meso_device->query_block_size(gpu_copy_dihed<0>);
        if (action & ACTION_PERM)
            gpu_copy_dihed<0> <<< n_block(n_transfer,threads_per_block), threads_per_block, 0, stream >>>(
                dev_ndihed_pinned.buf_d(),
                dev_dihed_atom1_pinned.buf_d(),
                dev_dihed_atom2_pinned.buf_d(),
                dev_dihed_atom3_pinned.buf_d(),
                dev_dihed_atom4_pinned.buf_d(),
                dev_dihed_type_pinned.buf_d(),
                dev_ndihed,
                dev_dihed_type,
                dev_dihed,
                meso_atom->dev_permute_from,
                atom->dihedral_per_atom,
                dev_dihed.pitch_elem(),
                dev_dihed_type.pitch_elem(),
                p_beg,
                n_transfer );
        e = meso_device->event("AtomVecTDPDRBC::dihed::C2G");
        e.record(stream);
    } else {
        int threads_per_block = meso_device->query_block_size( gpu_copy_dihed<1> );
        gpu_copy_dihed<1> <<< n_block( n_transfer, threads_per_block ), threads_per_block, 0, stream >>>(
            dev_ndihed_pinned.buf_d(),
            dev_dihed_atom1_pinned.buf_d(),
            dev_dihed_atom2_pinned.buf_d(),
            dev_dihed_atom3_pinned.buf_d(),
            dev_dihed_atom4_pinned.buf_d(),
            dev_dihed_type_pinned.buf_d(),
            dev_ndihed,
            dev_dihed_type,
            dev_dihed,
            NULL,
            atom->dihedral_per_atom,
            dev_dihed.pitch_elem(),
            dev_dihed_type.pitch_elem(),
            p_beg,
            n_transfer );
        dev_ndihed_pinned.download( n_transfer, stream, p_beg );
        dev_dihed_atom1_pinned.download( n_transfer * atom->dihedral_per_atom, stream, p_beg * atom->dihedral_per_atom );
        dev_dihed_atom2_pinned.download( n_transfer * atom->dihedral_per_atom, stream, p_beg * atom->dihedral_per_atom );
        dev_dihed_atom3_pinned.download( n_transfer * atom->dihedral_per_atom, stream, p_beg * atom->dihedral_per_atom );
        dev_dihed_atom4_pinned.download( n_transfer * atom->dihedral_per_atom, stream, p_beg * atom->dihedral_per_atom );
        dev_dihed_type_pinned.download( n_transfer * atom->dihedral_per_atom, stream, p_beg * atom->dihedral_per_atom );
        e = meso_device->event( "AtomVecTDPDRBC::dihed::G2C" );
        e.record( stream );
    }
    return e;
}

void AtomVecTDPDRBC::transfer_impl(
    std::vector<CUDAEvent> &events, AtomAttribute::Descriptor per_atom_prop, TransferDirection direction,
    int p_beg, int n_atom, int p_stream, int p_inc, int* permute_to, int* permute_from, int action, bool streamed )
{
    AtomVecDPDMolecular::transfer_impl( events, per_atom_prop, direction, p_beg, n_atom, p_stream, p_inc, permute_to, permute_from, action, streamed );
    p_stream = events.size() + p_inc;

    if( per_atom_prop & AtomAttribute::BOND ) {
        events.push_back(
            transfer_bond(
                direction, permute_from, p_beg, n_atom, meso_device->stream( p_stream += p_inc ), action ) );
    }
    if( per_atom_prop & AtomAttribute::ANGLE ) {
        events.push_back(
            transfer_scalar(
                dev_nangle_pinned, dev_nangle, direction, permute_from, p_beg, n_atom, meso_device->stream( p_stream += p_inc ), action ) );
        events.push_back(
            transfer_angle(
                direction, permute_to, p_beg, n_atom, meso_device->stream( p_stream += p_inc ), action ) );
    }
    if( per_atom_prop & AtomAttribute::DIHED ) {
        events.push_back(
            transfer_scalar(
                dev_ndihed_pinned, dev_ndihed, direction, permute_from, p_beg, n_atom, meso_device->stream( p_stream += p_inc ), action ) );
        events.push_back(
            transfer_dihed(
                direction, permute_to, p_beg, n_atom, meso_device->stream( p_stream += p_inc ), action ) );
    }
    if( per_atom_prop & AtomAttribute::CONCENT ) {
        events.push_back(
            transfer_vector(                                                    // Ansel
                dev_CONC_pinned, dev_CONC,
                direction, permute_from, p_beg, n_atom,                         // not permute_to.
                meso_device->stream( p_stream += p_inc ), action ) );
    }
    if( per_atom_prop & AtomAttribute::CONCENTF ) {
        events.push_back(
            transfer_vector(                                                    // Ansel
                dev_CONF_pinned, dev_CONF,
                direction, permute_from, p_beg, n_atom,
                meso_device->stream( p_stream += p_inc ), action ) );
    }

}

int AtomVecTDPDRBC::pack_comm( int n, int * list, double * buf,
                               int pbc_flag, int * pbc ) {
    int i, j, m;
    double dx, dy, dz;

    m = 0;
    if ( pbc_flag == 0 ) {
        for ( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0];
            buf[m++] = x[j][1];
            buf[m++] = x[j][2];
            for (int k = 0; k < n_species; k++ )
                buf[m++] = CONC[k][j];
        }
    } else {
        if ( domain->triclinic == 0 ) {
            dx = pbc[0] * domain->xprd;
            dy = pbc[1] * domain->yprd;
            dz = pbc[2] * domain->zprd;
        } else {
            dx = pbc[0] * domain->xprd + pbc[5] * domain->xy + pbc[4] * domain->xz;
            dy = pbc[1] * domain->yprd + pbc[3] * domain->yz;
            dz = pbc[2] * domain->zprd;
        }
        for ( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0] + dx;
            buf[m++] = x[j][1] + dy;
            buf[m++] = x[j][2] + dz;
            for (int k = 0; k < n_species; k++ )
                buf[m++] = CONC[k][j];
        }
    }
    return m;
}

int AtomVecTDPDRBC::pack_comm_vel( int n, int * list, double * buf,
                                   int pbc_flag, int * pbc ) {
    int i, j, m;
    double dx, dy, dz, dvx, dvy, dvz;

    m = 0;
    if ( pbc_flag == 0 ) {
        for ( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0];
            buf[m++] = x[j][1];
            buf[m++] = x[j][2];
            buf[m++] = v[j][0];
            buf[m++] = v[j][1];
            buf[m++] = v[j][2];
            for (int k = 0; k < n_species; k++ )
                buf[m++] = CONC[k][j];
            *( ( tagint * ) &buf[m++] ) = image[j];
        }
    } else {
        if ( domain->triclinic == 0 ) {
            dx = pbc[0] * domain->xprd;
            dy = pbc[1] * domain->yprd;
            dz = pbc[2] * domain->zprd;
        } else {
            dx = pbc[0] * domain->xprd + pbc[5] * domain->xy + pbc[4] * domain->xz;
            dy = pbc[1] * domain->yprd + pbc[3] * domain->yz;
            dz = pbc[2] * domain->zprd;
        }
        if ( !deform_vremap ) {
            for ( i = 0; i < n; i++ ) {
                j = list[i];
                buf[m++] = x[j][0] + dx;
                buf[m++] = x[j][1] + dy;
                buf[m++] = x[j][2] + dz;
                buf[m++] = v[j][0];
                buf[m++] = v[j][1];
                buf[m++] = v[j][2];
                for (int k = 0; k < n_species; k++ )
                    buf[m++] = CONC[k][j];

                // Change the image-count of ghost particles such that coord(unwrap(p))==coord(unwrap(ghost))   -Ansel
                // image[j] contains all 3 dimension information, in binary.
                int img = image[j];
                int ximg = (  img            & IMGMASK ) - IMGMAX;
                int yimg = ( (img>>IMGBITS)  & IMGMASK ) - IMGMAX;
                int zimg = ( (img>>IMG2BITS) & IMGMASK ) - IMGMAX;
                tagint newimg = ((tagint) (ximg-pbc[0] + IMGMAX) & IMGMASK) |
                                (((tagint) (yimg-pbc[1] + IMGMAX) & IMGMASK) << IMGBITS) |
                                (((tagint) (zimg-pbc[2] + IMGMAX) & IMGMASK) << IMG2BITS);

                *( ( tagint * ) &buf[m++] ) = newimg;
            }
        } else {
            dvx = pbc[0] * h_rate[0] + pbc[5] * h_rate[5] + pbc[4] * h_rate[4];
            dvy = pbc[1] * h_rate[1] + pbc[3] * h_rate[3];
            dvz = pbc[2] * h_rate[2];
            for ( i = 0; i < n; i++ ) {
                j = list[i];
                buf[m++] = x[j][0] + dx;
                buf[m++] = x[j][1] + dy;
                buf[m++] = x[j][2] + dz;
                if ( mask[i] & deform_groupbit ) {
                    buf[m++] = v[j][0] + dvx;
                    buf[m++] = v[j][1] + dvy;
                    buf[m++] = v[j][2] + dvz;
                } else {
                    buf[m++] = v[j][0];
                    buf[m++] = v[j][1];
                    buf[m++] = v[j][2];
                }
                for (int k = 0; k < n_species; k++ )
                    buf[m++] = CONC[k][j];

                // Change the image-count of ghost particles such that coord(unwrap(p))==coord(unwrap(ghost))   -Ansel
                // image[j] contains all 3 dimension information, in binary.
                int img = image[j];
                int ximg = (  img            & IMGMASK ) - IMGMAX;
                int yimg = ( (img>>IMGBITS)  & IMGMASK ) - IMGMAX;
                int zimg = ( (img>>IMG2BITS) & IMGMASK ) - IMGMAX;
                tagint newimg = ((tagint) (ximg-pbc[0] + IMGMAX) & IMGMASK) |
                                (((tagint) (yimg-pbc[1] + IMGMAX) & IMGMASK) << IMGBITS) |
                                (((tagint) (zimg-pbc[2] + IMGMAX) & IMGMASK) << IMG2BITS);

                *( ( tagint * ) &buf[m++] ) = newimg;
            }
        }
    }
    return m;
}

void AtomVecTDPDRBC::unpack_comm( int n, int first, double * buf ) {
    int i, m, last;

    m = 0;
    last = first + n;
    for ( i = first; i < last; i++ ) {
        x[i][0] = buf[m++];
        x[i][1] = buf[m++];
        x[i][2] = buf[m++];
        for (int k = 0; k < n_species; k++ )
            CONC[k][i] = buf[m++];
    }
}

void AtomVecTDPDRBC::unpack_comm_vel( int n, int first, double * buf ) {
    int i, m, last;

    m = 0;
    last = first + n;
    for ( i = first; i < last; i++ ) {
        x[i][0] = buf[m++];
        x[i][1] = buf[m++];
        x[i][2] = buf[m++];
        v[i][0] = buf[m++];
        v[i][1] = buf[m++];
        v[i][2] = buf[m++];
        for (int k = 0; k < n_species; k++ )
            CONC[k][i] = buf[m++];
        image[i] = *( ( tagint * ) &buf[m++] );         // over-write ghost particle image[].  Ansel
    }
}

int AtomVecTDPDRBC::pack_border( int n, int * list, double * buf,
                                 int pbc_flag, int * pbc ) {
    int i, j, m;
    double dx, dy, dz;

    m = 0;
    if ( pbc_flag == 0 ) {
        for ( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0];
            buf[m++] = x[j][1];
            buf[m++] = x[j][2];
            buf[m++] = tag[j];
            buf[m++] = type[j];
            buf[m++] = mask[j];
            buf[m++] = molecule[j];
            for (int k = 0; k < n_species; k++ )
                buf[m++] = CONC[k][j];
            *( ( tagint * ) &buf[m++] ) = image[j];
        }
    } else {
        if ( domain->triclinic == 0 ) {
            dx = pbc[0] * domain->xprd;
            dy = pbc[1] * domain->yprd;
            dz = pbc[2] * domain->zprd;
        } else {
            dx = pbc[0];
            dy = pbc[1];
            dz = pbc[2];
        }
        for ( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0] + dx;
            buf[m++] = x[j][1] + dy;
            buf[m++] = x[j][2] + dz;
            buf[m++] = tag[j];
            buf[m++] = type[j];
            buf[m++] = mask[j];
            buf[m++] = molecule[j];
            for (int k = 0; k < n_species; k++ )
                buf[m++] = CONC[k][j];

            // Ansel
            int img = image[j];
            int ximg = (  img            & IMGMASK ) - IMGMAX;
            int yimg = ( (img>>IMGBITS)  & IMGMASK ) - IMGMAX;
            int zimg = ( (img>>IMG2BITS) & IMGMASK ) - IMGMAX;
            tagint newimg = ((tagint) (ximg-pbc[0] + IMGMAX) & IMGMASK) |
                            (((tagint) (yimg-pbc[1] + IMGMAX) & IMGMASK) << IMGBITS) |
                            (((tagint) (zimg-pbc[2] + IMGMAX) & IMGMASK) << IMG2BITS);

            *( ( tagint * ) &buf[m++] ) = newimg;
        }
    }

    if ( atom->nextra_border )
        for ( int iextra = 0; iextra < atom->nextra_border; iextra++ )
            m += modify->fix[atom->extra_border[iextra]]->pack_border( n, list, &buf[m] );

    return m;
}

int AtomVecTDPDRBC::pack_border_vel( int n, int * list, double * buf,
                                     int pbc_flag, int * pbc ) {
    int i, j, m;
    double dx, dy, dz, dvx, dvy, dvz;

    m = 0;
    if ( pbc_flag == 0 ) {
        for ( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0];
            buf[m++] = x[j][1];
            buf[m++] = x[j][2];
            buf[m++] = tag[j];
            buf[m++] = type[j];
            buf[m++] = mask[j];
            buf[m++] = molecule[j];
            buf[m++] = v[j][0];
            buf[m++] = v[j][1];
            buf[m++] = v[j][2];
            for (int k = 0; k < n_species; k++ )
                buf[m++] = CONC[k][j];
            *( ( tagint * ) &buf[m++] ) = image[j];
        }
    } else {
        if ( domain->triclinic == 0 ) {
            dx = pbc[0] * domain->xprd;
            dy = pbc[1] * domain->yprd;
            dz = pbc[2] * domain->zprd;
        } else {
            dx = pbc[0];
            dy = pbc[1];
            dz = pbc[2];
        }
        if ( !deform_vremap ) {
            for ( i = 0; i < n; i++ ) {
                j = list[i];
                buf[m++] = x[j][0] + dx;
                buf[m++] = x[j][1] + dy;
                buf[m++] = x[j][2] + dz;
                buf[m++] = tag[j];
                buf[m++] = type[j];
                buf[m++] = mask[j];
                buf[m++] = molecule[j];
                buf[m++] = v[j][0];
                buf[m++] = v[j][1];
                buf[m++] = v[j][2];
                for (int k = 0; k < n_species; k++ )
                    buf[m++] = CONC[k][j];

                // Ansel
                int img = image[j];
                int ximg = (  img            & IMGMASK ) - IMGMAX;
                int yimg = ( (img>>IMGBITS)  & IMGMASK ) - IMGMAX;
                int zimg = ( (img>>IMG2BITS) & IMGMASK ) - IMGMAX;
                tagint newimg = ((tagint) (ximg-pbc[0] + IMGMAX) & IMGMASK) |
                                (((tagint) (yimg-pbc[1] + IMGMAX) & IMGMASK) << IMGBITS) |
                                (((tagint) (zimg-pbc[2] + IMGMAX) & IMGMASK) << IMG2BITS);

                *( ( tagint * ) &buf[m++] ) = newimg;
            }
        } else {
            dvx = pbc[0] * h_rate[0] + pbc[5] * h_rate[5] + pbc[4] * h_rate[4];
            dvy = pbc[1] * h_rate[1] + pbc[3] * h_rate[3];
            dvz = pbc[2] * h_rate[2];
            for ( i = 0; i < n; i++ ) {
                j = list[i];
                buf[m++] = x[j][0] + dx;
                buf[m++] = x[j][1] + dy;
                buf[m++] = x[j][2] + dz;
                buf[m++] = tag[j];
                buf[m++] = type[j];
                buf[m++] = mask[j];
                buf[m++] = molecule[j];
                if ( mask[i] & deform_groupbit ) {
                    buf[m++] = v[j][0] + dvx;
                    buf[m++] = v[j][1] + dvy;
                    buf[m++] = v[j][2] + dvz;
                } else {
                    buf[m++] = v[j][0];
                    buf[m++] = v[j][1];
                    buf[m++] = v[j][2];
                }
                for (int k = 0; k < n_species; k++ )
                    buf[m++] = CONC[k][j];

                // Ansel
                int img = image[j];
                int ximg = (  img            & IMGMASK ) - IMGMAX;
                int yimg = ( (img>>IMGBITS)  & IMGMASK ) - IMGMAX;
                int zimg = ( (img>>IMG2BITS) & IMGMASK ) - IMGMAX;
                tagint newimg = ((tagint) (ximg-pbc[0] + IMGMAX) & IMGMASK) |
                                (((tagint) (yimg-pbc[1] + IMGMAX) & IMGMASK) << IMGBITS) |
                                (((tagint) (zimg-pbc[2] + IMGMAX) & IMGMASK) << IMG2BITS);

                *( ( tagint * ) &buf[m++] ) = newimg;
            }
        }
    }

    if ( atom->nextra_border )
        for ( int iextra = 0; iextra < atom->nextra_border; iextra++ )
            m += modify->fix[atom->extra_border[iextra]]->pack_border( n, list, &buf[m] );

    return m;
}

void AtomVecTDPDRBC::unpack_border( int n, int first, double * buf ) {
    int i, m, last;

    m = 0;
    last = first + n;
    for ( i = first; i < last; i++ ) {
        if ( i == nmax ) grow( 0 );
        x[i][0] = buf[m++];
        x[i][1] = buf[m++];
        x[i][2] = buf[m++];
        tag[i] = static_cast<int> ( buf[m++] );
        type[i] = static_cast<int> ( buf[m++] );
        mask[i] = static_cast<int> ( buf[m++] );
        molecule[i] = static_cast<int> ( buf[m++] );
        for ( int k = 0; k < n_species; k++ )
            CONC[k][i] = buf[m++];
        image[i] = static_cast<int> (buf[m++]);                             // Ansel
    }

    if ( atom->nextra_border )
        for ( int iextra = 0; iextra < atom->nextra_border; iextra++ )
            m += modify->fix[atom->extra_border[iextra]]->
                 unpack_border( n, first, &buf[m] );
}

void AtomVecTDPDRBC::unpack_border_vel( int n, int first, double * buf ) {
    int i, m, last;

    m = 0;
    last = first + n;
    for ( i = first; i < last; i++ ) {
        if ( i == nmax ) grow( 0 );
        x[i][0] = buf[m++];
        x[i][1] = buf[m++];
        x[i][2] = buf[m++];
        tag[i] = static_cast<int> ( buf[m++] );
        type[i] = static_cast<int> ( buf[m++] );
        mask[i] = static_cast<int> ( buf[m++] );
        molecule[i] = static_cast<int> ( buf[m++] );
        v[i][0] = buf[m++];
        v[i][1] = buf[m++];
        v[i][2] = buf[m++];
        for ( int k = 0; k < n_species; k++ )
            CONC[k][i] = buf[m++];
        image[i] = *( ( tagint * ) &buf[m++] );                             // Ansel
    }

    if ( atom->nextra_border )
        for ( int iextra = 0; iextra < atom->nextra_border; iextra++ )
            m += modify->fix[atom->extra_border[iextra]]->
                 unpack_border( n, first, &buf[m] );
}

int AtomVecTDPDRBC::pack_exchange( int i, double * buf ) {
    int k;

    int m = 1;
    buf[m++] = x[i][0];
    buf[m++] = x[i][1];
    buf[m++] = x[i][2];
    buf[m++] = v[i][0];
    buf[m++] = v[i][1];
    buf[m++] = v[i][2];
    buf[m++] = tag[i];
    buf[m++] = type[i];
    buf[m++] = mask[i];
    buf[m] = 0.0;      // for valgrind
    *( ( tagint * ) &buf[m++] ) = image[i];
    buf[m++] = molecule[i];

    for (int k = 0; k < n_species; k++ )
        buf[m++] = CONC[k][i];

    buf[m++] = num_bond[i];
    for ( k = 0; k < num_bond[i]; k++ ) {
        buf[m++] = bond_type[i][k];
        buf[m++] = bond_atom[i][k];
        buf[m++] = bond_r0[i][k];
    }

    buf[m++] = num_angle[i];
    for ( k = 0; k < num_angle[i]; k++ ) {
        buf[m++] = angle_type[i][k];
        buf[m++] = angle_atom1[i][k];
        buf[m++] = angle_atom2[i][k];
        buf[m++] = angle_atom3[i][k];
        buf[m++] = angle_a0[i][k];
    }

    buf[m++] = num_dihedral[i];
    for ( k = 0; k < num_dihedral[i]; k++ ) {
        buf[m++] = dihedral_type[i][k];
        buf[m++] = dihedral_atom1[i][k];
        buf[m++] = dihedral_atom2[i][k];
        buf[m++] = dihedral_atom3[i][k];
        buf[m++] = dihedral_atom4[i][k];
    }

    buf[m++] = nspecial[i][0];
    buf[m++] = nspecial[i][1];
    buf[m++] = nspecial[i][2];
    for ( k = 0; k < nspecial[i][2]; k++ ) buf[m++] = special[i][k];

    if ( atom->nextra_grow )
        for ( int iextra = 0; iextra < atom->nextra_grow; iextra++ )
            m += modify->fix[atom->extra_grow[iextra]]->pack_exchange( i, &buf[m] );

    buf[0] = m;
    return m;
}

int AtomVecTDPDRBC::unpack_exchange( double * buf ) {
    int k;

    int nlocal = atom->nlocal;
    if ( nlocal == nmax ) grow( 0 );

    int m = 1;
    x[nlocal][0] = buf[m++];
    x[nlocal][1] = buf[m++];
    x[nlocal][2] = buf[m++];
    v[nlocal][0] = buf[m++];
    v[nlocal][1] = buf[m++];
    v[nlocal][2] = buf[m++];
    tag[nlocal] = static_cast<int> ( buf[m++] );
    type[nlocal] = static_cast<int> ( buf[m++] );
    mask[nlocal] = static_cast<int> ( buf[m++] );
    image[nlocal] = *( ( tagint * ) &buf[m++] );
    molecule[nlocal] = static_cast<int> ( buf[m++] );

    for (int k = 0; k < n_species; k++ )
        CONC[k][nlocal] = buf[m++];

    num_bond[nlocal] = static_cast<int> ( buf[m++] );
    for ( k = 0; k < num_bond[nlocal]; k++ ) {
        bond_type[nlocal][k] = static_cast<int> ( buf[m++] );
        bond_atom[nlocal][k] = static_cast<int> ( buf[m++] );
        bond_r0[nlocal][k] = buf[m++];
    }

    num_angle[nlocal] = static_cast<int> ( buf[m++] );
    for ( k = 0; k < num_angle[nlocal]; k++ ) {
        angle_type[nlocal][k] = static_cast<int> ( buf[m++] );
        angle_atom1[nlocal][k] = static_cast<int> ( buf[m++] );
        angle_atom2[nlocal][k] = static_cast<int> ( buf[m++] );
        angle_atom3[nlocal][k] = static_cast<int> ( buf[m++] );
        angle_a0[nlocal][k] = buf[m++];
    }

    num_dihedral[nlocal] = static_cast<int> ( buf[m++] );
    for ( k = 0; k < num_dihedral[nlocal]; k++ ) {
        dihedral_type[nlocal][k] = static_cast<int> ( buf[m++] );
        dihedral_atom1[nlocal][k] = static_cast<int> ( buf[m++] );
        dihedral_atom2[nlocal][k] = static_cast<int> ( buf[m++] );
        dihedral_atom3[nlocal][k] = static_cast<int> ( buf[m++] );
        dihedral_atom4[nlocal][k] = static_cast<int> ( buf[m++] );
    }

    nspecial[nlocal][0] = static_cast<int> ( buf[m++] );
    nspecial[nlocal][1] = static_cast<int> ( buf[m++] );
    nspecial[nlocal][2] = static_cast<int> ( buf[m++] );
    for ( k = 0; k < nspecial[nlocal][2]; k++ )
        special[nlocal][k] = static_cast<int> ( buf[m++] );

    if ( atom->nextra_grow )
        for ( int iextra = 0; iextra < atom->nextra_grow; iextra++ )
            m += modify->fix[atom->extra_grow[iextra]]->
                 unpack_exchange( nlocal, &buf[m] );

    atom->nlocal++;
    return m;
}

int AtomVecTDPDRBC::size_restart() {
    int i;

    int nlocal = atom->nlocal;
    int n = 0;
    for ( i = 0; i < nlocal; i++ )
        n += 15 + 3 * num_bond[i] + 5 * num_angle[i] + 5 * num_dihedral[i] + n_species;

    if ( atom->nextra_restart )
        for ( int iextra = 0; iextra < atom->nextra_restart; iextra++ )
            for ( i = 0; i < nlocal; i++ )
                n += modify->fix[atom->extra_restart[iextra]]->size_restart( i );

    return n;
}

int AtomVecTDPDRBC::pack_restart( int i, double * buf ) {
    int k;

    int m = 1;
    buf[m++] = x[i][0];
    buf[m++] = x[i][1];
    buf[m++] = x[i][2];
    buf[m++] = tag[i];
    buf[m++] = type[i];
    buf[m++] = mask[i];
    *( ( tagint * ) &buf[m++] ) = image[i];
    buf[m++] = v[i][0];
    buf[m++] = v[i][1];
    buf[m++] = v[i][2];
    buf[m++] = molecule[i];

    for (k = 0; k < n_species; k++ )
        buf[m++] = CONC[k][i];

    buf[m++] = num_bond[i];
    for ( k = 0; k < num_bond[i]; k++ ) {
        buf[m++] = MAX( bond_type[i][k], -bond_type[i][k] );
        buf[m++] = bond_atom[i][k];
        buf[m++] = bond_r0[i][k];
    }

    buf[m++] = num_angle[i];
    for ( k = 0; k < num_angle[i]; k++ ) {
        buf[m++] = MAX( angle_type[i][k], -angle_type[i][k] );
        buf[m++] = angle_atom1[i][k];
        buf[m++] = angle_atom2[i][k];
        buf[m++] = angle_atom3[i][k];
        buf[m++] = angle_a0[i][k];
    }

    buf[m++] = num_dihedral[i];
    for ( k = 0; k < num_dihedral[i]; k++ ) {
        buf[m++] = MAX( dihedral_type[i][k], -dihedral_type[i][k] );
        buf[m++] = dihedral_atom1[i][k];
        buf[m++] = dihedral_atom2[i][k];
        buf[m++] = dihedral_atom3[i][k];
        buf[m++] = dihedral_atom4[i][k];
    }

    if ( atom->nextra_restart )
        for ( int iextra = 0; iextra < atom->nextra_restart; iextra++ )
            m += modify->fix[atom->extra_restart[iextra]]->pack_restart( i, &buf[m] );

    buf[0] = m;
    return m;
}

int AtomVecTDPDRBC::unpack_restart( double * buf ) {
    int k;

    int nlocal = atom->nlocal;
    if ( nlocal == nmax ) {
        grow( 0 );
        if ( atom->nextra_store )
            memory->grow( atom->extra, nmax, atom->nextra_store, "atom:extra" );
    }

    int m = 1;
    x[nlocal][0] = buf[m++];
    x[nlocal][1] = buf[m++];
    x[nlocal][2] = buf[m++];
    tag[nlocal] = static_cast<int> ( buf[m++] );
    type[nlocal] = static_cast<int> ( buf[m++] );
    mask[nlocal] = static_cast<int> ( buf[m++] );
    image[nlocal] = *( ( tagint * ) &buf[m++] );
    v[nlocal][0] = buf[m++];
    v[nlocal][1] = buf[m++];
    v[nlocal][2] = buf[m++];
    molecule[nlocal] = static_cast<int> ( buf[m++] );

    for (k = 0; k < n_species; k++ )
        CONC[k][nlocal] = buf[m++];

    num_bond[nlocal] = static_cast<int> ( buf[m++] );
    for ( k = 0; k < num_bond[nlocal]; k++ ) {
        bond_type[nlocal][k] = static_cast<int> ( buf[m++] );
        bond_atom[nlocal][k] = static_cast<int> ( buf[m++] );
        bond_r0[nlocal][k] = buf[m++];
    }

    num_angle[nlocal] = static_cast<int> ( buf[m++] );
    for ( k = 0; k < num_angle[nlocal]; k++ ) {
        angle_type[nlocal][k] = static_cast<int> ( buf[m++] );
        angle_atom1[nlocal][k] = static_cast<int> ( buf[m++] );
        angle_atom2[nlocal][k] = static_cast<int> ( buf[m++] );
        angle_atom3[nlocal][k] = static_cast<int> ( buf[m++] );
        angle_a0[nlocal][k] = buf[m++];
    }

    num_dihedral[nlocal] = static_cast<int> ( buf[m++] );
    for ( k = 0; k < num_dihedral[nlocal]; k++ ) {
        dihedral_type[nlocal][k] = static_cast<int> ( buf[m++] );
        dihedral_atom1[nlocal][k] = static_cast<int> ( buf[m++] );
        dihedral_atom2[nlocal][k] = static_cast<int> ( buf[m++] );
        dihedral_atom3[nlocal][k] = static_cast<int> ( buf[m++] );
        dihedral_atom4[nlocal][k] = static_cast<int> ( buf[m++] );
    }

    nspecial[nlocal][0] = nspecial[nlocal][1] = nspecial[nlocal][2] = 0;

    double ** extra = atom->extra;
    if ( atom->nextra_store ) {
        int size = static_cast<int> ( buf[0] ) - m;
        for ( int i = 0; i < size; i++ ) extra[nlocal][i] = buf[m++];
    }

    atom->nlocal++;
    return m;
}

void AtomVecTDPDRBC::data_atom( double * coord, tagint imagetmp, char ** values ) {
    int nlocal = atom->nlocal;
    if ( nlocal == nmax ) grow( 0 );

    tag[nlocal] = atoi( values[0] );
    if ( tag[nlocal] <= 0 )
        error->one( FLERR, "Invalid atom ID in Atoms section of data file" );

    molecule[nlocal] = atoi( values[1] );

    type[nlocal] = atoi( values[2] );
    if ( type[nlocal] <= 0 || type[nlocal] > atom->ntypes )
        error->one( FLERR, "Invalid atom type in Atoms section of data file" );

    x[nlocal][0] = coord[0];
    x[nlocal][1] = coord[1];
    x[nlocal][2] = coord[2];

    for (int k = 0; k < n_species; k++ ) {
        CONC[k][nlocal] = atof( values[6+k] );  // atom-ID molecule-ID atom-type x y z CONC
    }

    image[nlocal] = imagetmp;

    mask[nlocal] = 1;
    v[nlocal][0] = 0.0;
    v[nlocal][1] = 0.0;
    v[nlocal][2] = 0.0;
    num_bond[nlocal] = 0;
    num_angle[nlocal] = 0;
    num_dihedral[nlocal] = 0;

    atom->nlocal++;
}

bigint AtomVecTDPDRBC::memory_usage() {
    bigint bytes = 0;

    if ( atom->memcheck( "tag" ) ) bytes += memory->usage( tag, nmax );
    if ( atom->memcheck( "type" ) ) bytes += memory->usage( type, nmax );
    if ( atom->memcheck( "mask" ) ) bytes += memory->usage( mask, nmax );
    if ( atom->memcheck( "image" ) ) bytes += memory->usage( image, nmax );
    if ( atom->memcheck( "x" ) ) bytes += memory->usage( x, nmax, 3 );
    if ( atom->memcheck( "v" ) ) bytes += memory->usage( v, nmax, 3 );
    if ( atom->memcheck( "f" ) ) bytes += memory->usage( f, nmax * comm->nthreads, 3 );

    if ( atom->memcheck( "molecule" ) ) bytes += memory->usage( molecule, nmax );
    if ( atom->memcheck( "nspecial" ) ) bytes += memory->usage( nspecial, nmax, 3 );
    if ( atom->memcheck( "special" ) )
        bytes += memory->usage( special, nmax, atom->maxspecial );

    if( atom->memcheck( "CONC" ) ) bytes += nmax * n_species * sizeof( r32 );                   // Ansel


    if ( atom->memcheck( "num_bond" ) ) bytes += memory->usage( num_bond, nmax );
    if ( atom->memcheck( "bond_type" ) )
        bytes += memory->usage( bond_type, nmax, atom->bond_per_atom );
    if ( atom->memcheck( "bond_atom" ) )
        bytes += memory->usage( bond_atom, nmax, atom->bond_per_atom );
    if ( atom->memcheck( "bond_r0" ) )
        bytes += memory->usage( bond_r0, nmax, atom->bond_per_atom );

    if ( atom->memcheck( "num_angle" ) ) bytes += memory->usage( num_angle, nmax );
    if ( atom->memcheck( "angle_type" ) )
        bytes += memory->usage( angle_type, nmax, atom->angle_per_atom );
    if ( atom->memcheck( "angle_atom1" ) )
        bytes += memory->usage( angle_atom1, nmax, atom->angle_per_atom );
    if ( atom->memcheck( "angle_atom2" ) )
        bytes += memory->usage( angle_atom2, nmax, atom->angle_per_atom );
    if ( atom->memcheck( "angle_atom3" ) )
        bytes += memory->usage( angle_atom3, nmax, atom->angle_per_atom );
    if ( atom->memcheck( "angle_a0" ) )
        bytes += memory->usage( angle_a0, nmax, atom->angle_per_atom );

    if ( atom->memcheck( "num_dihedral" ) ) bytes += memory->usage( num_dihedral, nmax );
    if ( atom->memcheck( "dihedral_type" ) )
        bytes += memory->usage( dihedral_type, nmax, atom->dihedral_per_atom );
    if ( atom->memcheck( "dihedral_atom1" ) )
        bytes += memory->usage( dihedral_atom1, nmax, atom->dihedral_per_atom );
    if ( atom->memcheck( "dihedral_atom2" ) )
        bytes += memory->usage( dihedral_atom2, nmax, atom->dihedral_per_atom );
    if ( atom->memcheck( "dihedral_atom3" ) )
        bytes += memory->usage( dihedral_atom3, nmax, atom->dihedral_per_atom );
    if ( atom->memcheck( "dihedral_atom4" ) )
        bytes += memory->usage( dihedral_atom4, nmax, atom->dihedral_per_atom );

    return bytes;
}

void AtomVecTDPDRBC::force_clear( AtomAttribute::Descriptor range, int vflag )
{
    // clear force on all particles
    // newton flag is always off in MESO-MVV, so never include ghosts
    int p_beg, p_end, n_work;
    resolve_work_range( range, p_beg, p_end );
    if( meso_neighbor->includegroup ) p_end = min( p_end, meso_atom->nfirst );
    n_work = p_end - p_beg;

    dev_force.set( 0.0, meso_device->stream(), p_beg, n_work );
    dev_CONF.set( 0.0, meso_device->stream(), p_beg, n_work );
    if( vflag ) dev_virial.set( 0.0, meso_device->stream(), p_beg, n_work );
}
