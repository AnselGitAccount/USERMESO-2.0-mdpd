#include "stdlib.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"
#include "memory.h"
#include "error.h"
#include "bond.h"
#include "force.h"

#include "atom_meso.h"
#include "domain_meso.h"
#include "engine_meso.h"
#include "neighbor_meso.h"
#include "atom_vec_mdpd_atomic_meso.h"
#include "input.h"

using namespace LAMMPS_NS;

AtomVecMDPDAtomic::AtomVecMDPDAtomic( LAMMPS *lmp ) :
    AtomVecDPDAtomic( lmp ),
    dev_rho( lmp, "AtomVecMDPDAtomic::dev_rho" ),
    dev_phi( lmp, "AtomVecMDPDAtomic::dev_phi" ),
    dev_rho_pinned( lmp, "AtomVecMDPDAtomic::dev_rho_pinned" ),
    dev_phi_pinned( lmp, "AtomVecMDPDAtomic::dev_phi_pinned" )
//    dev_therm_merged( lmp, "AtomVecMDPDAtomic::dev_therm_merged" )
{
    if (lmp->input->narg == 1)
    	error->all( FLERR, "For atom_style mdpd/atomic/meso, an estimation to maximum local particle density must be given. "
    			"Or input 0 for default.");
    else ested_max_particle_density = atof(lmp->input->arg[1]);

    comm_x_only    = 0;
    comm_f_only    = 0;
    mass_type      = 1;
    size_forward   = 5;   // xyz + rho + phi
    size_border    = 8;   // xyz + tag + type + mask + rho + phi
    size_velocity  = 3;
    size_data_atom = 5;   // No rho or phi in data file.
    size_data_vel  = 4;
    xcol_data      = 3;

    cudable        = 1;
    pre_sort     = AtomAttribute::LOCAL  | AtomAttribute::COORD ;
    post_sort    = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL | AtomAttribute::RHO | AtomAttribute::PHI;
    pre_border   = AtomAttribute::BORDER | AtomAttribute::ESSENTIAL | AtomAttribute::RHO | AtomAttribute::PHI;
    post_border  = AtomAttribute::GHOST  | AtomAttribute::ESSENTIAL | AtomAttribute::RHO | AtomAttribute::PHI;
    pre_comm     = AtomAttribute::BORDER | AtomAttribute::COORD     | AtomAttribute::VELOC | AtomAttribute::RHO | AtomAttribute::PHI;
    post_comm    = AtomAttribute::GHOST  | AtomAttribute::COORD     | AtomAttribute::VELOC | AtomAttribute::RHO | AtomAttribute::PHI;
    pre_exchange = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL | AtomAttribute::RHO | AtomAttribute::PHI;
    pre_output   = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL | AtomAttribute::FORCE | AtomAttribute::RHO | AtomAttribute::PHI;

    rho = NULL;
    phi = NULL;
}

void AtomVecMDPDAtomic::copy( int i, int j, int delflag )
{
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
    rho[j] = rho[i];
    phi[j] = phi[i];
}

void AtomVecMDPDAtomic::grow( int n )
{
    unpin_host_array();
    if( n == 0 ) n = max( nmax + growth_inc, ( int )( nmax * growth_mul ) );
    grow_cpu( n );
    grow_device( n );
    pin_host_array();
}

void AtomVecMDPDAtomic::grow_cpu( int n )
{
    AtomVecDPDAtomic::grow_cpu( n );

    rho = memory->grow( atom->rho, nmax, "atom:rho" );
    phi = memory->grow( atom->phi, nmax, "atom:phi" );
}

void AtomVecMDPDAtomic::grow_device( int nmax_new )
{
    AtomVecDPDAtomic::grow_device( nmax_new );

    // gpu global memory
    meso_atom->dev_rho = dev_rho.grow( nmax_new, false, true );
    meso_atom->dev_phi = dev_phi.grow( nmax_new, false, true );
//    meso_atom->dev_therm_merged = dev_therm_merged.grow( nmax_new, false, false );

    // texture
    meso_atom->tex_rho.bind( dev_rho );   // Use rho in dev_rho
    meso_atom->tex_phi.bind( dev_phi );
    meso_atom->tex_mass.bind( dev_mass );
//    meso_atom->tex_misc("therm").bind( dev_therm_merged );   // Will not use rho in tex_therm_merged
}

void AtomVecMDPDAtomic::pin_host_array()
{
    AtomVecDPDAtomic::pin_host_array();

    if( atom->rho ) dev_rho_pinned.map_host( atom->nmax, atom->rho );
    if( atom->phi ) dev_phi_pinned.map_host( atom->nmax, atom->phi );
}

void AtomVecMDPDAtomic::unpin_host_array()
{
    AtomVecDPDAtomic::unpin_host_array();

    dev_rho_pinned.unmap_host( atom->rho );
    dev_phi_pinned.unmap_host( atom->phi );
}

void AtomVecMDPDAtomic::transfer_impl(
    std::vector<CUDAEvent> &events, AtomAttribute::Descriptor per_atom_prop, TransferDirection direction,
    int p_beg, int n_atom, int p_stream, int p_inc, int* permute_to, int* permute_from, int action, bool streamed )
{
    AtomVecDPDAtomic::transfer_impl( events, per_atom_prop, direction, p_beg, n_atom, p_stream, p_inc, permute_to, permute_from, action, streamed );
    p_stream = events.size() + p_inc;

    if( per_atom_prop & AtomAttribute::RHO ) {
        events.push_back(
            transfer_scalar(
                dev_rho_pinned, dev_rho, direction, permute_from, p_beg, n_atom, meso_device->stream( p_stream += p_inc ), action ) );
    }
    if( per_atom_prop & AtomAttribute::PHI ) {
        events.push_back(
            transfer_scalar(
                dev_phi_pinned, dev_phi, direction, permute_from, p_beg, n_atom, meso_device->stream( p_stream += p_inc ), action ) );
    }
}

__global__ void gpu_merge_xvt_mdpd(
    r64* __restrict coord_x, r64* __restrict coord_y, r64* __restrict coord_z,
    r64* __restrict veloc_x, r64* __restrict veloc_y, r64* __restrict veloc_z,
    int* __restrict type, int* __restrict tag,
    r64* __restrict mass, r64* __restrict rho,
    float4* __restrict coord_merged,
    float4* __restrict veloc_merged,
//    float4* __restrict therm_merged,
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

//        float4 therm;
//        therm.x = rho[i];
//        therm.y = 1.0 / mass[i];
//        therm.w = __uint_as_float( premix_TEA<32>( tag[i], seed2 ) );
//        therm_merged[i] = therm;
    }
}

void AtomVecMDPDAtomic::dp2sp_merged( int seed, int p_beg, int p_end, bool offset )
{
    r64 cx = 0., cy = 0., cz = 0.;
    if( offset ) {
        cx = 0.5 * ( meso_domain->subhi[0] + meso_domain->sublo[0] );
        cy = 0.5 * ( meso_domain->subhi[1] + meso_domain->sublo[1] );
        cz = 0.5 * ( meso_domain->subhi[2] + meso_domain->sublo[2] );
    }

    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_merge_xvt_mdpd, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_merge_xvt_mdpd, cudaFuncCachePreferL1 );
    }

    gpu_merge_xvt_mdpd <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        dev_coord(0), dev_coord(1), dev_coord(2),
        dev_veloc(0), dev_veloc(1), dev_veloc(2),
        dev_type,
        dev_tag,
        dev_mass,
        dev_rho,
        dev_coord_merged,
        dev_veloc_merged,
//        dev_therm_merged,
        cx, cy, cz,
        seed,
        seed*1664525U+1013904223U,
        p_beg, p_end );
}

/* ---------------------------------------------------------------------- */

int AtomVecMDPDAtomic::pack_comm( int n, int *list, double *buf,
                                  int pbc_flag, int *pbc )
{
    int i, j, m;
    double dx, dy, dz;

    m = 0;
    if( pbc_flag == 0 ) {
        for( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0];
            buf[m++] = x[j][1];
            buf[m++] = x[j][2];
            buf[m++] = rho[j];
            buf[m++] = phi[j];
        }
    } else {
        if( domain->triclinic == 0 ) {
            dx = pbc[0] * domain->xprd;
            dy = pbc[1] * domain->yprd;
            dz = pbc[2] * domain->zprd;
        } else {
            dx = pbc[0] * domain->xprd + pbc[5] * domain->xy + pbc[4] * domain->xz;
            dy = pbc[1] * domain->yprd + pbc[3] * domain->yz;
            dz = pbc[2] * domain->zprd;
        }
        for( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0] + dx;
            buf[m++] = x[j][1] + dy;
            buf[m++] = x[j][2] + dz;
            buf[m++] = rho[j];
            buf[m++] = phi[j];
        }
    }
    return m;
}

int AtomVecMDPDAtomic::pack_comm_vel( int n, int *list, double *buf,
                                      int pbc_flag, int *pbc )
{
    int i, j, m;
    double dx, dy, dz;

    m = 0;
    if( pbc_flag == 0 ) {
        for( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0];
            buf[m++] = x[j][1];
            buf[m++] = x[j][2];
            buf[m++] = v[j][0];
            buf[m++] = v[j][1];
            buf[m++] = v[j][2];
            buf[m++] = rho[j];
            buf[m++] = phi[j];
        }
    } else {
        if( domain->triclinic == 0 ) {
            dx = pbc[0] * domain->xprd;
            dy = pbc[1] * domain->yprd;
            dz = pbc[2] * domain->zprd;
        } else {
            dx = pbc[0] * domain->xprd + pbc[5] * domain->xy + pbc[4] * domain->xz;
            dy = pbc[1] * domain->yprd + pbc[3] * domain->yz;
            dz = pbc[2] * domain->zprd;
        }
        for( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0] + dx;
            buf[m++] = x[j][1] + dy;
            buf[m++] = x[j][2] + dz;
            buf[m++] = v[j][0];
            buf[m++] = v[j][1];
            buf[m++] = v[j][2];
            buf[m++] = rho[j];
            buf[m++] = phi[j];
        }
    }
    return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecMDPDAtomic::unpack_comm( int n, int first, double *buf )
{
    int i, m, last;

    m = 0;
    last = first + n;
    for( i = first; i < last; i++ ) {
        x[i][0] = buf[m++];
        x[i][1] = buf[m++];
        x[i][2] = buf[m++];
        rho[i]  = buf[m++];
        phi[i]  = buf[m++];
    }
}

void AtomVecMDPDAtomic::unpack_comm_vel( int n, int first, double *buf )
{
    int i, m, last;

    m = 0;
    last = first + n;
    for( i = first; i < last; i++ ) {
        x[i][0] = buf[m++];
        x[i][1] = buf[m++];
        x[i][2] = buf[m++];
        v[i][0] = buf[m++];
        v[i][1] = buf[m++];
        v[i][2] = buf[m++];
        rho[i]  = buf[m++];
        phi[i]  = buf[m++];
    }
}

/* ---------------------------------------------------------------------- */

int AtomVecMDPDAtomic::pack_border( int n, int *list, double *buf,
                                    int pbc_flag, int *pbc )
{
    int i, j, m;
    double dx, dy, dz;

    m = 0;
    if( pbc_flag == 0 ) {
        for( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0];
            buf[m++] = x[j][1];
            buf[m++] = x[j][2];
            buf[m++] = rho[j];
            buf[m++] = phi[j];
            buf[m++] = tag[j];
            buf[m++] = type[j];
            buf[m++] = mask[j];
        }
    } else {
        if( domain->triclinic == 0 ) {
            dx = pbc[0] * domain->xprd;
            dy = pbc[1] * domain->yprd;
            dz = pbc[2] * domain->zprd;
        } else {
            dx = pbc[0];
            dy = pbc[1];
            dz = pbc[2];
        }
        for( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0] + dx;
            buf[m++] = x[j][1] + dy;
            buf[m++] = x[j][2] + dz;
            buf[m++] = rho[j];
            buf[m++] = phi[j];
            buf[m++] = tag[j];
            buf[m++] = type[j];
            buf[m++] = mask[j];
        }
    }
    return m;
}

int AtomVecMDPDAtomic::pack_border_vel( int n, int *list, double *buf,
                                        int pbc_flag, int *pbc )
{
    int i, j, m;
    double dx, dy, dz;

    m = 0;
    if( pbc_flag == 0 ) {
        for( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0];
            buf[m++] = x[j][1];
            buf[m++] = x[j][2];
            buf[m++] = v[j][0];
            buf[m++] = v[j][1];
            buf[m++] = v[j][2];
            buf[m++] = rho[j];
            buf[m++] = phi[j];
            buf[m++] = tag[j];
            buf[m++] = type[j];
            buf[m++] = mask[j];
        }
    } else {
        if( domain->triclinic == 0 ) {
            dx = pbc[0] * domain->xprd;
            dy = pbc[1] * domain->yprd;
            dz = pbc[2] * domain->zprd;
        } else {
            dx = pbc[0];
            dy = pbc[1];
            dz = pbc[2];
        }
        for( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0] + dx;
            buf[m++] = x[j][1] + dy;
            buf[m++] = x[j][2] + dz;
            buf[m++] = v[j][0];
            buf[m++] = v[j][1];
            buf[m++] = v[j][2];
            buf[m++] = rho[j];
            buf[m++] = phi[j];
            buf[m++] = tag[j];
            buf[m++] = type[j];
            buf[m++] = mask[j];
        }
    }
    return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecMDPDAtomic::unpack_border( int n, int first, double *buf )
{
    int i, m, last;

    m = 0;
    last = first + n;
    for( i = first; i < last; i++ ) {
        if( i == nmax ) grow( 0 );
        x[i][0] = buf[m++];
        x[i][1] = buf[m++];
        x[i][2] = buf[m++];
        rho[i]  = buf[m++];
        phi[i]  = buf[m++];
        tag[i]  = static_cast<int>( buf[m++] );
        type[i] = static_cast<int>( buf[m++] );
        mask[i] = static_cast<int>( buf[m++] );
    }
}

void AtomVecMDPDAtomic::unpack_border_vel( int n, int first, double *buf )
{
    int i, m, last;

    m = 0;
    last = first + n;
    for( i = first; i < last; i++ ) {
        if( i == nmax ) grow( 0 );
        x[i][0] = buf[m++];
        x[i][1] = buf[m++];
        x[i][2] = buf[m++];
        v[i][0] = buf[m++];
        v[i][1] = buf[m++];
        v[i][2] = buf[m++];
        rho[i]  = buf[m++];
        phi[i]  = buf[m++];
        tag[i]  = static_cast<int>( buf[m++] );
        type[i] = static_cast<int>( buf[m++] );
        mask[i] = static_cast<int>( buf[m++] );
    }
}

/* ----------------------------------------------------------------------
     pack data for atom I for sending to another proc
     xyz must be 1st 3 values, so comm::exchange() can test on them
------------------------------------------------------------------------- */

int AtomVecMDPDAtomic::pack_exchange( int i, double *buf )
{
    int m = 1;
    buf[m++] = x[i][0];
    buf[m++] = x[i][1];
    buf[m++] = x[i][2];
    buf[m++] = v[i][0];
    buf[m++] = v[i][1];
    buf[m++] = v[i][2];
    buf[m++] = rho[i];
    buf[m++] = phi[i];
    buf[m++] = tag[i];
    buf[m++] = type[i];
    buf[m++] = mask[i];
    buf[m++] = image[i];

    if( atom->nextra_grow )
        for( int iextra = 0; iextra < atom->nextra_grow; iextra++ )
            m += modify->fix[atom->extra_grow[iextra]]->pack_exchange( i, &buf[m] );

    buf[0] = m;
    return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMDPDAtomic::unpack_exchange( double *buf )
{
    int nlocal = atom->nlocal;
    if( nlocal == nmax ) grow( 0 );

    int m = 1;
    x[nlocal][0] = buf[m++];
    x[nlocal][1] = buf[m++];
    x[nlocal][2] = buf[m++];
    v[nlocal][0] = buf[m++];
    v[nlocal][1] = buf[m++];
    v[nlocal][2] = buf[m++];
    rho[nlocal]  = buf[m++];
    phi[nlocal]  = buf[m++];
    tag[nlocal]  = static_cast<int>( buf[m++] );
    type[nlocal] = static_cast<int>( buf[m++] );
    mask[nlocal] = static_cast<int>( buf[m++] );
    image[nlocal] = static_cast<int>( buf[m++] );

    if( atom->nextra_grow )
        for( int iextra = 0; iextra < atom->nextra_grow; iextra++ )
            m += modify->fix[atom->extra_grow[iextra]]->
                 unpack_exchange( nlocal, &buf[m] );

    atom->nlocal++;
    return m;
}

/* ----------------------------------------------------------------------
     size of restart data for all atoms owned by this proc
     include extra data stored by fixes
------------------------------------------------------------------------- */

int AtomVecMDPDAtomic::size_restart()
{
    int n = 12 * atom->nlocal;

    if( atom->nextra_restart )
        for( int iextra = 0; iextra < atom->nextra_restart; iextra++ )
            for( int i = 0; i < atom->nlocal; i++ )
                n += modify->fix[atom->extra_restart[iextra]]->size_restart( i );

    return n;
}

/* ----------------------------------------------------------------------
     pack atom I's data for restart file including extra quantities
     xyz must be 1st 3 values, so that read_restart can test on them
     molecular types may be negative, but write as positive
------------------------------------------------------------------------- */

int AtomVecMDPDAtomic::pack_restart( int i, double *buf )
{
    int m = 1;
    buf[m++] = x[i][0];
    buf[m++] = x[i][1];
    buf[m++] = x[i][2];
    buf[m++] = v[i][0];
    buf[m++] = v[i][1];
    buf[m++] = v[i][2];
    buf[m++] = rho[i];
    buf[m++] = phi[i];
    buf[m++] = tag[i];
    buf[m++] = type[i];
    buf[m++] = mask[i];
    buf[m++] = image[i];

    if( atom->nextra_restart )
        for( int iextra = 0; iextra < atom->nextra_restart; iextra++ )
            m += modify->fix[atom->extra_restart[iextra]]->pack_restart( i, &buf[m] );

    buf[0] = m;
    return m;
}

/* ----------------------------------------------------------------------
     unpack data for one atom from restart file including extra quantities
------------------------------------------------------------------------- */

int AtomVecMDPDAtomic::unpack_restart( double *buf )
{
    int nlocal = atom->nlocal;
    if( nlocal == nmax ) {
        grow( 0 );
        if( atom->nextra_store )
            atom->extra = memory->grow( atom->extra, nmax, atom->nextra_store, "atom:extra" );
    }

    int m = 1;
    x[nlocal][0] = buf[m++];
    x[nlocal][1] = buf[m++];
    x[nlocal][2] = buf[m++];
    v[nlocal][0] = buf[m++];
    v[nlocal][1] = buf[m++];
    v[nlocal][2] = buf[m++];
    rho[nlocal]  = buf[m++];
    phi[nlocal]  = buf[m++];
    tag[nlocal]  = static_cast<int>( buf[m++] );
    type[nlocal] = static_cast<int>( buf[m++] );
    mask[nlocal] = static_cast<int>( buf[m++] );
    image[nlocal] = static_cast<int>( buf[m++] );

    double **extra = atom->extra;
    if( atom->nextra_store ) {
        int size = static_cast<int>( buf[0] ) - m;
        for( int i = 0; i < size; i++ ) extra[nlocal][i] = buf[m++];
    }

    atom->nlocal++;
    return m;
}

void AtomVecMDPDAtomic::data_atom( double *coord, int imagetmp, char **values )
{
    int nlocal = atom->nlocal;
    if( nlocal == nmax ) grow( 0 );

    tag[nlocal] = atoi( values[0] );
    if( tag[nlocal] <= 0 )
        error->one( FLERR, "Invalid atom ID in Atoms section of data file" );

    type[nlocal] = atoi( values[1] );
    if( type[nlocal] <= 0 || type[nlocal] > atom->ntypes )
        error->one( FLERR, "Invalid atom type in Atoms section of data file" );

    x[nlocal][0] = coord[0];
    x[nlocal][1] = coord[1];
    x[nlocal][2] = coord[2];

    rho[nlocal]  = 0;
    phi[nlocal]  = 0;

    image[nlocal] = imagetmp;

    mask[nlocal] = 1;
    v[nlocal][0] = 0.0;
    v[nlocal][1] = 0.0;
    v[nlocal][2] = 0.0;

    atom->nlocal++;
}

/* ----------------------------------------------------------------------
     unpack hybrid quantities from one line in Atoms section of data file
     initialize other atom quantities for this sub-style
------------------------------------------------------------------------- */

//int AtomVecMDPDAtomic::data_atom_hybrid(int nlocal, char **values)
//{
//  v[nlocal][0] = 0.0;
//  v[nlocal][1] = 0.0;
//  v[nlocal][2] = 0.0;
//  T[nlocal]    = 0.0;
//
//  return 0;
//}

/* ----------------------------------------------------------------------
     return # of bytes of allocated memory
------------------------------------------------------------------------- */

bigint AtomVecMDPDAtomic::memory_usage()
{
    bigint bytes = 0;

    if( atom->memcheck( "tag" ) ) bytes += nmax * sizeof( int );
    if( atom->memcheck( "type" ) ) bytes += nmax * sizeof( int );
    if( atom->memcheck( "mask" ) ) bytes += nmax * sizeof( int );
    if( atom->memcheck( "image" ) ) bytes += nmax * sizeof( int );
    if( atom->memcheck( "x" ) ) bytes += nmax * 3 * sizeof( double );
    if( atom->memcheck( "v" ) ) bytes += nmax * 3 * sizeof( double );
    if( atom->memcheck( "f" ) ) bytes += nmax * 3 * sizeof( double );
    if( atom->memcheck( "i_vf" ) ) bytes += nmax * 6 * sizeof( double );
    if( atom->memcheck( "rho" ) ) bytes += nmax * sizeof( double );
    if( atom->memcheck( "phi" ) ) bytes += nmax * sizeof( double );

    return bytes;
}

void AtomVecMDPDAtomic::force_clear( AtomAttribute::Descriptor range, int vflag )
{
    // clear force on all particles
    // newton flag is always off in MESO-MVV, so never include ghosts
    int p_beg, p_end, n_work;
    resolve_work_range( range, p_beg, p_end );
    if( meso_neighbor->includegroup ) p_end = min( p_end, meso_atom->nfirst );
    n_work = p_end - p_beg;

    dev_force.set( 0.0, meso_device->stream(), p_beg, n_work );
    if( vflag ) dev_virial.set( 0.0, meso_device->stream(), p_beg, n_work );
}

