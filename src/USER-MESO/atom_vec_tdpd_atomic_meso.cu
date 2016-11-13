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
#include "atom_vec_tdpd_atomic_meso.h"
#include "input.h"

using namespace LAMMPS_NS;

AtomVecTDPDAtomic::AtomVecTDPDAtomic( LAMMPS *lmp ) :
    AtomVecDPDAtomic( lmp ),
    dev_CONC( lmp, "AtomVecTDPDAtomic::dev_CONC", 1 ),
    dev_CONF( lmp, "AtomVecTDPDAtomic::dev_CONF", 1 ),
    dev_CONC_pinned( lmp, "AtomVecTDPDAtomic::dev_CONC_pinned" ),
    dev_CONF_pinned( lmp, "AtomVecTDPDAtomic::dev_CONF_pinned" ),
    dev_therm_merged( lmp, "AtomVecTDPDAtomic::dev_therm_merged" ),
    n_species( 1 )
{
    if (lmp->input->narg == 1) error->all( FLERR, "Incorrect number of args for atom_style tdpd/atomic/meso");
    else n_species = atoi(lmp->input->arg[1]);

    dev_CONC.set_d(n_species);
    dev_CONF.set_d(n_species);

    comm_x_only    = 0;
    comm_f_only    = 0;
    mass_type      = 1;
    size_forward   = 3 + n_species;
    size_border    = 6 + 1 + n_species;
    size_velocity  = 3;
    size_data_atom = 5 + n_species;
    size_data_vel  = 4;
    xcol_data      = 3;

    cudable        = 1;
    pre_sort     = AtomAttribute::LOCAL  | AtomAttribute::COORD;
    post_sort    = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL | AtomAttribute::CONCENT;
    pre_border   = AtomAttribute::BORDER | AtomAttribute::ESSENTIAL | AtomAttribute::CONCENT;
    post_border  = AtomAttribute::GHOST  | AtomAttribute::ESSENTIAL | AtomAttribute::CONCENT;
    pre_comm     = AtomAttribute::BORDER | AtomAttribute::COORD     | AtomAttribute::VELOC | AtomAttribute::CONCENT;
    post_comm    = AtomAttribute::GHOST  | AtomAttribute::COORD     | AtomAttribute::VELOC | AtomAttribute::CONCENT;
    pre_exchange = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL | AtomAttribute::CONCENT;
    pre_output   = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL | AtomAttribute::CONCENT | AtomAttribute::FORCE ;

    CONC = CONF = NULL;         // initializes to NULL so that when first time grow_CPU runs, n_species is taken care of.
}

void AtomVecTDPDAtomic::copy( int i, int j, int delflag )
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
    for (uint k=0; k<n_species; k++) {
        CONC[k][j] = CONC[k][i];
    }
}

void AtomVecTDPDAtomic::grow( int n )
{
    unpin_host_array();
    if( n == 0 ) n = max( nmax + growth_inc, ( int )( nmax * growth_mul ) );
    grow_cpu( n );
    grow_device( n );
    pin_host_array();
}

void AtomVecTDPDAtomic::grow_cpu( int n )
{
    CONC = memory->grow_soa( atom->CONC, n_species, n, nmax, "atom:CONC" );
    CONF = memory->grow_soa( atom->CONF, n_species, n, nmax, "atom:CONF" );

    AtomVecDPDAtomic::grow_cpu( n );
}

void AtomVecTDPDAtomic::grow_device( int nmax_new )
{
    AtomVecDPDAtomic::grow_device( nmax_new );

    meso_atom->dev_CONC = dev_CONC.grow( nmax_new );
    meso_atom->dev_CONF = dev_CONF.grow( nmax_new );
    meso_atom->dev_therm_merged = dev_therm_merged.grow( nmax_new, false, false );

    meso_atom->tex_misc("therm").bind( dev_therm_merged );
}

void AtomVecTDPDAtomic::pin_host_array()
{
    AtomVecDPDAtomic::pin_host_array();

    if( atom->CONC ) dev_CONC_pinned.map_host( n_species * atom->nmax, &( atom->CONC[0][0] ) );
    if( atom->CONF ) dev_CONF_pinned.map_host( n_species * atom->nmax, &( atom->CONF[0][0] ) );

}

void AtomVecTDPDAtomic::unpin_host_array()
{
    AtomVecDPDAtomic::unpin_host_array();

    dev_CONC_pinned.unmap_host( atom->CONC ? atom->CONC[0] : NULL );
    dev_CONF_pinned.unmap_host( atom->CONF ? atom->CONF[0] : NULL );
}

void AtomVecTDPDAtomic::transfer_impl(
    std::vector<CUDAEvent> &events, AtomAttribute::Descriptor per_atom_prop, TransferDirection direction,
    int p_beg, int n_atom, int p_stream, int p_inc, int* permute_to, int* permute_from, int action, bool streamed )
{
    AtomVecDPDAtomic::transfer_impl( events, per_atom_prop, direction, p_beg, n_atom, p_stream, p_inc, permute_to, permute_from, action, streamed );
    p_stream = events.size() + p_inc;

    if( per_atom_prop & AtomAttribute::CONCENT ) {
        events.push_back(
            transfer_vector(
                dev_CONC_pinned, dev_CONC,
                direction, permute_from, p_beg, n_atom,
                meso_device->stream( p_stream += p_inc ), action ) );
    }
    if( per_atom_prop & AtomAttribute::CONCENTF ) {
        events.push_back(
            transfer_vector(
                dev_CONF_pinned, dev_CONF,
                direction, permute_from, p_beg, n_atom,
                meso_device->stream( p_stream += p_inc ), action ) );
    }

}

__global__ void gpu_merge_xvt(
    r64* __restrict coord_x, r64* __restrict coord_y, r64* __restrict coord_z,
    r64* __restrict veloc_x, r64* __restrict veloc_y, r64* __restrict veloc_z,
    int* __restrict type, int* __restrict tag, r64* __restrict mass,
    float4* __restrict coord_merged,
    float4* __restrict veloc_merged,
    float4* __restrict therm_merged,
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

        float4 therm;
        therm.y = 1.0 / mass[i];
        therm.w = __uint_as_float( premix_TEA<32>( tag[i], seed2 ) );
        therm_merged[i] = therm;
    }
}

void AtomVecTDPDAtomic::dp2sp_merged( int seed, int p_beg, int p_end, bool offset )
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
        dev_therm_merged,
        cx, cy, cz,
        seed,
        seed*1664525U+1013904223U,
        p_beg, p_end );
}

/* ---------------------------------------------------------------------- */

int AtomVecTDPDAtomic::pack_comm( int n, int *list, double *buf,
                                  int pbc_flag, int *pbc )
{
    int i, j, k, m;
    double dx, dy, dz;

    m = 0;
    if( pbc_flag == 0 ) {
        for( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0];
            buf[m++] = x[j][1];
            buf[m++] = x[j][2];
            for ( k = 0; k < n_species; k++ )
                buf[m++] = CONC[k][j];
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
            for ( k = 0; k < n_species; k++ )
                buf[m++] = CONC[k][j];
        }
    }
    return m;
}


int AtomVecTDPDAtomic::pack_comm_vel( int n, int *list, double *buf,
                                      int pbc_flag, int *pbc )
{
    int i, j, k, m;
    double dx, dy, dz;
    double phantom;

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
        }
        for( k = 0; k < n_species; k++ ) {
            for ( i = 0; i < n; i++ ) {
                j = list[i];
                buf[m++] = CONC[k][j];
            }
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
        }
        for( k = 0; k < n_species; k++ ) {
            for ( i = 0; i < n; i++ ) {
                j = list[i];
                buf[m++] = CONC[k][j];
            }
        }
    }
    return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecTDPDAtomic::unpack_comm( int n, int first, double *buf )
{
    int i, k, m, last;

    m = 0;
    last = first + n;
    for( i = first; i < last; i++ ) {
        x[i][0] = buf[m++];
        x[i][1] = buf[m++];
        x[i][2] = buf[m++];
        for ( k = 0; k < n_species; k++ )
            CONC[k][i] = buf[m++];
    }
}

void AtomVecTDPDAtomic::unpack_comm_vel( int n, int first, double *buf )
{
    int i, k, m, last;
    double phantom;

    m = 0;
    last = first + n;
    for( i = first; i < last; i++ ) {
        x[i][0] = buf[m++];
        x[i][1] = buf[m++];
        x[i][2] = buf[m++];
        v[i][0] = buf[m++];
        v[i][1] = buf[m++];
        v[i][2] = buf[m++];
    }
    for( k = 0; k < n_species; k++ ) {
        for ( i = first; i < last; i++ ) {
            CONC[k][i] = buf[m++];
        }
    }
}

/* ---------------------------------------------------------------------- */

int AtomVecTDPDAtomic::pack_border( int n, int *list, double *buf,
                                    int pbc_flag, int *pbc )
{
    int i, j, k, m;
    double dx, dy, dz;

    m = 0;
    if( pbc_flag == 0 ) {
        for( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0];
            buf[m++] = x[j][1];
            buf[m++] = x[j][2];
            for ( k = 0; k < n_species; k++ )
                buf[m++] = CONC[k][j];
            buf[m++] = tag[j];
            buf[m++] = type[j];
            buf[m++] = mask[j];
            buf[m++] = image[j];
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
            for ( k = 0; k < n_species; k++ )
                buf[m++] = CONC[k][j];
            buf[m++] = tag[j];
            buf[m++] = type[j];
            buf[m++] = mask[j];
            buf[m++] = image[j];
        }
    }
    return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecTDPDAtomic::pack_border_vel( int n, int *list, double *buf,
                                        int pbc_flag, int *pbc )
{
    int i, j, k, m;
    double dx, dy, dz;
    double phantom;

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
            buf[m++] = tag[j];
            buf[m++] = type[j];
            buf[m++] = mask[j];
            buf[m++] = image[j];
        }
        for( k = 0; k < n_species; k++ ) {
            for ( i = 0; i < n; i++ ) {
                j = list[i];
                buf[m++] = CONC[k][j];
            }
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
            buf[m++] = tag[j];
            buf[m++] = type[j];
            buf[m++] = mask[j];
            buf[m++] = image[j];
        }
        for( k = 0; k < n_species; k++ ) {
            for ( i = 0; i < n; i++ ) {
                j = list[i];
                buf[m++] = CONC[k][j];
            }
        }
    }
    return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecTDPDAtomic::unpack_border( int n, int first, double *buf )
{
    int i, k, m, last;

    m = 0;
    last = first + n;
    for( i = first; i < last; i++ ) {
        if( i == nmax ) grow( 0 );
        x[i][0] = buf[m++];
        x[i][1] = buf[m++];
        x[i][2] = buf[m++];
        for ( k = 0; k < n_species; k++ )
            CONC[k][i] = buf[m++];
        tag[i]  = static_cast<int>( buf[m++] );
        type[i] = static_cast<int>( buf[m++] );
        mask[i] = static_cast<int>( buf[m++] );
        image[i] = static_cast<int>( buf[m++] );
    }
}



void AtomVecTDPDAtomic::unpack_border_vel( int n, int first, double *buf )
{
    int i, k, m, last;
    double phantom;

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
        tag[i]  = static_cast<int>( buf[m++] );
        type[i] = static_cast<int>( buf[m++] );
        mask[i] = static_cast<int>( buf[m++] );
        image[i] = static_cast<int>( buf[m++] );
    }
    for( k = 0; k < n_species; k++ ) {
        for ( i = first; i < last; i++ ) {
            CONC[k][i] = buf[m++];
        }
    }
}

/* ----------------------------------------------------------------------
     pack data for atom I for sending to another proc
     xyz must be 1st 3 values, so comm::exchange() can test on them
------------------------------------------------------------------------- */

int AtomVecTDPDAtomic::pack_exchange( int i, double *buf )
{
    int m = 1;
    buf[m++] = x[i][0];
    buf[m++] = x[i][1];
    buf[m++] = x[i][2];
    buf[m++] = v[i][0];
    buf[m++] = v[i][1];
    buf[m++] = v[i][2];
    for (int k = 0; k < n_species; k++ )
        buf[m++] = CONC[k][i];
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

int AtomVecTDPDAtomic::unpack_exchange( double *buf )
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
    for (int k = 0; k < n_species; k++ )
        CONC[k][nlocal] = buf[m++];
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

int AtomVecTDPDAtomic::size_restart()
{

    int n = (11 + n_species) * atom->nlocal;

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

int AtomVecTDPDAtomic::pack_restart( int i, double *buf )
{
    int m = 1;
    buf[m++] = x[i][0];
    buf[m++] = x[i][1];
    buf[m++] = x[i][2];
    buf[m++] = v[i][0];
    buf[m++] = v[i][1];
    buf[m++] = v[i][2];
    for (int k = 0; k < n_species; k++ )
        buf[m++] = CONC[k][i];
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

int AtomVecTDPDAtomic::unpack_restart( double *buf )
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
    for (int k = 0; k < n_species; k++ )
        CONC[k][nlocal] = buf[m++];
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

void AtomVecTDPDAtomic::data_atom( double *coord, int imagetmp, char **values )
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


    for (int k = 0; k < n_species; k++ ) {
        CONC[k][nlocal] = atof( values[5+k] );
        //printf("data_atom CONC %g\n",CONC[k][nlocal]);
    }
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

/* ----------------------------------------------------------------------
     return # of bytes of allocated memory
------------------------------------------------------------------------- */

bigint AtomVecTDPDAtomic::memory_usage()
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
    if( atom->memcheck( "CONC" ) ) bytes += nmax * n_species * sizeof( r32 );

    return bytes;
}

void AtomVecTDPDAtomic::force_clear( AtomAttribute::Descriptor range, int vflag )
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

