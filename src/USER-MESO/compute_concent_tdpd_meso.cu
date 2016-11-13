

#include "mpi.h"
#include "string.h"
#include "update.h"
#include "force.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"
#include "group.h"
#include "error.h"

#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "compute_concent_tdpd_meso.h"
#include "engine_meso.h"

using namespace LAMMPS_NS;

MesoComputeConcTDPD::MesoComputeConcTDPD( LAMMPS *lmp, int narg, char **arg ) :
    Compute( lmp, narg, arg ),
    MesoPointers( lmp ),
    conc( lmp, "MesoComputeConcTDPD::conc" ),
    c( lmp, "MesoComputeConcTDPD::c" )
{
    if( narg != 4 ) error->all( __FILE__, __LINE__, "Illegal compute concentration command" );
    nth_species = atoi(arg[3])-1;           // compute the nth species. starting from ZERO.

    scalar_flag = 1;
    vector_flag = 0;
    extscalar = 0;

    conc.grow( 1 );
    c.grow( 1 );
}

MesoComputeConcTDPD::~MesoComputeConcTDPD()
{
}

void MesoComputeConcTDPD::setup()
{
}

__global__ void sum_internal_CONC(
    r32 ** __restrict CONC,
    int * __restrict mask,
    r32 * __restrict sum,
    int * __restrict count,
    const int groupbit,
    const int n,
    const int nth_species
) {
    int k = nth_species;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    r32 conc = 0;
    int c = 0;
    if ( i < n ) {
        if ( mask[i] & groupbit ) {
            conc = CONC[k][i];
            c = 1;
        }
    }
    r32 concsum = __warp_sum( conc );
    int csum = __warp_sum( c );
    if ( __laneid() == 0 ) {
        atomicAdd( sum, concsum );
        atomicAdd( count, csum );
    }
}

double MesoComputeConcTDPD::compute_scalar()
{
    invoked_scalar = update->ntimestep;

    conc.set( 0, meso_device->stream() );
    c.set( 0, meso_device->stream() );

    size_t threads_per_block = meso_device->query_block_size( gpu_reduce_sum_host<float> );
    sum_internal_CONC <<< ( atom->nlocal + threads_per_block - 1 ) / threads_per_block, threads_per_block, 0, meso_device->stream() >>> (
        meso_atom->dev_CONC.ptrs(),
        meso_atom->dev_mask,
        conc,
        c,
        groupbit,
        atom->nlocal,
        nth_species);

    float  concsum;
    int    csum;
    conc.download( &concsum, 1, meso_device->stream() );
    c.download( &csum, 1, meso_device->stream() );

    meso_device->stream().sync();

    u64 csum_u64 = csum, total_atoms;
    float fscalar = 0;
    MPI_Allreduce( &concsum, &fscalar, 1, MPI_FLOAT, MPI_SUM, world );
    MPI_Allreduce( &csum_u64, &total_atoms, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, world );
    fscalar /= total_atoms;
    scalar = static_cast<double> (fscalar);
    return scalar;
}


