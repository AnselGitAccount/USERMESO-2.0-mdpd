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
#include "compute_pressure_meso.h"
#include "engine_meso.h"

using namespace LAMMPS_NS;

MesoComputePressure::MesoComputePressure( LAMMPS *lmp, int narg, char **arg ) :
		Compute( lmp, narg, arg ),
		MesoPointers( lmp ),
		p( lmp, "MesoComputePressure::p" ),
	    per_atom_pressure( lmp, "MesoComputeTemp::per_atom_pressure" )

{
    if( narg != 4 ) error->all( __FILE__, __LINE__, "Illegal compute pressure command" );

    scalar_flag = 1;
    vector_flag = 0;
    extscalar = 0;
    tempflag = 1;
    pfactor = 0;
    pressflag = 1;

    // store temperature ID used by pressure computation
    // insure it is valid for temperature computation

    int n = strlen(arg[3]) + 1;
    id_temp = new char[n];
    strcpy(id_temp,arg[3]);

    int icompute = modify->find_compute(id_temp);
    if (icompute < 0)
      error->all(FLERR,"Could not find compute pressure temperature ID");
    if (modify->compute[icompute]->tempflag == 0)
      error->all(FLERR,
                 "Compute pressure temperature ID does not compute temperature");

    // process optional args

    if (narg == 4) {
      keflag = 1;
      pairflag = 1;
      bondflag = angleflag = dihedralflag = improperflag = 1;
      kspaceflag = fixflag = 1;
    } else {
      keflag = 0;
      pairflag = 0;
      bondflag = angleflag = dihedralflag = improperflag = 0;
      kspaceflag = fixflag = 0;
      int iarg = 4;
      while (iarg < narg) {
        if (strcmp(arg[iarg],"ke") == 0) keflag = 1;
        else if (strcmp(arg[iarg],"pair") == 0) pairflag = 1;
        else if (strcmp(arg[iarg],"bond") == 0) bondflag = 1;
        else if (strcmp(arg[iarg],"angle") == 0) angleflag = 1;
        else if (strcmp(arg[iarg],"dihedral") == 0) dihedralflag = 1;
        else if (strcmp(arg[iarg],"improper") == 0) improperflag = 1;
        else if (strcmp(arg[iarg],"kspace") == 0) kspaceflag = 1;
        else if (strcmp(arg[iarg],"fix") == 0) fixflag = 1;
        else if (strcmp(arg[iarg],"virial") == 0) {
          pairflag = 1;
          bondflag = angleflag = dihedralflag = improperflag = 1;
          kspaceflag = fixflag = 1;
        } else error->all(FLERR,"Illegal compute pressure command");
        iarg++;
      }
    }

    p.grow( 1 );
}

MesoComputePressure::~MesoComputePressure()
{
}

void MesoComputePressure::setup()
{
}

void MesoComputePressure::init()
{
	int icompute = modify->find_compute(id_temp);
	if (icompute < 0)
		error->all(FLERR,"Could not find compute pressure temperature ID");
	temperature = modify->compute[icompute];
}

__global__ void gpu_pressure_scalar(
    r64* __restrict virial_xx,
    r64* __restrict virial_yy,
    r64* __restrict virial_zz,
    int* __restrict mask,
    r64* __restrict pressure,
    int  groupbit,
    int  n
)
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x ){
    	if( mask[i] & groupbit ) {
    		pressure[i] = virial_xx[i] +  virial_yy[i] + virial_zz[i];
    	}
    }
}

double MesoComputePressure::compute_scalar()
{
	invoked_scalar = update->ntimestep;

	// invoke temperature it it hasn't been already
	// need to compute temperature prior to compute pressure

	double t;
	if (keflag) {
	    if (temperature->invoked_scalar != update->ntimestep)
	      t = temperature->compute_scalar();
	    else t = temperature->scalar;
	}

    if( atom->nlocal > per_atom_pressure.n_elem() ) per_atom_pressure.grow( atom->nlocal, false, false );
    size_t threads_per_block = meso_device->query_block_size( gpu_pressure_scalar );
    gpu_pressure_scalar <<< n_block( atom->nlocal, threads_per_block ), threads_per_block, 0, meso_device->stream() >>> (
        meso_atom->dev_virial(0),
        meso_atom->dev_virial(1),
        meso_atom->dev_virial(2),
        meso_atom->dev_mask,
        per_atom_pressure,
        groupbit,
        atom->nlocal );

    // Sum virial_xx over all atoms.
	threads_per_block = meso_device->query_block_size( gpu_reduce_sum_host<double> ) ;
	gpu_reduce_sum_host <<< 1, threads_per_block, 0, meso_device->stream() >>> ( per_atom_pressure.ptr(), p.ptr(), atom->nlocal );
	meso_device->stream().sync();

	double virial_sum;
	MPI_Allreduce( p, &virial_sum, 1, MPI_DOUBLE, MPI_SUM, world );

	double inv_volume = 1.0 / (domain->xprd * domain->yprd * domain->zprd);
    if (keflag)
      scalar = (temperature->dof * force->boltz * t + virial_sum) / 3.0 * inv_volume * force->nktv2p;
    else
      scalar = virial_sum / 3.0 * inv_volume * force->nktv2p;

	return scalar;
}

