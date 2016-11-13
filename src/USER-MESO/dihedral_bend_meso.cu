#include "mpi.h"
#include "math.h"
#include "stdlib.h"
#include "domain.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "update.h"

#include "engine_meso.h"
#include "comm_meso.h"
#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "dihedral_bend_meso.h"
#include "neighbor_meso.h"

using namespace LAMMPS_NS;

#define SMALL 0.000001

MesoDihedralBend::MesoDihedralBend( LAMMPS *lmp ):
    		DihedralHarmonic( lmp ),
    		MesoPointers( lmp ),
    		dev_k (lmp, "MesoDihedralBend::dev_k"),
    		dev_theta0 (lmp, "MesoDihedralBend::dev_theta0")
{
	coeff_alloced = 0;
}

MesoDihedralBend::~MesoDihedralBend()
{
	if (allocated) {
		memory->destroy(theta0);
		memory->destroy(k0);
	}
}

template < int evflag >
__global__ void gpu_dihedral_bend(
	    texobj tex_coord_merged,
		r64*  __restrict force_x,
	    r64*  __restrict force_y,
	    r64*  __restrict force_z,
	    int*  __restrict ndihedral,
	    int4* __restrict dihedrals,
	    int*  __restrict dihed_type,
		r32*  __restrict k_global,
		r32*  __restrict theta0_global,
		int*  __restrict image,
	    const float3 period,
	    const int padding,
	    const int padding_type,
		const int n_type,
		const int n_local)
{

	extern __shared__ r32 shared_data[];
	r32* k = &shared_data[0];
	r32* theta0 = &shared_data[n_type+1];

	for ( int i = threadIdx.x; i < n_type + 1; i += blockDim.x ) {
		k[i]		= k_global[i];
		theta0[i] 	= theta0_global[i];
	}
	__syncthreads();



	for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_local; i += gridDim.x * blockDim.x ) {
		int n = ndihedral[i];		// particle i is in n dihedrals
		r32 s1x,s1y,s1z,s2x,s2y,s2z,s3x,s3y,s3z,s4x,s4y,s4z;
		r32 fx=0,fy=0,fz=0;

		for (int p = 0; p < n; p++) { // loop over those n dihedrals
			int4 dih = dihedrals[ i + p * padding ];
			int i1 = dih.x;
			int i2 = dih.y;
			int i3 = dih.z;
			int i4 = dih.w;
			int type = dihed_type[ i + p * padding_type ];

			f3i coord1 = tex1Dfetch<float4>( tex_coord_merged, i1 );
			f3i coord2 = tex1Dfetch<float4>( tex_coord_merged, i2 );
			f3i coord3 = tex1Dfetch<float4>( tex_coord_merged, i3 );
			f3i coord4 = tex1Dfetch<float4>( tex_coord_merged, i4 );


			// 2-1 distance
			r32 d21x = minimum_image( coord2.x - coord1.x, period.x );
			r32 d21y = minimum_image( coord2.y - coord1.y, period.y );
			r32 d21z = minimum_image( coord2.z - coord1.z, period.z );

			// 3-1 distance
			r32 d31x = minimum_image( coord3.x - coord1.x, period.x );
			r32 d31y = minimum_image( coord3.y - coord1.y, period.y );
			r32 d31z = minimum_image( coord3.z - coord1.z, period.z );

			// 3-2 distance
			r32 d32x = minimum_image( coord3.x - coord2.x, period.x );
			r32 d32y = minimum_image( coord3.y - coord2.y, period.y );
			r32 d32z = minimum_image( coord3.z - coord2.z, period.z );

			// 3-4 distance
			r32 d34x = minimum_image( coord3.x - coord4.x, period.x );
			r32 d34y = minimum_image( coord3.y - coord4.y, period.y );
			r32 d34z = minimum_image( coord3.z - coord4.z, period.z );

			// 2-4 distance
			r32 d24x = minimum_image( coord2.x - coord4.x, period.x );
			r32 d24y = minimum_image( coord2.y - coord4.y, period.y );
			r32 d24z = minimum_image( coord2.z - coord4.z, period.z );

			// 1-4 distance
			r32 d14x = minimum_image( coord1.x - coord4.x, period.x );
			r32 d14y = minimum_image( coord1.y - coord4.y, period.y );
			r32 d14z = minimum_image( coord1.z - coord4.z, period.z );



		    // calculate normals
		    r32 n1x = d21y*d31z - d31y*d21z;
		    r32 n1y = d31x*d21z - d21x*d31z;
		    r32 n1z = d21x*d31y - d31x*d21y;
		    r32 n2x = d34y*d24z - d24y*d34z;
		    r32 n2y = d24x*d34z - d34x*d24z;
		    r32 n2z = d34x*d24y - d24x*d34y;
		    r32 n1 = n1x*n1x + n1y*n1y + n1z*n1z;
		    r32 n2 = n2x*n2x + n2y*n2y + n2z*n2z;
		    r32 nn = sqrtf(n1*n2);

		    // cos(theta) and sin(theta) calculation
		    r32 costheta = (n1x*n2x + n1y*n2y + n1z*n2z)/nn;
		    if (costheta > 1.0f) costheta = 1.0f;
		    if (costheta < -1.0f) costheta = -1.0f;
		    r32 sintheta = sqrtf(1.0f-costheta*costheta);
		    if (sintheta < SMALL) sintheta = SMALL;
		    r32 mx = (n1x-n2x)*d14x + (n1y-n2y)*d14y + (n1z-n2z)*d14z;
		    if (mx < 0)
		      sintheta = -sintheta;

		    // coeffs calculation
		    r32 alfa = k[type]*(cosf(theta0[type])-costheta*sinf(theta0[type])/sintheta);
		    r32 a11 = -alfa*costheta/n1;
		    r32 a12 = alfa/nn;
		    r32 a22 = -alfa*costheta/n2;


		    // forces calculation
		    if (i==i1) {
		        s1x = a11*(n1y*d32z - n1z*d32y) + a12*(n2y*d32z - n2z*d32y);
		        s1y = a11*(n1z*d32x - n1x*d32z) + a12*(n2z*d32x - n2x*d32z);
		        s1z = a11*(n1x*d32y - n1y*d32x) + a12*(n2x*d32y - n2y*d32x);
		    	fx += s1x;
		    	fy += s1y;
		    	fz += s1z;
		    } else if (i==i2) {
		        s2x = a11*(n1z*d31y - n1y*d31z) + a22*(n2y*d34z - n2z*d34y) +
		              a12*(n2z*d31y - n2y*d31z + n1y*d34z - n1z*d34y);
		        s2y = a11*(n1x*d31z - n1z*d31x) + a22*(n2z*d34x - n2x*d34z) +
		              a12*(n2x*d31z - n2z*d31x + n1z*d34x - n1x*d34z);
		        s2z = a11*(n1y*d31x - n1x*d31y) + a22*(n2x*d34y - n2y*d34x) +
		              a12*(n2y*d31x - n2x*d31y + n1x*d34y - n1y*d34x);
		    	fx += s2x;
		    	fy += s2y;
		    	fz += s2z;
		    } else if (i==i3) {
		        s3x = a11*(n1y*d21z - n1z*d21y) + a22*(n2z*d24y - n2y*d24z) +
		              a12*(n2y*d21z - n2z*d21y + n1z*d24y - n1y*d24z);
		        s3y = a11*(n1z*d21x - n1x*d21z) + a22*(n2x*d24z - n2z*d24x) +
		              a12*(n2z*d21x - n2x*d21z + n1x*d24z - n1z*d24x);
		        s3z = a11*(n1x*d21y - n1y*d21x) + a22*(n2y*d24x - n2x*d24y) +
		              a12*(n2x*d21y - n2y*d21x + n1y*d24x - n1x*d24y);
		    	fx += s3x;
		    	fy += s3y;
		    	fz += s3z;
		    } else if (i==i4) { //i==i4
		        s4x = a22*(n2z*d32y - n2y*d32z) + a12*(n1z*d32y - n1y*d32z);
		        s4y = a22*(n2x*d32z - n2z*d32x) + a12*(n1x*d32z - n1z*d32x);
		        s4z = a22*(n2y*d32x - n2x*d32y) + a12*(n1y*d32x - n1x*d32y);
		    	fx += s4x;
		    	fy += s4y;
		    	fz += s4z;
		    }
		}

		force_x[i] += fx;
		force_y[i] += fy;
		force_z[i] += fz;
	}
}



void MesoDihedralBend::compute( int eflag, int vflag )
{
	if (!coeff_alloced ) alloc_coeff();

	static GridConfig grid_cfg_1, grid_cfg_EV_1;
	if (!grid_cfg_EV_1.x) {
		grid_cfg_EV_1 = meso_device->occu_calc.right_peak( 0, gpu_dihedral_bend<1>, ( atom->ndihedraltypes + 1 ) * 2 * sizeof(r32), cudaFuncCachePreferL1 );
		cudaFuncSetCacheConfig( gpu_dihedral_bend<1>, cudaFuncCachePreferL1 );
		grid_cfg_1    = meso_device->occu_calc.right_peak( 0, gpu_dihedral_bend<0>, ( atom->ndihedraltypes + 1 ) * 2 * sizeof(r32), cudaFuncCachePreferL1 );
		cudaFuncSetCacheConfig( gpu_dihedral_bend<0>, cudaFuncCachePreferL1 );
	}

    float3 period;
    period.x = ( domain->xperiodic ) ? ( domain->xprd ) : ( 0. );
    period.y = ( domain->yperiodic ) ? ( domain->yprd ) : ( 0. );
    period.z = ( domain->zperiodic ) ? ( domain->zprd ) : ( 0. );

	if (eflag || vflag) {
		gpu_dihedral_bend<1> <<< grid_cfg_EV_1.x, grid_cfg_EV_1.y, ( atom->ndihedraltypes + 1 ) * 2 * sizeof(r32), meso_device->stream() >>> (
	            meso_atom->tex_coord_merged,
	            meso_atom->dev_force(0), meso_atom->dev_force(1), meso_atom->dev_force(2),
				meso_atom->dev_ndihed,
				meso_atom->dev_dihed_mapped,
				meso_atom->dev_dihed_type,
				dev_k,
				dev_theta0,
				meso_atom->dev_image,
				period,
				meso_atom->dev_dihed_mapped.pitch_elem(),
				meso_atom->dev_dihed_type.pitch_elem(),
				atom->ndihedraltypes,
				atom->nlocal);
	} else {
		gpu_dihedral_bend<0> <<< grid_cfg_1.x, grid_cfg_1.y, ( atom->ndihedraltypes + 1 ) * 2 * sizeof(r32), meso_device->stream() >>> (
	            meso_atom->tex_coord_merged,
	            meso_atom->dev_force(0), meso_atom->dev_force(1), meso_atom->dev_force(2),
				meso_atom->dev_ndihed,
				meso_atom->dev_dihed_mapped,
				meso_atom->dev_dihed_type,
				dev_k,
				dev_theta0,
				meso_atom->dev_image,
				period,
				meso_atom->dev_dihed_mapped.pitch_elem(),
				meso_atom->dev_dihed_type.pitch_elem(),
				atom->ndihedraltypes,
				atom->nlocal);
	}

}



void MesoDihedralBend::alloc_coeff()
{
	if ( coeff_alloced ) return;

	coeff_alloced = 1;
	int n = atom->ndihedraltypes;
	dev_k.grow( n+1, false, false );
	dev_theta0.grow( n+1, false, false );
	dev_k.upload( k0, n+1, meso_device->stream() );
	dev_theta0.upload( theta0, n+1, meso_device->stream() );

}


void MesoDihedralBend::allocate()
{
	//allocated = 1;
	int n = atom->ndihedraltypes;

	DihedralHarmonic::allocate();
	memory->create(theta0,n+1,"MesoDihedralBend::theta0");
	memory->create(k0,n+1,"MesoDihedralBend::k0");


}

void MesoDihedralBend::coeff(int narg, char **arg)
{

	if (narg != 3) error->all(FLERR, "Incorrect args in MesoDihedralBend_coeff command");
	if (!allocated) allocate();

	int ilo,ihi;
	force->bounds(arg[0],atom->ndihedraltypes,ilo,ihi);

	r32 k_one = atof(arg[1]);
	r32 theta0_one = atof(arg[2]);

	int count = 0;
	for (int i = ilo; i <= ihi; i++) {
		k0[i] = k_one;
		theta0[i] = theta0_one*M_PI/180.0;
		setflag[i] = 1;
		count++;
	}

	if (count == 0) error->all(FLERR, "Incorrect args in MesoDihedralBend_coeff command");
}
