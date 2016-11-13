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
#include "angle_area_volume_meso.h"
#include "neighbor_meso.h"


using namespace LAMMPS_NS;


MesoAngleAreaVolume::MesoAngleAreaVolume(LAMMPS *lmp) :
    Angle( lmp ),
    MesoPointers( lmp ),
    dev_ka (lmp, "MesoAngleAreavolume::dev_ka"),
    dev_a0 (lmp, "MesoAngleAreavolume::dev_a0"),
    dev_kv (lmp, "MesoAngleAreavolume::dev_kv"),
    dev_v0 (lmp, "MesoAngleAreavolume::dev_v0"),
    dev_kl (lmp, "MesoAngleAreavolume::dev_kl"),
    dev_ttyp(lmp, "MesoAngleAreavolume::dev_ttyp"),
    dev_nm (lmp, "MesoAngleAreavolume::dev_nm"),
    datt(lmp, "MesoAngleAreavolume::datt"),
    dath(lmp, "MesoAngleAreavolume::dath"),
    dev_datt(lmp, "MesoAngleAreavolume::dev_datt"),
    dev_dath(lmp, "MesoAngleAreavolume::dev_dath"),
    dev_datt_laststep(lmp, "MesoAngleAreaVolume::dev_datt_laststep")
{
    coeff_alloced = 0;
}

MesoAngleAreaVolume::~MesoAngleAreaVolume()
{
    if (allocated) {
        memory->sfree(ka);
        memory->sfree(a0);
        memory->sfree(kv);
        memory->sfree(v0);
        memory->sfree(kl);
    }
    if (init_on) {
        memory->sfree(ttyp);
        memory->sfree(ttyp1);
    }

    if ( avrequest != MPI_REQUEST_NULL )  MPI_Wait( &avrequest, &avstatus );

}

void MesoAngleAreaVolume::alloc_coeff()
{
    if ( coeff_alloced ) return;

    coeff_alloced = 1;
    int n = atom->nangletypes;
    dev_ka.grow( n+1, false, false );
    dev_a0.grow( n+1, false, false );
    dev_kv.grow( n+1, false, false );
    dev_v0.grow( n+1, false, false );
    dev_kl.grow( n+1, false, false );
    dev_ka.upload( ka, n+1, meso_device->stream() );
    dev_a0.upload( a0, n+1, meso_device->stream() );
    dev_kv.upload( kv, n+1, meso_device->stream() );
    dev_v0.upload( v0, n+1, meso_device->stream() );
    dev_kl.upload( kl, n+1, meso_device->stream() );
}

void MesoAngleAreaVolume::allocate()
{
    int i;
    allocated = 1;
    int n = atom->nangletypes;
    init_on = 0;

    memory->create(ka,n+1,"angle:ka");
    memory->create(a0,n+1,"angle:a0");
    memory->create(kv,n+1,"angle:kv");
    memory->create(v0,n+1,"angle:v0");
    memory->create(kl,n+1,"angle:kl");

    memory->create(setflag,n+1,"angle:setflag");
    for (i = 1; i <= n; i++) setflag[i] = 0;
}


void MesoAngleAreaVolume::coeff(int narg, char **arg)
{
    if (narg != 6) error->all(FLERR, "Incorrect number of args in angle_coeff command");
    if (!allocated) allocate();

    int i,ilo,ihi;
    force->bounds(arg[0],atom->nangletypes,ilo,ihi);

    int nnarg = 1;
    r32 ka_one = atof(arg[nnarg++]);
    r32 a0_one = atof(arg[nnarg++]);
    r32 kv_one = atof(arg[nnarg++]);
    r32 v0_one = atof(arg[nnarg++]);
    r32 kl_one = atof(arg[nnarg++]);

    int count = 0;
    for (i = ilo; i <= ihi; i++) {
        ka[i] = ka_one;
        a0[i] = a0_one;
        kv[i] = kv_one;
        v0[i] = v0_one;
        kl[i] = kl_one;
        setflag[i] = 1;
        count++;
    }

    if (count == 0) error->all(FLERR, "Incorrect args in angle_coeff command");

}


/* Loop over all particles, find total area and volume of each RBC. */
template <int evflag>
__global__ void gpu_angle_areavolume_tot(
    r64*  __restrict coord_x,
    r64*  __restrict coord_y,
    r64*  __restrict coord_z,
    int*  __restrict molecule,
    int*  __restrict nangle,
    int4* __restrict angles,
    r64*  dath,
    const int nm,
    int* __restrict image,
    const r32 box_size_x,
    const r32 box_size_y,
    const r32 box_size_z,
    const int padding,
    const int n_type,
    const int n_local )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_local ; i += gridDim.x * blockDim.x ) {
        int n = nangle[i];
        int m = (molecule[i]-1000) - 1;

        int img1,img2,img3,ximg,yimg,zimg;
        r64 aa_sum = 0, vv_sum = 0;
        for (int p = 0; p < n; p++) {
            int4 angle = angles[ i + p * padding];
            int i1 = angle.x;
            int i2 = angle.y;
            int i3 = angle.z;

            r64 coord1x = coord_x[i1];
            r64 coord1y = coord_y[i1];
            r64 coord1z = coord_z[i1];
            r64 coord2x = coord_x[i2];
            r64 coord2y = coord_y[i2];
            r64 coord2z = coord_z[i2];
            r64 coord3x = coord_x[i3];
            r64 coord3y = coord_y[i3];
            r64 coord3z = coord_z[i3];

            // unmap

            img1 = image[i1];
            ximg = (  img1              & IMGMASK ) - IMGMAX;
            yimg = ( (img1 >> IMGBITS)  & IMGMASK ) - IMGMAX;
            zimg = ( (img1 >> IMG2BITS) & IMGMASK ) - IMGMAX;
            coord1x += ximg * box_size_x;
            coord1y += yimg * box_size_y;
            coord1z += zimg * box_size_z;
            //printf("img1 = %i %i %i\n",ximg,yimg,zimg);
            img2 = image[i2];
            ximg = (  img2              & IMGMASK ) - IMGMAX;
            yimg = ( (img2 >> IMGBITS)  & IMGMASK ) - IMGMAX;
            zimg = ( (img2 >> IMG2BITS) & IMGMASK ) - IMGMAX;
            coord2x += ximg * box_size_x;
            coord2y += yimg * box_size_y;
            coord2z += zimg * box_size_z;
            //printf("img2 = %d %d %d\n",ximg,yimg,zimg);
            img3 = image[i3];
            ximg = (  img3              & IMGMASK ) - IMGMAX;
            yimg = ( (img3 >> IMGBITS)  & IMGMASK ) - IMGMAX;
            zimg = ( (img3 >> IMG2BITS) & IMGMASK ) - IMGMAX;
            coord3x += ximg * box_size_x;
            coord3y += yimg * box_size_y;
            coord3z += zimg * box_size_z;
            //printf("img3 = %d %d %d\n",ximg,yimg,zimg);


            // 2-1 distance
            r64 d21x = coord2x - coord1x;
            r64 d21y = coord2y - coord1y;
            r64 d21z = coord2z - coord1z;

            // 3-1 distance
            r64 d31x = coord3x - coord1x;
            r64 d31y = coord3y - coord1y;
            r64 d31z = coord3z - coord1z;

            // calculate normal
            r64 nx = d21y*d31z - d31y*d21z;
            r64 ny = d31x*d21z - d21x*d31z;
            r64 nz = d21x*d31y - d31x*d21y;
            r64 nn = sqrt(nx*nx + ny*ny + nz*nz);


            r64 mx = coord1x + coord2x + coord3x;
            r64 my = coord1y + coord2y + coord3y;
            r64 mz = coord1z + coord2z + coord3z;



            // calculate area and volume, then accumulate
            r64 aa = 0.5*nn;
            r64 vv = (nx*mx + ny*my + nz*mz) / 18.0;
            aa_sum += aa;
            vv_sum += vv;

        }

        if (n!=0) { //exclude fluid particle to save computation.
            atomic_add( dath + m, aa_sum/3.0);
            atomic_add( dath + m + nm, vv_sum/3.0);
        }
    }
}


template <int evflag>
__global__ void gpu_angle_areavolume_force(
    r64*  __restrict coord_x,
    r64*  __restrict coord_y,
    r64*  __restrict coord_z,
    int*  __restrict molecule,
    r64*  __restrict force_x,
    r64*  __restrict force_y,
    r64*  __restrict force_z,
    int*  __restrict nangle,
    int4* __restrict angles,
    r64*  __restrict angle_a0,
    r32*  __restrict ka_global,
    r32*  __restrict a0_global,
    r32*  __restrict kv_global,
    r32*  __restrict v0_global,
    r32*  __restrict kl_global,
    int*  __restrict ttyp_global,
    r64*  datt,
    const int nm,
    int* __restrict image,
    const r32 box_size_x,
    const r32 box_size_y,
    const r32 box_size_z,
    const int padding,
    const int padding_a0,
    const int n_type,
    const int n_local )
{
    extern __shared__ r32 shared_data[];
    r32* ka = &shared_data[0];
    r32* a0 = &shared_data[n_type+1];
    r32* kv = &shared_data[2*(n_type+1)];
    r32* v0 = &shared_data[3*(n_type+1)];
    r32* kl = &shared_data[4*(n_type+1)];
    int* ttyp = (int*) &shared_data[5*(n_type+1)];

    for ( int i = threadIdx.x ; i < n_type+1; i += blockDim.x ) {
        ka  [i] = ka_global[i];
        a0  [i] = a0_global[i];
        kv  [i] = kv_global[i];
        v0  [i] = v0_global[i];
        kl  [i] = kl_global[i];
        ttyp[i] = ttyp_global[i];
    }
    __syncthreads();


    /* loop over all particles to find force */
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_local ; i += gridDim.x * blockDim.x ) {
        int n = nangle[i];
        int m = (molecule[i]-1000) - 1;

        r32 fx=0.0, fy=0.0, fz=0.0;
        int img1,img2,img3,ximg,yimg,zimg;
        for (int p = 0; p < n; p++) {
            int4 angle = angles[ i + p * padding];
            int i1 = angle.x;
            int i2 = angle.y;
            int i3 = angle.z;
            int type = angle.w;

            r64 coord1x = coord_x[i1];
            r64 coord1y = coord_y[i1];
            r64 coord1z = coord_z[i1];
            r64 coord2x = coord_x[i2];
            r64 coord2y = coord_y[i2];
            r64 coord2z = coord_z[i2];
            r64 coord3x = coord_x[i3];
            r64 coord3y = coord_y[i3];
            r64 coord3z = coord_z[i3];

            // unmap
            img1 = image[i1];
            ximg = (  img1              & IMGMASK ) - IMGMAX;
            yimg = ( (img1 >> IMGBITS)  & IMGMASK ) - IMGMAX;
            zimg = ( (img1 >> IMG2BITS) & IMGMASK ) - IMGMAX;
            coord1x += ximg * box_size_x;
            coord1y += yimg * box_size_y;
            coord1z += zimg * box_size_z;
            img2 = image[i2];
            ximg = (  img2              & IMGMASK ) - IMGMAX;
            yimg = ( (img2 >> IMGBITS)  & IMGMASK ) - IMGMAX;
            zimg = ( (img2 >> IMG2BITS) & IMGMASK ) - IMGMAX;
            coord2x += ximg * box_size_x;
            coord2y += yimg * box_size_y;
            coord2z += zimg * box_size_z;
            img3 = image[i3];
            ximg = (  img3              & IMGMASK ) - IMGMAX;
            yimg = ( (img3 >> IMGBITS)  & IMGMASK ) - IMGMAX;
            zimg = ( (img3 >> IMG2BITS) & IMGMASK ) - IMGMAX;
            coord3x += ximg * box_size_x;
            coord3y += yimg * box_size_y;
            coord3z += zimg * box_size_z;



            // 2-1 distance
            r64 d21x = coord2x - coord1x;
            r64 d21y = coord2y - coord1y;
            r64 d21z = coord2z - coord1z;

            // 3-1 distance
            r64 d31x = coord3x - coord1x;
            r64 d31y = coord3y - coord1y;
            r64 d31z = coord3z - coord1z;

            // 3-2 distance
            r64 d32x = coord3x - coord2x;
            r64 d32y = coord3y - coord2y;
            r64 d32z = coord3z - coord2z;

            // calculate normal
            r64 nx = d21y*d31z - d31y*d21z;
            r64 ny = d31x*d21z - d21x*d31z;
            r64 nz = d21x*d31y - d31x*d21y;
            r64 nn = sqrt(nx*nx + ny*ny + nz*nz);

            r64 mx = coord1x + coord2x + coord3x;
            r64 my = coord1y + coord2y + coord3y;
            r64 mz = coord1z + coord2z + coord3z;


            // calculate coeffs
            r32 ar0 = angle_a0[ i + p * padding_a0 ];
            r32 coefl = 0.5f*kl[type]*(ar0-0.5f*nn)/ar0/nn;
            r32 coefa = 0.5f*ka[type]*(a0[type]-datt[m])/a0[type]/nn;
            r32 coefca = coefl + coefa;
            r32 coefv = kv[type]*(v0[type]-datt[m+nm])/v0[type]/18.0f;



            // The force of the local particle is all I need.
            if (i==i1) {
                r32 s1x = coefca*(ny*d32z - nz*d32y);
                r32 s1y = coefca*(nz*d32x - nx*d32z);
                r32 s1z = coefca*(nx*d32y - ny*d32x);
                r32 s1xv = coefv*(nx + d32z*my - d32y*mz);
                r32 s1yv = coefv*(ny - d32z*mx + d32x*mz);
                r32 s1zv = coefv*(nz + d32y*mx - d32x*my);
                fx += s1x + s1xv;
                fy += s1y + s1yv;
                fz += s1z + s1zv;
            } else if (i==i2) {
                r32 s2x = coefca*(nz*d31y - ny*d31z);
                r32 s2y = coefca*(nx*d31z - nz*d31x);
                r32 s2z = coefca*(ny*d31x - nx*d31y);
                r32 s2xv = coefv*(nx - d31z*my + d31y*mz);
                r32 s2yv = coefv*(ny + d31z*mx - d31x*mz);
                r32 s2zv = coefv*(nz - d31y*mx + d31x*my);
                fx += s2x + s2xv;
                fy += s2y + s2yv;
                fz += s2z + s2zv;
            } else { // (i==i3)
                r32 s3x = coefca*(ny*d21z - nz*d21y);
                r32 s3y = coefca*(nz*d21x - nx*d21z);
                r32 s3z = coefca*(nx*d21y - ny*d21x);
                r32 s3xv = coefv*(nx + d21z*my - d21y*mz);
                r32 s3yv = coefv*(ny - d21z*mx + d21x*mz);
                r32 s3zv = coefv*(nz + d21y*mx - d21x*my);
                fx += s3x + s3xv;
                fy += s3y + s3yv;
                fz += s3z + s3zv;
            }

        }

        force_x[i] += fx;
        force_y[i] += fy;
        force_z[i] += fz;

    }

}



void MesoAngleAreaVolume::compute( int eflag, int vflag )
{
    if (!coeff_alloced ) alloc_coeff();

    static GridConfig grid_cfg_1, grid_cfg_EV_1;
    if (!grid_cfg_EV_1.x) {
        grid_cfg_EV_1 = meso_device->occu_calc.right_peak( 0, gpu_angle_areavolume_tot<1>, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_angle_areavolume_tot<1>, cudaFuncCachePreferL1 );
        grid_cfg_1    = meso_device->occu_calc.right_peak( 0, gpu_angle_areavolume_tot<0>, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_angle_areavolume_tot<0>, cudaFuncCachePreferL1 );
    }

    static GridConfig grid_cfg, grid_cfg_EV;
    if (!grid_cfg_EV.x) {
        grid_cfg_EV = meso_device->occu_calc.right_peak( 0, gpu_angle_areavolume_force<1>, ( atom->nbondtypes + 1 ) * (5 * sizeof( r32 ) + 1 * sizeof( int )), cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_angle_areavolume_force<1>, cudaFuncCachePreferL1 );
        grid_cfg    = meso_device->occu_calc.right_peak( 0, gpu_angle_areavolume_force<0>, ( atom->nbondtypes + 1 ) * (5 * sizeof( r32 ) + 1 * sizeof( int )), cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_angle_areavolume_force<0>, cudaFuncCachePreferL1 );
    }


    if (init_on == 0) {
        int m;
        n_mol = 0;
        for( int i=0; i < atom->nlocal; i++ ) {
            n_mol = std::max( n_mol, atom->molecule[i] );
        }
        n_mol -= 1000;
        MPI_Allreduce(&n_mol,&nm,1,MPI_INT,MPI_MAX,world);
        // nm is the maximum molecule-ID across all ranks.
        // Assume RBC molecule-ID starts from 1001, and the others starts with 1.
        // Not necessarily starting 1001. But it saves space.

        if (nm <= 0)
            error->all(FLERR, "Did you forget to set RBC molecule-ID > 1000?");

        memory->create(ttyp,nm,"MesoAngleAreavolume_meso::ttyp");
        memory->create(ttyp1,nm,"MesoAngleAreavolume_meso::ttyp1");
        for (int n = 0; n < nm; n++) ttyp1[n] = 0;
        for (int n=0; n < atom->nlocal; n++) {
            m = (atom->molecule[n]-1000) -1;
            if (m >= 0 ) {          // molecule-ID indicates RBC
                ttyp1[m] = atom->angle_type[n][0];          // for atom n, the type of angle associated with it.
            }
        }
        MPI_Allreduce(&ttyp1,ttyp,nm,MPI_INT,MPI_MAX,world);
        dev_ttyp.grow( nm, false, false );
        dev_ttyp.upload( ttyp, nm, meso_device->stream() );


        dath.grow( 2*nm, false, true );
        datt.grow( 2*nm, false, true );
        dev_dath.grow( 2*nm, false, false );
        dev_datt.grow( 2*nm, false, false );
        dev_datt_laststep.grow( 2*nm, false, true );


        // Calculate the area and volume at the beginning.
        dev_dath.set( 0, meso_device->stream() );
        gpu_angle_areavolume_tot<0> <<< grid_cfg_1.x, grid_cfg_1.y, 0, meso_device->stream() >>> (
            meso_atom->dev_coord(0), meso_atom->dev_coord(1), meso_atom->dev_coord(2),
            meso_atom->dev_mole,
            meso_atom->dev_nangle,
            meso_atom->dev_angle_mapped,
            dev_dath,
            nm,
            meso_atom->dev_image,
            domain->xprd, domain->yprd, domain->zprd,
            meso_atom->dev_angle_mapped.pitch_elem(),
            atom->nangletypes,
            atom->nlocal );
        dev_dath.download( dath, 2*nm, meso_device->stream() );

        cudaDeviceSynchronize();
        MPI_Barrier(world);
        MPI_Iallreduce(dath,datt,2*nm,MPI_DOUBLE,MPI_SUM,world,&avrequest);
        dev_datt_laststep.upload( datt, 2*nm, meso_device->stream() );

        cudaDeviceSynchronize();
        init_on = 1;
    }


    if (eflag || vflag) {
        MPI_Wait( &avrequest, &avstatus );
        dev_datt_laststep.upload( datt, 2*nm, meso_device->stream(6) );
        CUDAEvent Event0 = meso_device->event( "datt_upload_to_dev_datt_laststep");
        Event0.record( meso_device->stream(6) );

        dev_dath.set( 0, meso_device->stream() ); // must reset dev_dath to 0 at every timestep.

        /* kernel 1 - get total area and volume of each RBC. */
        gpu_angle_areavolume_tot<1> <<< grid_cfg_EV_1.x, grid_cfg_EV_1.y, 0, meso_device->stream() >>> (
            meso_atom->dev_coord(0), meso_atom->dev_coord(1), meso_atom->dev_coord(2),
            meso_atom->dev_mole,
            meso_atom->dev_nangle,
            meso_atom->dev_angle_mapped,
            dev_dath,
            nm,
            meso_atom->dev_image,
            domain->xprd, domain->yprd, domain->zprd,
            meso_atom->dev_angle_mapped.pitch_elem(),
            atom->nangletypes,
            atom->nlocal );
        dev_dath.download( dath, 2*nm, meso_device->stream() );

        CUDAEvent Event1 = meso_device->event( "dev_dath_download_to_dath");
        Event1.record( meso_device->stream() );

        Event0.sync();

        /* kernel 2 - Now I have area and volume of each RBC, loop over all angles to find force on each particle. */
        gpu_angle_areavolume_force<1> <<< grid_cfg_EV.x, grid_cfg_EV.y, ( atom->nangletypes + 1 ) * (5 * sizeof( r32 ) + 1 * sizeof( int )), meso_device->stream() >>> (
            meso_atom->dev_coord(0), meso_atom->dev_coord(1), meso_atom->dev_coord(2),
            meso_atom->dev_mole,
            meso_atom->dev_force(0), meso_atom->dev_force(1), meso_atom->dev_force(2),
            meso_atom->dev_nangle,
            meso_atom->dev_angle_mapped,
            meso_atom->dev_angle_a0,
            dev_ka,         // r32
            dev_a0,         // r32
            dev_kv,         // r32
            dev_v0,         // r32
            dev_kl,         // r32
            dev_ttyp,       // int
            dev_datt_laststep,      // r64
            nm,
            meso_atom->dev_image,
            domain->xprd, domain->yprd, domain->zprd,
            meso_atom->dev_angle_mapped.pitch_elem(),
            meso_atom->dev_angle_a0.pitch_elem(),
            atom->nangletypes,
            atom->nlocal );

        Event1.sync();

        /* SUM total area and volume over all ranks for each RBC */
        MPI_Iallreduce(dath,datt,2*nm,MPI_DOUBLE,MPI_SUM,world,&avrequest);

    } else {
        MPI_Wait( &avrequest, &avstatus );
        dev_datt_laststep.upload( datt, 2*nm, meso_device->stream(6) );
        CUDAEvent Event0 = meso_device->event( "datt_upload_to_dev_datt_laststep");
        Event0.record( meso_device->stream(6) );

        dev_dath.set( 0, meso_device->stream() ); // must reset dev_dath to 0 at every timestep.

        /* kernel 1 - get total area and volume of each RBC. */
        gpu_angle_areavolume_tot<0> <<< grid_cfg_EV_1.x, grid_cfg_EV_1.y, 0, meso_device->stream() >>> (
            meso_atom->dev_coord(0), meso_atom->dev_coord(1), meso_atom->dev_coord(2),
            meso_atom->dev_mole,
            meso_atom->dev_nangle,
            meso_atom->dev_angle_mapped,
            dev_dath,
            nm,
            meso_atom->dev_image,
            domain->xprd, domain->yprd, domain->zprd,
            meso_atom->dev_angle_mapped.pitch_elem(),
            atom->nangletypes,
            atom->nlocal );
        dev_dath.download( dath, 2*nm, meso_device->stream() );

        CUDAEvent Event1 = meso_device->event( "dev_dath_download_to_dath");
        Event1.record( meso_device->stream() );

        Event0.sync();

        /* kernel 2 - Now I have area and volume of each RBC, loop over all angles to find force on each particle. */
        gpu_angle_areavolume_force<0> <<< grid_cfg_EV.x, grid_cfg_EV.y, ( atom->nangletypes + 1 ) * (5 * sizeof( r32 ) + 1 * sizeof( int )), meso_device->stream() >>> (
            meso_atom->dev_coord(0), meso_atom->dev_coord(1), meso_atom->dev_coord(2),
            meso_atom->dev_mole,
            meso_atom->dev_force(0), meso_atom->dev_force(1), meso_atom->dev_force(2),
            meso_atom->dev_nangle,
            meso_atom->dev_angle_mapped,
            meso_atom->dev_angle_a0,
            dev_ka,         // r32
            dev_a0,         // r32
            dev_kv,         // r32
            dev_v0,         // r32
            dev_kl,         // r32
            dev_ttyp,       // int
            dev_datt_laststep,      // r64
            nm,
            meso_atom->dev_image,
            domain->xprd, domain->yprd, domain->zprd,
            meso_atom->dev_angle_mapped.pitch_elem(),
            meso_atom->dev_angle_a0.pitch_elem(),
            atom->nangletypes,
            atom->nlocal );

        Event1.sync();

        /* SUM total area and volume over all ranks for each RBC */
        MPI_Iallreduce(dath,datt,2*nm,MPI_DOUBLE,MPI_SUM,world,&avrequest);
    }


}

double MesoAngleAreaVolume::equilibrium_angle(int x)
{
    return -1;
}

void MesoAngleAreaVolume::write_restart(FILE *fp)
{}

void MesoAngleAreaVolume::read_restart(FILE *fp)
{}

double MesoAngleAreaVolume::single(int type, int i1, int i2, int i3)
{
    return -1;
}
