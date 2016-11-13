
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
#include "bond_wlc_pow_all_visc_meso.h"
#include "neighbor_meso.h"

using namespace LAMMPS_NS;


MesoBondWLCPowAllVisc::MesoBondWLCPowAllVisc( LAMMPS *lmp ):
    Bond(lmp),
    MesoPointers( lmp ),
    dev_temp( lmp, "MesoBondWLCPowAllVisc::dev_temp" ),
    dev_r0( lmp, "MesoBondWLCPowAllVisc::dev_r0" ),
    dev_mu_targ( lmp, "MesoBondWLCPowAllVisc::dev_mu_targ" ),
    dev_qp( lmp, "MesoBondWLCPowAllVisc::dev_qp" ),
    dev_gamc( lmp, "MesoBondWLCPowAllVisc::dev_gamc" ),
    dev_gamt( lmp, "MesoBondWLCPowAllVisc::dev_gamt" ),
    dev_sigc( lmp, "MesoBondWLCPowAllVisc::dev_sigc" ),
    dev_sigt( lmp, "MesoBondWLCPowAllVisc::dev_sigt" )
{
    coeff_alloced = 0;
}

MesoBondWLCPowAllVisc::~MesoBondWLCPowAllVisc()
{
    if (setflag) memory->destroy(setflag);
}

void MesoBondWLCPowAllVisc::allocate_gpu()
{
    if( coeff_alloced ) return;

    coeff_alloced = 1;
    int n = atom->nbondtypes;
    dev_temp.grow( n + 1, false, false );
    dev_r0.grow( n + 1, false, false );
    dev_mu_targ.grow( n + 1, false, false );
    dev_qp.grow( n + 1, false, false );
    dev_gamc.grow( n + 1, false, false );
    dev_gamt.grow( n + 1, false, false );
    dev_sigc.grow( n + 1, false, false );
    dev_sigt.grow( n + 1, false, false );
    dev_temp.upload( temp.data(), n+1, meso_device->stream() );
    dev_r0.upload( r0.data(), n+1, meso_device->stream() );
    dev_mu_targ.upload( mu_targ.data(), n+1, meso_device->stream() );
    dev_qp.upload( qp.data(), n+1, meso_device->stream() );
    dev_gamc.upload( gamc.data(), n+1, meso_device->stream() );
    dev_gamt.upload( gamt.data(), n+1, meso_device->stream() );
    dev_sigc.upload( sigc.data(), n+1, meso_device->stream() );
    dev_sigt.upload( sigt.data(), n+1, meso_device->stream() );
}

void MesoBondWLCPowAllVisc::allocate_cpu()
{
    allocated = 1;
    int n = atom->nbondtypes;
    temp.resize(n+1);
    r0.resize(n+1);
    mu_targ.resize(n+1);
    qp.resize(n+1);
    gamc.resize(n+1);
    gamt.resize(n+1);
    sigc.resize(n+1);
    sigt.resize(n+1);
    memory->create(setflag,n+1,"bond:setflag");
    for (int i = 1; i <= n; i++) setflag[i] = 0;
}

template <int evflag>
__global__ void gpu_bond_wlcpowallvisc(
    texobj tex_coord_merged,
    r64*  __restrict force_x,
    r64*  __restrict force_y,
    r64*  __restrict force_z,
    texobj tex_veloc,
    int*  __restrict nbond,
    int2* __restrict bonds,
    r64*  __restrict bond_r0,
    r32*  __restrict temp_global,
    r32*  __restrict r0_global,
    r32*  __restrict mu_targ_global,
    r32*  __restrict qp_global,
    r32*  __restrict gamc_global,
    r32*  __restrict gamt_global,
    r32*  __restrict sigc_global,
    r32*  __restrict sigt_global,
    const float3 period,
    const int padding,
    const int n_type,
    const int n_local )
{
    extern __shared__ r32 shared_data[];
    r32* temp       = &shared_data[0];
    r32* r0         = &shared_data[1*(n_type+1)];
    r32* mu_targ    = &shared_data[2*(n_type+1)];
    r32* qp         = &shared_data[3*(n_type+1)];
    r32* gamc       = &shared_data[4*(n_type+1)];
    r32* gamt       = &shared_data[5*(n_type+1)];
    r32* sigc       = &shared_data[6*(n_type+1)];
    r32* sigt       = &shared_data[7*(n_type+1)];

    for ( int i = threadIdx.x; i < n_type + 1; i += blockDim.x ) {
        temp[i]     = temp_global[i];
        r0[i]       = r0_global[i];
        mu_targ[i]  = mu_targ_global[i];
        qp[i]       = qp_global[i];
        gamc[i]     = gamc_global[i];
        gamt[i]     = gamt_global[i];
        sigc[i]     = sigc_global[i];
        sigt[i]     = sigt_global[i];
    }
    __syncthreads();


    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_local ; i += gridDim.x * blockDim.x ) {
        int n = nbond[i];
        f3i coord1 = tex1Dfetch<float4>( tex_coord_merged, i );
        f3u veloc1 = tex1Dfetch<float4>( tex_veloc, i );
        r32 fxi = 0.0, fyi = 0.0, fzi = 0.0;

        for( int p = 0; p < n ; p++ ) {
            int j    = bonds[ i + p*padding ].x;
            int type = bonds[ i + p*padding ].y;
            f3i coord2 = tex1Dfetch<float4>( tex_coord_merged, j );
            r32 delx = minimum_image( coord1.x - coord2.x, period.x ) ;
            r32 dely = minimum_image( coord1.y - coord2.y, period.y ) ;
            r32 delz = minimum_image( coord1.z - coord2.z, period.z ) ;
            f3u veloc2   = tex1Dfetch<float4>( tex_veloc, j );
            r32 dvx  = veloc1.x - veloc2.x;
            r32 dvy  = veloc1.y - veloc2.y;
            r32 dvz  = veloc1.z - veloc2.z;

            r32 l0 = bond_r0[ i + p*padding ];
            r32 ra = sqrtf(delx*delx + dely*dely + delz*delz);
            r32 lmax = l0*r0[type];
            r32 rr = 1.0f/r0[type];
            r32 kph = powf(l0,qp[type])*temp[type]*(0.25f/((1.0f-rr)*(1.0f-rr))-0.25f+rr);
            r32 mu = 0.25f*r32(_SQRT_3)*(temp[type]*(-0.25f/((1.0f-rr)*(1.0f-rr)) + 0.25f + 0.5f*rr/((1.0f-rr)*(1.0f-rr)*(1.0f-rr)))/(lmax*rr) + kph*(qp[type]+1.0f)/powf(l0,qp[type]+1.0f));
            r32 lambda = mu/mu_targ[type];
            kph = kph*mu_targ[type]/mu;
            rr = ra/lmax;
            r32 rlogarg = powf(ra,qp[type]+1.0f);
            r32 vv = (delx*dvx + dely*dvy + delz*dvz)/ra;


            if (rr >= 0.99) rr = 0.99f;
            if (rlogarg < 0.01) rlogarg = 0.01f;


            float4 wrr;
            r32 ww[3][3];

            for (int tes=0; tes<3; tes++) {
                for (int see=0; see<3; see++) {
                    ww[tes][see] = gaussian_TEA_fast<4> (veloc1.i > veloc2.i, veloc1.i+tes, veloc2.i+see);
                }
            }

            wrr.w = (ww[0][0]+ww[1][1]+ww[2][2])/3.0f;
            wrr.x = (ww[0][0]-wrr.w)*delx + 0.5f*(ww[0][1]+ww[1][0])*dely + 0.5f*(ww[0][2]+ww[2][0])*delz;  // traceless symmetric 0
            wrr.y = 0.5f*(ww[1][0]+ww[0][1])*delx + (ww[1][1]-wrr.w)*dely + 0.5f*(ww[1][2]+ww[2][1])*delz;  // traceless symmetric 1
            wrr.z = 0.5f*(ww[2][0]+ww[0][2])*delx + 0.5f*(ww[2][1]+ww[1][2])*dely + (ww[2][2]-wrr.w)*delz;  // traceless symmetric 2

            r32 fforce = - temp[type]*(0.25f/(1.0f-rr)/(1.0f-rr)-0.25f+rr)/lambda/ra + kph/rlogarg + (sigc[type]*wrr.w - gamc[type]*vv)/ra;
            r32 fxij = delx*fforce - gamt[type]*dvx + sigt[type]*wrr.x/ra;
            r32 fyij = dely*fforce - gamt[type]*dvy + sigt[type]*wrr.y/ra;
            r32 fzij = delz*fforce - gamt[type]*dvz + sigt[type]*wrr.z/ra;

            fxi += fxij;
            fyi += fyij;
            fzi += fzij;
        }
        force_x[i] += fxi;
        force_y[i] += fyi;
        force_z[i] += fzi;

    }
}


void MesoBondWLCPowAllVisc::compute(int eflag, int vflag)
{
    if( !coeff_alloced ) allocate_gpu();

    static GridConfig grid_cfg, grid_cfg_EV;
    if( !grid_cfg_EV.x ) {
        grid_cfg_EV = meso_device->occu_calc.right_peak( 0, gpu_bond_wlcpowallvisc<1>, ( atom->nbondtypes + 1 ) * 8 * sizeof( r32 ), cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_bond_wlcpowallvisc<1>, cudaFuncCachePreferL1 );
        grid_cfg    = meso_device->occu_calc.right_peak( 0, gpu_bond_wlcpowallvisc<0>, ( atom->nbondtypes + 1 ) * 8 * sizeof( r32 ), cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_bond_wlcpowallvisc<0>, cudaFuncCachePreferL1 );
    }

    float3 period;
    period.x = ( domain->xperiodic ) ? ( domain->xprd ) : ( 0. );
    period.y = ( domain->yperiodic ) ? ( domain->yprd ) : ( 0. );
    period.z = ( domain->zperiodic ) ? ( domain->zprd ) : ( 0. );

    if( eflag || vflag ) {
        gpu_bond_wlcpowallvisc<1> <<< grid_cfg_EV.x, grid_cfg_EV.y, ( atom->nbondtypes + 1 ) * 8 * sizeof( r32 ), meso_device->stream() >>> (
            meso_atom->tex_coord_merged,
            meso_atom->dev_force(0), meso_atom->dev_force(1), meso_atom->dev_force(2),
            meso_atom->tex_veloc_merged,
            meso_atom->dev_nbond,
            meso_atom->dev_bond_mapped,
            meso_atom->dev_bond_r0,
            dev_temp,
            dev_r0,
            dev_mu_targ,
            dev_qp,
            dev_gamc,
            dev_gamt,
            dev_sigc,
            dev_sigt,
            period,
            meso_atom->dev_bond_mapped.pitch_elem(),
            atom->nbondtypes,
            atom->nlocal );
    }   else {
        gpu_bond_wlcpowallvisc<0> <<< grid_cfg.x, grid_cfg.y, ( atom->nbondtypes + 1 ) * 8 * sizeof( r32 ), meso_device->stream() >>> (
            meso_atom->tex_coord_merged,
            meso_atom->dev_force(0), meso_atom->dev_force(1), meso_atom->dev_force(2),
            meso_atom->tex_veloc_merged,
            meso_atom->dev_nbond,
            meso_atom->dev_bond_mapped,
            meso_atom->dev_bond_r0,
            dev_temp,
            dev_r0,
            dev_mu_targ,
            dev_qp,
            dev_gamc,
            dev_gamt,
            dev_sigc,
            dev_sigt,
            period,
            meso_atom->dev_bond_mapped.pitch_elem(),
            atom->nbondtypes,
            atom->nlocal );
    }

}

void MesoBondWLCPowAllVisc::coeff(int narg, char **arg)
{
    if (narg != 7) error->all(FLERR,"Incorrect args for bond coefficients");
    if (!allocated) allocate_cpu();

    int ilo, ihi;
    force->bounds(arg[0],atom->nbondtypes,ilo,ihi);

    int iarg = 0;
    float temp_one = force->numeric(FLERR,arg[++iarg]);
    float r0_one = force->numeric(FLERR,arg[++iarg]);
    float mu_one = force->numeric(FLERR,arg[++iarg]);
    float qp_one = force->numeric(FLERR,arg[++iarg]);
    float gamc_one = force->numeric(FLERR,arg[++iarg]);
    float gamt_one = force->numeric(FLERR,arg[++iarg]);

    int count = 0;
    for (int i = ilo; i <= ihi; i++) {
        temp[i] = temp_one;
        r0[i] = r0_one;
        mu_targ[i] = mu_one;
        qp[i] = qp_one;
        gamc[i] = gamc_one;
        gamt[i] = gamt_one;
        setflag[i] = 1;
        count++;
    }
    if (count == 0) error->all(FLERR,"Incorrect args for bond coefficients");

    double sdtt = sqrt(update->dt);
    for (int i = 1; i <= atom->nbondtypes; i++) {
        if (setflag[i] == 0) error->all(FLERR,"All bond coeffs are not set");
        if (gamt[i]>3.0*gamc[i]) error->all(FLERR,"Gamma_t > 3*Gamma_c");
        sigc[i] = sqrt(2.0*temp[i]*(3.0*gamc[i]-gamt[i]))/sdtt;
        sigt[i] = 2.0*sqrt(gamt[i]*temp[i])/sdtt;
    }
}

double MesoBondWLCPowAllVisc::equilibrium_distance(int i)
{
    return r0[i];
}

void MesoBondWLCPowAllVisc::write_restart(FILE *fp)
{
    fwrite(&temp[1],sizeof(float),atom->nbondtypes,fp);
    fwrite(&r0[1],sizeof(float),atom->nbondtypes,fp);
    fwrite(&mu_targ[1],sizeof(float),atom->nbondtypes,fp);
    fwrite(&qp[1],sizeof(float),atom->nbondtypes,fp);
    fwrite(&gamc[1],sizeof(float),atom->nbondtypes,fp);
    fwrite(&gamt[1],sizeof(float),atom->nbondtypes,fp);
}

void MesoBondWLCPowAllVisc::read_restart(FILE *fp)
{
    allocate_cpu();

    if (comm->me == 0) {
        fread(&temp[1],sizeof(float),atom->nbondtypes,fp);
        fread(&r0[1],sizeof(float),atom->nbondtypes,fp);
        fread(&mu_targ[1],sizeof(float),atom->nbondtypes,fp);
        fread(&qp[1],sizeof(float),atom->nbondtypes,fp);
        fread(&gamc[1],sizeof(float),atom->nbondtypes,fp);
        fread(&gamt[1],sizeof(float),atom->nbondtypes,fp);
    }
    MPI_Bcast(&temp[1],atom->nbondtypes,MPI_FLOAT,0,world);
    MPI_Bcast(&r0[1],atom->nbondtypes,MPI_FLOAT,0,world);
    MPI_Bcast(&mu_targ[1],atom->nbondtypes,MPI_FLOAT,0,world);
    MPI_Bcast(&qp[1],atom->nbondtypes,MPI_FLOAT,0,world);
    MPI_Bcast(&gamc[1],atom->nbondtypes,MPI_FLOAT,0,world);
    MPI_Bcast(&gamt[1],atom->nbondtypes,MPI_FLOAT,0,world);


    for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

void MesoBondWLCPowAllVisc::write_data(FILE *fp)
{
    for (int i = 1; i <= atom->nbondtypes; i++)
        fprintf(fp,"%d %g %g %g %g %g %g %g\n",i,temp[i],r0[i],mu_targ[i],qp[i],gamc[i],gamt[i]);
}

void MesoBondWLCPowAllVisc::settings(int narg, char **arg)
{}

double MesoBondWLCPowAllVisc::single(int type, double rsq, int i, int j, double &fforce)
{
    return 0;
}
