#ifdef FIX_CLASS

FixStyle(aboundary/rho/meso,FixArbitraryBoundaryRho)

#else

#ifndef LMP_MESO_FIX_ARBITRARY_RHO_BOUNDARY
#define LMP_MESO_FIX_ARBITRARY_RHO_BOUNDARY

#include "fix.h"
#include "meso.h"

namespace LAMMPS_NS {

/*
 * In addition to calculate a particle's proximity to solid-wall (as in fix_abounary_meso.cu/h),
 * this fix also calculate a particle's proximity to other solvent particles.
 * The proximity is quantified as particle density - rho.
 *
 * Ansel
 */

class FixArbitraryBoundaryRho : public Fix, protected MesoPointers
{
public:
    FixArbitraryBoundaryRho(LAMMPS *lmp, int narg, char **arg);

    virtual void init();
    virtual int setmask();
    virtual void pre_force(int);
    virtual void setup_pre_force(int);
    virtual void post_integrate();
    virtual int pack_comm(int, int *, double *, int, int *);
    virtual void unpack_comm(int, int, double *);


protected:
    int flag_compute_arb_bc;
    int wall_group, wall_groupbit;
    int mobile_group, mobile_groupbit;
    r64 rc, sigma, frac, xcut;
    r64 cut_rho, rho_factor, rho_wall;
    r64 cut_phi, phi_factor, phi_c;
    r64 dw_factor, dtv, dtf;
};

}

#endif

#endif
