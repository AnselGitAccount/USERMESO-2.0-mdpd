#ifdef BOND_CLASS

BondStyle(wlc_pow_all_visc/meso,MesoBondWLCPowAllVisc)

#else

#ifndef LMP_MESO_BOND_WLCPOWALLVISC
#define LMP_MESO_BOND_WLCPOWALLVISC

#include "bond.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoBondWLCPowAllVisc : public Bond, protected MesoPointers
{
public:
    MesoBondWLCPowAllVisc(class LAMMPS *);
    ~MesoBondWLCPowAllVisc();

    void settings(int, char **);
    void coeff(int, char **);
    void compute(int, int);
    double equilibrium_distance(int);
    void write_restart(FILE *);
    void read_restart(FILE *);
    void write_data(FILE *);
    double single(int, double, int, int, double &);

protected:
    DeviceScalar<r32> dev_temp;
    DeviceScalar<r32> dev_r0;
    DeviceScalar<r32> dev_mu_targ;
    DeviceScalar<r32> dev_qp;
    DeviceScalar<r32> dev_gamc;
    DeviceScalar<r32> dev_gamt;
    DeviceScalar<r32> dev_sigc;
    DeviceScalar<r32> dev_sigt;
    std::vector<r32> temp, r0, mu_targ, qp, gamc, gamt, sigc, sigt;

    void    allocate_gpu();
    void    allocate_cpu();
    int     coeff_alloced;
};

}

#endif

#endif
