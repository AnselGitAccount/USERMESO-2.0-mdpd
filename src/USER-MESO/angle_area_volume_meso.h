#ifdef ANGLE_CLASS

AngleStyle(areavolume/meso,MesoAngleAreaVolume)

#else

#ifndef LMP_MESO_ANGLE_AREAVOLUME
#define LMP_MESO_ANGLE_AREAVOLUME

#include "meso.h"
#include "angle.h"

namespace LAMMPS_NS {

class MesoAngleAreaVolume : public Angle, protected MesoPointers
{
public:
    MesoAngleAreaVolume(class LAMMPS *);
    ~MesoAngleAreaVolume();
    void compute( int eflag, int vflag );
    void coeff(int, char **);
    double equilibrium_angle(int);
    void write_restart(FILE *);
    void read_restart(FILE *);
    double single(int, int, int, int);

protected:
    DeviceScalar<r32> dev_ka, dev_a0, dev_kv, dev_v0, dev_kl;
    DeviceScalar<int> dev_ttyp, dev_nm;
    DeviceScalar<r64> dev_datt, dev_dath, dev_datt_laststep;
    r32 *ka, *a0, *kv, *v0, *kl;
    int init_on;
    HostScalar<r64> datt, dath;
    int *ttyp, *ttyp1;
    void    alloc_coeff();
    void    allocate();
    int     coeff_alloced;
    int     nm, n_mol;
    MPI_Request avrequest;
    MPI_Status avstatus;
};

}

#endif

#endif
