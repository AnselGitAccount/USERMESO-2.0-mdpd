

#ifdef FIX_CLASS

FixStyle(addconf/tdpd/meso,MesoFixAddConfTDPD)

#else

#ifndef LMP_MESO_FIX_ADD_CONF_TDPD
#define LMP_MESO_FIX_ADD_CONF_TDPD

#include "fix.h"

namespace LAMMPS_NS {

class MesoFixAddConfTDPD : public Fix, protected MesoPointers {
public:
    MesoFixAddConfTDPD(class LAMMPS *, int, char **);
    virtual int setmask();
    virtual void init();
    virtual void setup(int);
    virtual void post_force(int);

private:
    int n_species;
    DeviceScalar<r32> cf;       // concentration flux
    r64 lo, hi;                 // location range
    int index;
};

}

#endif

#endif
