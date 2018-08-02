#ifdef FIX_CLASS

FixStyle(aveforce/meso,MesoFixAveForce)

#else

#ifndef LMP_MESO_FIX_AVE_FORCE
#define LMP_MESO_FIX_AVE_FORCE

#include "fix.h"

namespace LAMMPS_NS {

class MesoFixAveForce : public Fix, protected MesoPointers {
public:
    MesoFixAveForce(class LAMMPS *, int, char **);
    virtual int setmask();
    virtual void init();
    virtual void setup(int);
    virtual void post_force(int);

private:
    int xstyle, ystyle, zstyle;
    r64 xvalue, yvalue, zvalue;
    DeviceScalar<r64> foriginal;
    DeviceScalar<r64> foriginal_all;
};

}

#endif

#endif
