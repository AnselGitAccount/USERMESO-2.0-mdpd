#ifdef FIX_CLASS

FixStyle(setforce/meso,MesoFixSetForce)

#else

#ifndef LMP_MESO_FIX_SET_FORCE
#define LMP_MESO_FIX_SET_FORCE

#include "fix.h"

namespace LAMMPS_NS {

class MesoFixSetForce : public Fix, protected MesoPointers {
public:
    MesoFixSetForce(class LAMMPS *, int, char **);
    virtual int setmask();
    virtual void init();
    virtual void setup(int);
    virtual void post_force(int);

private:
    r64 fx, fy, fz;
    int fx_flag, fy_flag, fz_flag;
};

}

#endif

#endif
