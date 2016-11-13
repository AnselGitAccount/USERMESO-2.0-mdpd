#ifdef FIX_CLASS

FixStyle(valve/meso,FixFluxValve)

#else

#ifndef LMP_MESO_FIX_FLUX_VALVE
#define LMP_MESO_FIX_FLUX_VALVE

#include "fix.h"
#include "meso.h"

namespace LAMMPS_NS {

class FixFluxValve : public Fix, protected MesoPointers
{
public:
    FixFluxValve(LAMMPS *lmp, int narg, char **arg);

    virtual int setmask();
    virtual void post_force(int);

protected:
    r64 lo[3], hi[3];
    r64 vo[3], drag;
};

}

#endif

#endif
