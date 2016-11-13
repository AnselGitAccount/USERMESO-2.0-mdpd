

#ifdef FIX_CLASS

FixStyle(nve/tdpd/meso,FixNVETDPDMeso)

#else

#ifndef LMP_MESO_FIX_NVE_TDPD
#define LMP_MESO_FIX_NVE_TDPD

#include "fix_nve.h"
#include "meso.h"

namespace LAMMPS_NS {

class FixNVETDPDMeso : public Fix, protected MesoPointers
{
public:
    FixNVETDPDMeso(class LAMMPS *, int, char **);
    virtual void init();
    virtual int setmask();
    virtual void initial_integrate(int);
    virtual void final_integrate();
    virtual void reset_dt();
protected:
    double dtv;
};

}

#endif

#endif
