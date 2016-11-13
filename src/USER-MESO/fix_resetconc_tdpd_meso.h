
#ifdef FIX_CLASS

FixStyle(resetconc/tdpd/meso,MesoFixResetConcTDPD)

#else

#ifndef LMP_MESO_FIX_RESET_CONC_TDPD
#define LMP_MESO_FIX_RESET_CONC_TDPD

#include "fix.h"

namespace LAMMPS_NS {

class MesoFixResetConcTDPD : public Fix, protected MesoPointers {
public:
    MesoFixResetConcTDPD(class LAMMPS *, int, char **);
    ~MesoFixResetConcTDPD();
    virtual int setmask();
    virtual void init();
    virtual void setup(int);
    virtual void post_force(int);

private:
    int n_species;
    DeviceScalar<r32> c;        //concentration
    bigint begin;   // the step of which reset is imposed.
    bigint nextreset; // the step of which next reset is imosed.
    bigint interval; // interval of which reset is imposed. if 0, only reset once at begin; else, nextreset = currstep + interval.
    bigint currstep;
};

}

#endif

#endif
