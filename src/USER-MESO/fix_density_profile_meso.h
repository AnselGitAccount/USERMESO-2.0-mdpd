#ifdef FIX_CLASS

FixStyle(density/meso,MesoFixDensityProf)

#else

#ifndef LMP_MESO_FIX_DENSITY_PROF
#define LMP_MESO_FIX_DENSITY_PROF

#include "fix.h"

namespace LAMMPS_NS {

class MesoFixDensityProf : public Fix, protected MesoPointers {
public:
	MesoFixDensityProf(class LAMMPS *, int, char **);
    ~MesoFixDensityProf();
    virtual int setmask();
    virtual void setup(int);
    virtual void post_integrate();

protected:
    std::string output;
    int  n_bin, along;
    r64  bin_size, every, window;
    bigint last_dump_time, count_frames;
    DeviceScalar<uint> dev_count;
    virtual void dump( bigint tstamp );

    void compute();
};

}

#endif

#endif
