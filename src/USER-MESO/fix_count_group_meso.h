#ifdef FIX_CLASS

FixStyle(countgroup/meso,MesoFixCountGroup)

#else

#ifndef LMP_MESO_FIX_COUNT_GROUP
#define LMP_MESO_FIX_COUNT_GROUP

#include "fix.h"

namespace LAMMPS_NS {

class MesoFixCountGroup : public Fix, protected MesoPointers {
public:
    MesoFixCountGroup(class LAMMPS *, int, char **);
    ~MesoFixCountGroup();
    virtual int setmask();
    virtual void setup(int);
    virtual void post_integrate();

protected:
    std::vector<int> target_groupbits;
    std::string output;
    int me, nprocs, ngroups;
    bigint dump_time, every;
    DeviceScalar<uint> dev_count;    // stores count(group) for all groups.
    DeviceScalar<int>  dev_groupbits;
    std::ofstream fout;

    virtual void dump( bigint tstamp );
    void compute();
};

}

#endif

#endif
