#ifdef FIX_CLASS

FixStyle(wall/region/meso,MesoFixWallRegion)

#else

#ifndef LMP_MESO_FIX_WALL_REGION
#define LMP_MESO_FIX_WALL_REGION

#include "fix.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoFixWallRegion : public Fix, protected MesoPointers {
public:
	MesoFixWallRegion(class LAMMPS *, int, char **);
	~MesoFixWallRegion();
    virtual int setmask();
    virtual void init();
    virtual void setup(int);
    virtual void pre_exchange();
    virtual void end_of_step();

protected:
    int regstyle;
    int iregion;
    char *idregion;

    void bounce_back();
};

}

#endif

#endif
