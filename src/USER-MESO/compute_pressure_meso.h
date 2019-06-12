#ifdef COMPUTE_CLASS

ComputeStyle(pressure/meso,MesoComputePressure)

#else

#ifndef LMP_MESO_COMPUTE_PRESSURE
#define LMP_MESO_COMPUTE_PRESSURE

#include "compute.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoComputePressure : public Compute, protected MesoPointers {
public:
    MesoComputePressure(class LAMMPS *, int, char **);
    ~MesoComputePressure();
    virtual void init();
    virtual void setup();
    virtual double compute_scalar();

private:
    int keflag,pairflag,bondflag,angleflag,dihedralflag,improperflag;
    int fixflag,kspaceflag;
    double pfactor;
    HostScalar<r64> p;
    DeviceScalar<r64> per_atom_pressure;
    char *id_temp;

    Compute *temperature;
};

}

#endif

#endif

