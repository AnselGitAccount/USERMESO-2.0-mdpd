
#ifdef COMPUTE_CLASS

ComputeStyle(conc/tdpd/meso,MesoComputeConcTDPD)

#else

#ifndef LMP_MESO_COMPUTE_CONCEN_TDPD
#define LMP_MESO_COMPUTE_CONCEN_TDPD

#include "compute.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoComputeConcTDPD : public Compute, protected MesoPointers {
public:
    MesoComputeConcTDPD(class LAMMPS *, int, char **);
    ~MesoComputeConcTDPD();
    virtual void init() {}
    virtual void setup();
    virtual double compute_scalar();

private:
    int nth_species;
    DeviceScalar<r32>    conc;      // concentration.
    DeviceScalar<int>    c;         // look like a counter.
};

}

#endif

#endif
