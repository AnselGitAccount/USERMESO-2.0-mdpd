#ifdef DIHEDRAL_CLASS

DihedralStyle(bend/meso,MesoDihedralBend)

#else

#ifndef LMP_MESO_DIHEDRAL_BEND
#define LMP_MESO_DIHEDRAL_BEND

#include "dihedral_harmonic.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoDihedralBend : public DihedralHarmonic, protected MesoPointers
{
public:
    MesoDihedralBend(class LAMMPS *);
    ~MesoDihedralBend();

    void compute( int eflag, int vflag );
    void alloc_coeff();
    void coeff(int, char **);

protected:
    DeviceScalar<r32> dev_k, dev_theta0;
    r32 *k0,*theta0;
    int coeff_alloced;
    void allocate();
};

}

#endif

#endif
