
#ifdef PAIR_CLASS

PairStyle(tdpd/meso,MesoPairTDPD)

#else

#ifndef LMP_MESO_PAIR_TDPD
#define LMP_MESO_PAIR_TDPD

#include "pair.h"
#include "meso.h"

namespace LAMMPS_NS {

namespace TDPD_COEFFICIENTS {
const static int n_chemcoeff = 3;       // Number of specie-dependent coefficients.
const static int n_coeff  = 7;
const static int p_cut    = 0;
const static int p_cutsq  = 1;
const static int p_cutinv = 2;
const static int p_s1     = 3;
const static int p_a0     = 4;
const static int p_gamma  = 5;
const static int p_sigma  = 6;
const static int p_cutc   = 7;          // species specific
const static int p_kappa  = 8;          // species specific
const static int p_s2     = 9;          // species specific
}

class MesoPairTDPD : public Pair, protected MesoPointers {
public:
    MesoPairTDPD(class LAMMPS *);
    ~MesoPairTDPD();
    void   compute(int, int);
    void   compute_bulk(int, int);
    void   compute_border(int, int);
    void   settings(int, char **);
    void   coeff(int, char **);
    void   init_style();
    double init_one(int, int);
    void   write_restart(FILE *);
    void   read_restart(FILE *);
    void   write_restart_settings(FILE *);
    void   read_restart_settings(FILE *);
    double single(int, int, int, int, double, double, double, double &);

protected:
    int      seed;
    bool     coeff_ready;
    DeviceScalar<r64> dev_coefficients;

    double   cut_global;
    double **cut;
    double **cutinv;
    double **s1;
    double **a0;
    double **gamma;
    double **sigma;
    double ***cutc;                 // species specific
    double ***kappa;                // species specific
    double ***s2;                   // species specific
    int n_species;                  // Number of Chemical Species.

    class RanMars *random;

    virtual void allocate();
    virtual void prepare_coeff();
    virtual void compute_kernel(int, int, int, int);
    virtual uint seed_now();
};

}

#endif

#endif


