#ifdef PAIR_CLASS

PairStyle(mdpd/meso,MesoPairMDPD)

#else

#ifndef LMP_MESO_PAIR_MDPD
#define LMP_MESO_PAIR_MDPD

#include "pair.h"
#include "meso.h"

namespace LAMMPS_NS {

namespace MDPD_COEFFICIENTS {
const static int n_coeff  = 6;
const static int p_cut    = 0;
const static int p_cut_r  = 1;
const static int p_A_att  = 2;
const static int p_B_rep  = 3;
const static int p_gamma  = 4;
const static int p_sigma  = 5;
}

class MesoPairMDPD : public Pair, protected MesoPointers {
public:
    MesoPairMDPD(class LAMMPS *);
    ~MesoPairMDPD();
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
    void   write_data(FILE *);
    void   read_data(FILE *);
//    double single(int, int, int, int, double, double, double, double &);

protected:
    int      seed;
    bool     coeff_ready;
    DeviceScalar<r64> dev_coefficients;

    double   cut_global, temperature;
    int      flag_arb_bc;
    int 	 mobile_group, mobile_groupbit;
    int 	 wall_group, wall_groupbit;
    double **cut;
    double **cut_r;
    double **A_att;
    double **B_rep;
    double **gamma;
    double **sigma;

    class RanMars *random;

    virtual void allocate();
    virtual void prepare_coeff();
    virtual void compute_kernel(int, int, int, int);
    virtual uint seed_now();
};

}

#endif

#endif
