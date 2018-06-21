#ifdef ATOM_CLASS

AtomStyle(mdpd/atomic/meso,AtomVecMDPDAtomic)

#else

#ifndef LMP_MESO_ATOM_VEC_MDPD_ATOMIC
#define LMP_MESO_ATOM_VEC_MDPD_ATOMIC

#include "meso.h"
#include "atom_vec_dpd_atomic_meso.h"

namespace LAMMPS_NS {

class AtomVecMDPDAtomic: public AtomVecDPDAtomic
{
public:
    AtomVecMDPDAtomic(LAMMPS *lmp);
    ~AtomVecMDPDAtomic() {}

    virtual void copy(int, int, int);
    virtual void grow(int);
    virtual void grow_cpu(int);
    virtual void grow_device(int);
    virtual int  pack_comm(int, int *, double *, int, int *);
    virtual int  pack_comm_vel(int, int *, double *, int, int *);
    virtual void unpack_comm(int, int, double *);
    virtual void unpack_comm_vel(int, int, double *);
    virtual int  pack_border(int, int *, double *, int, int *);
    virtual int  pack_border_vel(int, int *, double *, int, int *);
    virtual void unpack_border(int, int, double *);
    virtual void unpack_border_vel(int, int, double *);
    virtual int  pack_exchange(int, double *);
    virtual int  unpack_exchange(double *);
    virtual int  size_restart();
    virtual int  pack_restart(int, double *);
    virtual int  unpack_restart(double *);
//  virtual void create_atom(int, double *);
    virtual void data_atom(double *, int, char **);
//  virtual int  data_atom_hybrid(int, char **);
    virtual void pin_host_array();
    virtual void unpin_host_array();
    virtual void force_clear( AtomAttribute::Descriptor, int );
    virtual void dp2sp_merged( int seed, int p_beg, int p_end, bool offset = false );
    virtual bigint memory_usage();

    double ested_max_particle_density;

protected:
    DeviceScalar<r64> dev_rho;
    DeviceScalar<r64> dev_phi;
    Pinned<r64> dev_rho_pinned;
    Pinned<r64> dev_phi_pinned;
//    DeviceScalar<float4> dev_therm_merged;
    double *rho;
    double *phi;


    virtual void transfer_impl( std::vector<CUDAEvent> &events, AtomAttribute::Descriptor per_atom_prop, TransferDirection direction, int p_beg, int n_atom, int p_stream, int p_inc, int* permute_to, int* permute_from, int action, bool streamed);
};


}

#endif

#endif
