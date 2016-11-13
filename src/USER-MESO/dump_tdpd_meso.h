
#ifdef DUMP_CLASS

DumpStyle(tdpd/meso,DumpTDPD)

#else

#ifndef LMP_DUMP_TDPD_H
#define LMP_DUMP_TDPD_H

#include "dump.h"

namespace LAMMPS_NS {

class DumpTDPD : public Dump {
public:
    DumpTDPD(LAMMPS *, int, char**);

private:
    int scale_flag;            // 1 if atom coords are scaled, 0 if no
    int image_flag;            // 1 if append box count to atom coords, 0 if no
    int unwrap_flag;           // 1 if unwrap particle positions by box count

    char *columns;             // column labels

    int nth_species;                   // the chemical to dump

    void init_style();
    int modify_param(int, char **);
    void write_header(bigint);
    void pack(int *);
    void write_data(int, double *);

    typedef void (DumpTDPD::*FnPtrHeader)(bigint);
    FnPtrHeader header_choice;           // ptr to write header functions
    void header_binary(bigint);
    void header_binary_triclinic(bigint);
    void header_item(bigint);
    void header_item_triclinic(bigint);

    typedef void (DumpTDPD::*FnPtrPack)(int *);
    FnPtrPack pack_choice;               // ptr to pack functions
    void pack_unwrap(int *);
    void pack_scale_image(int *);
    void pack_scale_noimage(int *);
    void pack_noscale_image(int *);
    void pack_noscale_noimage(int *);
    void pack_scale_image_triclinic(int *);
    void pack_scale_noimage_triclinic(int *);

    typedef void (DumpTDPD::*FnPtrData)(int, double *);
    FnPtrData write_choice;              // ptr to write data functions
    void write_binary(int, double *);
    void write_image(int, double *);
    void write_noimage(int, double *);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

*/
