#!/bin/bash

EXEC=../../src/lmp_meso


function avg {
	echo $sum
        awk "{sum+=\$$1} END { print sum/NR}"
}


printf "Simulation system \t Loop time \n"


for fp in tDPDmeso_RBCs_in_box_*Heme35_Vol{8192,16384,32768,65536}.data
do
    printf "%s \t" ${fp}
 	printf "%.4f\n" $(OMP_NUM_THREADS=6 GOMP_CPU_AFFINITY=0-5 mpirun -np 1 ${EXEC} -in tDPD_RBC_spec1_Single_GPU.in -var datafile ${fp} $f 2>temp_output.file | grep Loop | tail -n 5 | avg 4 | awk '{print $1}')
done


rm temp_output.file

