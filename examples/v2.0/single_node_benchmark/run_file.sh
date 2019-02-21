#!/bin/bash

EXEC=../../src/lmp_meso


function avg {
	echo $sum
        awk "{sum+=\$$1} END { print sum/NR}"
}


printf "RBC-count, Hematocrit, Volume, Loop time \n"


for fp in tDPDmeso_RBCs_in_box_*Heme35_Vol{8192,16384,32768,65536}.data
do

 	nRBC=$(echo ${fp} | awk -v FS="(nRBC|_H)" '{print $2}' | bc -l)
 	Hematocrit=$(echo ${fp} | awk -v FS="(Heme|_V)" '{print $2}' | bc -l)
 	Vol=$(echo ${fp} | awk -v FS="(Vol|.data)" '{print $2}' | bc -l)
 	
 	Looptime=$(OMP_NUM_THREADS=6 GOMP_CPU_AFFINITY=0-5 mpirun -np 1 ${EXEC} -in tDPD_RBC_spec1_Single_GPU.in -var datafile ${fp} $f 2>temp_output.file | grep Loop | tail -n 5 | avg 4 | awk '{print $1}')
 	
 	echo $nRBC, $Hematocrit, $Vol, $Looptime
done


rm temp_output.file

