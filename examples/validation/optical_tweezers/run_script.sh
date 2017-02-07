#!/bin/bash


sf=24.4686		    # 1pN to 24.4686; unit mapping
epsilonN=10 		# 10 particles; So divide by 10
sf=$(echo "$sf/$epsilonN" | bc -l)


for c in {0..200..10}
do
	sforce=$(echo "scale=4; ($sf * $c)/1" | bc -l)	
	mpirun -np 2 ../../../src/lmp_meso -in optical_tweezers.in -var sforce $sforce	
done


