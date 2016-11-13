#!/bin/bash

for f in $(ls *.cpp *.h)
do
  echo $f
  d=`diff $f /home/ytang/src/LAMMPS-current/src/$f`
  if [ -n "$d" ]; then
    echo -e "\e[00;31m$d\e[00m"
  fi
done
