#!/bin/bash

NPROC=$1

cd bin
mpiexec -n $NPROC ./xewf2d 
