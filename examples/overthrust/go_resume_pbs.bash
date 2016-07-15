#!/bin/bash
#PBS -S /bin/bash

## job name and output file
#PBS -N inversion
#PBS -j oe
#PBS -o $PBS_JOBID.o

###########################################################
# USER PARAMETERS

#PBS -l mem=66gb
#PBS -l nodes=6:ppn=8,walltime=00:05:00:00
#PBS -M gian@ualberta.ca
#PBS -m bea
##PBS -q debug

###########################################################

# Script to automate adjoint simulations. 

cd $PBS_O_WORKDIR
cat $PBS_NODEFILE > nodefile
module load intel
sfresume
