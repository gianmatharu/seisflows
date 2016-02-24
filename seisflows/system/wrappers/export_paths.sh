#!/bin/bash

# Wrapper to pbsdsh python scripts. Exports
# required env variables.

# set env variables

PATH=$1:PATH
export LD_LIBRARY_PATH=$2
export TASK_ID=$3

# set python arguments
exe=$4
output=$5
classname=$6
funcname=$7
pythonpath=$8

$exe $output $classname $funcname $pythonpath
