#!/bin/bash

currentdir=$PWD
echo $currentdir

for i in `seq -f "%03g" 1 16`;
   do
      cd $i 
      mv $currentdir/$i/traces/obs/* $currentdir/$i/
      
      rm -rf INPUT
      rm -rf bin
      rm -rf traces/syn
      rm -rf traces/adj
      rm -rf traces
      cd $currentdir
   done    
