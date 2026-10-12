#!/bin/bash

# To submit to the EPP dedicated machines
sbatch -p epp --ntasks=1 --mem-per-cpu=2012 --time=03:00:00 --array=0-9 ./node.sh

# To submit to the general cluster of workstations
#sbatch --ntasks=1 --mem-per-cpu=2012 --time=03:00:00 --array=0-9 ./node.sh
