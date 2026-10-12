#!/bin/bash

export OMP_NUM_THREADS=1
job=$SLURM_ARRAY_TASK_ID
#additional_bash_arg=$1

tdir=$(mktemp -d)

## Example commands for transferring input files from network storage to the execution node
#input_dir=/storage/dune/dune10kt_1x2x6_mcc11-0_samples/mcc11/v2_pndr/
#input_file=Pandora_Events_${job}.pndr
#cp ${input_dir}/${input_file} ${tdir}/

## Setting up a temporary working directory on the execution node and cding into it
output_dir="${tdir}/run_${job}"
mkdir $output_dir && cd $output_dir

# Note - need to update this to point at the absolute path to your Python environment
source <path to your Python environment>/bin/activate
# Get output that confirms which Python environment is running
which python

python my_python_script.py

echo -e "Listing"
ls -l ${tdir}/run_${job}

# Example compression of any output files you want to copy back to network storage
tar czf output_files_${job}.tar.gz *.npy *.npz

# cd back to the directory you were in when you submitted
cd $SLURM_SUBMIT_DIR

# Copy the tar file with all of your outputs from the execution node to your submission directory
cp ${tdir}/run_${job}/output_files_${job}.tar.gz .

# Remove the temporary directory on the execution node (not really needed, but good practice to clean up)
rm -rf ${tdir}

