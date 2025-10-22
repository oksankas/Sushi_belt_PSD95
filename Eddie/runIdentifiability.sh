#!/bin/sh
# Grid Engine options
# 8 compute cores, total 64 GB memory. Runtime limit of 48 hours.
#
#$ -cwd
#$ -pe sharedmem 8
#$ -l h_vmem=8G
#$ -l h_rt=47:59:00


# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load anaconda
conda activate mypython
nrnivmodl


# Run the program
python3 run3W_CA1_Identifiability.py