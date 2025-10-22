#!/bin/sh
# Grid Engine options
# 8 compute cores, total 64 GB memory. Runtime limit of 48 hours.
#
#$ -cwd
#$ -pe sharedmem 8
#$ -l h_vmem=8G
#$ -l h_rt=47:59:00

# expected Eddie command:
# qsub run3M_DG_optimization.sh DG_PSO_GA_20240802

# Initialise the environment modules
#. /etc/profile.d/modules.sh

# Load Python
#module load anaconda
#conda activate mypython
nrnivmodl

chunkSize=512
seed=255

odir=$1

mkdir -p odir
#cp CA1_20reg_1dv_model.py runCA1_20reg_1dv_3M_Identifiability.py Identifiability_PSO_GA.py ${odir}/
export odir seed
echo "$odir" "$seed"

# Run the program
python3 run3M_DG_optimization.py