#!/bin/sh
# Grid Engine options
# 8 compute cores, total 64 GB memory. Runtime limit of 48 hours.
#
#$ -cwd
#$ -pe sharedmem 8
#$ -l h_vmem=8G
#$ -l h_rt=47:59:00

# expected Eddie command:
# qsub runEdita3M_dual_soma_optimization_restart.sh dual_soma_PSO_GA_20240902 dual_soma_PSO_GA_20240826/Edita_20reg_1dv_dual_soma_optimization_pso10_nm10_ga10_best_pso_ga.csv

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load anaconda
conda activate mypython
nrnivmodl

chunkSize=512
seed=255

odir=$1
ifile = $2
mkdir -p odir
export odir seed ifile
echo "$odir" "$seed" "$ifile"

# Run the program
python3 runEdita3M_dual_soma_optimization_restart.py