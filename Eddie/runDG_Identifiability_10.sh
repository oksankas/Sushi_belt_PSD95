#!/bin/sh
# Grid Engine options
# 8 compute cores, total 64 GB memory. Runtime limit of 48 hours.
#
#$ -cwd
#$ -pe sharedmem 8
#$ -l h_vmem=8G
#$ -l h_rt=47:59:00

# expected Eddie command (take into account that with valnum=20 there are 21 parameters in the set, so
# the size of the array is not 20*25=500 but is 21*25=525) :
# qsub -t 1-294 runDGIdentifiability.sh identDG20240815 20

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load anaconda
conda activate mypython
nrnivmodl

odir=$1
valnum=$2
seed=255

mkdir -p odir
#cp Edita_20reg_1dv_model.py runEdita_20reg_1dv_3M_Identifiability.py Identifiability_PSO_GA.py ${odir}/
export odir valnum seed
echo "$odir" "$valnum" "$seed"

# Run the program
python3 runDG_Identifiability_10dv.py
