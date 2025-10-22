#!/bin/sh
# Grid Engine options
# 8 compute cores, total 64 GB memory. Runtime limit of 48 hours.
#
#$ -cwd
#$ -pe sharedmem 8
#$ -l h_vmem=8G
#$ -l h_rt=47:59:00

# expected Eddie command:
# qsub -t 1-128 runSobol20reg.sh sobol20240621 256

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load anaconda
conda activate mypython
nrnivmodl

chunkSize=512
seed=255

odir=$1
if [ -n "$2" ]
  then
    chunkSize=$2
    echo "ChunkSize supplied $chunkSize"
fi

mkdir -p odir
export odir chunkSize seed
echo "$SGE_TASK_ID" "$odir" "$chunkSize" "$seed"

# Run the program
python3 run_20reg_1dv_3M_Sobol.py