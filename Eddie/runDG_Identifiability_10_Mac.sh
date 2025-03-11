#!/bin/sh
# ./runDGIdentifiability.sh identDG20240820 20

# Initialise the environment modules
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
