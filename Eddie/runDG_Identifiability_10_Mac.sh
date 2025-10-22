#!/bin/sh
# ./runDGIdentifiability.sh identDG20240820 20

# Initialise the environment modules
nrnivmodl

odir=$1
valnum=$2
seed=255

mkdir -p odir
export odir valnum seed
echo "$odir" "$valnum" "$seed"

# Run the program
python3 runDG_Identifiability_10dv.py
