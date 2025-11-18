Here we adapted the sushi-belt model (Williams et al., 2016) to model the distribution of endogenous post-synaptic density protein 95 (PSD95) across the dendritic tree of excitatory neuron types in intact mouse brain. PSD95 is a highly abundant scaffolding protein in excitatory synapses that interacts with more than 100 postsynaptic proteins, organising them into complexes and super-complexes which facilitate the processing of incoming signals into the cell. PSD95 is essential for learning and memory and its mutations are associated with neurodevelopmental and behavioural disorders, e.g. Schizophrenia, ASD and bipolar disease. Our modelling of PSD95 distribution reveals which cellular processes can feasibly explain the complex protein distribution patterns observed in excitatory neuron types. Additionally, our model implicates specific cellular processes that alter during development and ageing in the brain to produce changed patterns of molecular distribution of synaptic proteins

This code is designed to simulate and identify extended sushi-belt model describing trafficking of proteins and RNA within neurons.

The model is based on the code from [that repository](https://github.com/ahwillia/Williams-etal-Synaptic-Transport) and [paper](http://dx.doi.org/10.7554/eLife.20556).

Adapted Sushi-Belt model for Synapse Proteomics © 2025 by Oksana Sorokina is licensed under CC BY 4.0 

The multiple versions of the same function may stem from legacy code, which was used to evaluate preliminary model structure. Once the structure was decided the code was forked into model definition, simulation infrastructure and simulation experiment script hence multiple versions. 

This is the research code only -NOT intended to be a finished product but rather an adjunct to simulations we describe in the paper: https://www.biorxiv.org/content/10.1101/2025.07.31.667853v1.abstract

Repository is structured as separate Data, Simulations, Py_notebooks, and Eddie folders, where Data folder contains the respective experimental data, Py_notebooks -contains notebooks for specific tasks, Eddie - GridEngine scripts for execution of the code on HPC platform Eddie.
The  simulation folder of the repository is organized to support the simulation pipeline; it contains: basic model e.g. DG_density_10reg_1dv_model_dvonly.py; simulation framework, e.g. Optimize_PSO_GA_restart.py and simulation experiment code, which combine model, simulation parameters etc., e.g. run3M_DG_10dv_optimization.py. 
_


# Installation

Install the local version of conda or Anaconda, then create and configure environment as follow:
```
conda update -n base -c defaults conda
conda create -n sushibelt3.9 python=3.9
conda activate sushibelt3.9
conda install -c conda-forge numpy matplotlib notebook ipython scipy 
pip3 install neuron==8.2.7
pip3 install pandas scikit-opt
```

For analysis we will need additional package, so clone source code from the GitHub repository https://github.com/lptolik/PyNeuron-Toolbox and install it:
```
cd /path/to/PyNeuron-Toolbox/reposidory
python3 setup.py install
```

To check that the code is working run following commands:

```
cd /path/to/Sushi_belt_PSD95
cd simulations
nrnivmodl

python3 runSushi.py
```

The following code will run a fully fledged optimisation run, which could take a while (~24h on Eddie cluster):
```
cd /path/to/Sushi_belt_PSD95
cd simulation
nrnivmodl

chunkSize=512
seed=255

odir=test

mkdir -p odir
export odir seed
echo "$odir" "$seed"

# Run the program
python3 run3M_DG_optimization.py
```