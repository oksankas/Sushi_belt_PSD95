# If in Jupyter or within ipython run both following lines
# If in the terminal run nrnivmodl to load Neuron engine
# %%bash
# nrnivmodl

import sys
from neuron import h
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.linalg
from PyNeuronToolbox.record import ez_record,ez_convert
from PyNeuronToolbox.morphology import dist_between,allsec_preorder

#np.random.seed(123456789)

import logging
FORMAT = '%(asctime)s :: %(levelname)s :: %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)
log = logging.getLogger("Edita_GA-logger")

import pandas as pd
from scipy.stats import qmc

import sushibelt
import time

log.info("function defined")

logFolder = '.'
runPrefix = 'sobol'

def sampleModel(model,chunk=0,chunkSize=256,seed=100):
    Ndim=len(model.lowb)
    cnames=['Cnt']+model.parnames+['Cost','Chi2']
    model.modelPrefix = f"{model.modelPrefix}_{runPrefix}_{chunk}_{chunkSize}"
    sampler = qmc.Sobol(d=Ndim, seed=seed)
    if chunk>1:
        skip = (chunk - 1) * chunkSize
        log.info(f"Skip first {skip} Sobol points")
        sampler = sampler.fast_forward(skip)
    dt = sampler.random(chunkSize)
    pset = model.lowb+(model.upbga-model.lowb)*dt
    for i in range(chunkSize):
        bestChi2 = model.costFunction(pset[i], retype='chi2')

