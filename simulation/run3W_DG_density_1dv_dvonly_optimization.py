
import numpy as np
import math
import pandas as pd
import os
#np.random.seed(123456789)

import logging
FORMAT = '%(asctime)s :: %(levelname)s :: %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)
log = logging.getLogger("GA-logger")

import DG_density_10reg_1dv_model_dvonly as model
import Optimize_PSO_GA_restart as rn

runPrefix = 'optimization_pso10_nm10_ga10'
rn.runPrefix = runPrefix
logFolder = 'DG_1dv_PSO_GA_density_3W_dvonly_20241031'
#logFolder = os.environ['odir']

# Check whether the specified path exists or not
isExist = os.path.exists(logFolder)
if not isExist:
    # Create a new directory because it does not exist
       os.makedirs(logFolder)
       log.info(f"Work folder {logFolder} is created.")

model.logFolder=logFolder
#model.lowb[0]=0.8
rn.logFolder=logFolder

# Read new data
cname0='D0W3'
cname7='D7W3'
N=model.N
d0w = -1 * np.ones(N)
for i in range(model.expD.shape[0]):
    abb = model.expD['Abbreviation'][i]
    sidx= model.segIdx[abb]
    d0w[sidx] *= -1*model.expD[f"{cname0}_MEAN"][i]/len(sidx)
for i in range(N):
    if d0w[i]<0:
        d0w[i] = model.bgSignal
model.dinit = d0w/np.sum(d0w)

model.target = np.array(model.expD[f"{cname7}_MEAN"])/np.sum(model.expD[f"{cname0}_MEAN"]) #norm target to Day0 sum to take into accound degradation
model.targSD = np.array(model.expD[f"{cname7}_SD"])/np.sum(model.expD[f"{cname0}_MEAN"]) #results to fit to
model.tnorm = np.sum(model.target ** 2)

model.modelPrefix = 'DG_density_10reg_1dv_3W_dvonly'

bestX,bestY,bestChi2 = rn.optimizeCF(model,pso_iter=150, nm_iter=100, ga_cycles=200,prob_mut=0.01)
#quick test setup
#bestX,bestY,bestChi2 = rn.optimizeCF(model,pso_iter=1, nm_iter=5, ga_cycles=2,prob_mut=0.01)