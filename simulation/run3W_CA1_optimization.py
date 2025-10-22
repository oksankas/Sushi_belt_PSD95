
import numpy as np
import math
import pandas as pd
import os
#np.random.seed(123456789)

import logging
FORMAT = '%(asctime)s :: %(levelname)s :: %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)
log = logging.getLogger("GA-logger")

import CA1_model as model
import Optimize_PSO_GA as rn

modelPrefix = 'CA1_3w'
runPrefix = 'optimization_pso10_nm10_ga10'
rn.runPrefix = runPrefix
logFolder = 'ident20240620'
# Check whether the specified path exists or not
isExist = os.path.exists(logFolder)
if not isExist:
    # Create a new directory because it does not exist
       os.makedirs(logFolder)
       log.info(f"Work folder {logFolder} is created.")

model.logFolder=logFolder
rn.logFolder=logFolder

expD=pd.read_csv('../data/CA1_gradient.csv')
cname0='D0W3'
cname7='D7W3'
d0w = -1 * np.ones(model.N)
for i in range(expD.shape[0]):
    abb = expD['Abbreviation'][i]
    sidx= model.segIdx[abb]
    d0w[sidx] *= -1*expD[f"{cname0}_MEAN"][i]/len(sidx)
for i in range(model.N):
    if d0w[i]<0:
        d0w[i] = model.bgSignal
dinit = d0w/np.sum(d0w)
target = np.array(expD[f"{cname7}_MEAN"])/np.sum(expD[f"{cname0}_MEAN"]) #results to fit to
targSD = np.array(expD[f"{cname7}_SD"])/np.sum(expD[f"{cname0}_MEAN"]) #measurement errors

model.dinit = dinit
model.d0w = d0w
model.target = target
model.modelPrefix = modelPrefix
model.targSD = targSD

bestX,bestY,bestChi2 = rn.optimizeCF(model,pso_iter=10, nm_iter=10, ga_cycles=10)