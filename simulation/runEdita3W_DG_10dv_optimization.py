
import numpy as np
import math
import pandas as pd
import os
#np.random.seed(123456789)

import logging
FORMAT = '%(asctime)s :: %(levelname)s :: %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)
log = logging.getLogger("Edita_GA-logger")

import Edita_DG_10reg_10dv_model as model
import Optimize_PSO_GA as rn
#import Optimize_PSO_GA as rn

runPrefix = 'optimization_pso50_nm100_ga200'
rn.runPrefix = runPrefix
logFolder = 'DG_3W_10dv_PSO_GA__20240919'
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

prev_fname = '/Users/anatolii-sorokin/Documents/Projects/neuron-model/DG_10dv_PSO_GA__20240820/Edita_DG_10reg_10dv_3M_optimization_pso50_nm100_ga200_best_pso_ga.csv'

model.modelPrefix = 'Edita_DG_10reg_10dv_3W'
# model.parnames=['dv_rd1','dv_rd2','dv_rd3','dv_rd4','dv_rd5',
#           'dv_rd6','dv_rd7','dv_rd8','dv_rd9','dv_rd10']
# model.lowb=np.array([-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,])
# model.upbga=np.array([1,1,1,1,1,1,1,1,1,1])
# model.bnds=Bounds(lb=model.lowb,ub=model.upbga)
# model.Ndim=len(model.lowb)


bestX,bestY,bestChi2 = rn.optimizeCF(model,pso_iter=50, nm_iter=100, ga_cycles=200,prob_mut=0.01)
#quick test setup
#bestX,bestY,bestChi2 = rn.optimizeCF(model,pso_iter=1, nm_iter=5, ga_cycles=2,prob_mut=0.01)