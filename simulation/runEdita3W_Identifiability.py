
import numpy as np
import math
import pandas as pd
import os
#np.random.seed(123456789)

import logging
FORMAT = '%(asctime)s :: %(levelname)s :: %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)
log = logging.getLogger("Edita_GA-logger")

import Edita_model as model
import Identifiability_PSO_GA_restart as rn

modelPrefix = model.modelPrefix
runPrefix = 'identifiability_pso10_nm10_ga10'
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

expD=pd.read_csv('data/CA1_gradient.csv')
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
model.targSD = targSD
model.modelPrefix = modelPrefix

chiCounter = 0
i = 0
jval = [1 + k for k in range(rn.Nvals)]

numPar = len(model.parnames)
for pni in model.parnames :
    for pvj in rn.parvals :
        rn.cfiCounter = 0
        bestX, bestY = rn.profileChiSq(model, pni, pvj, pso_iter=10, nm_iter=10, ga_cycles=10)
        pidx=[k for k in range(numPar) if model.parnames[k] == pni][0]
        pval= model.lowb[pidx] + (model.upbga[pidx] - model.lowb[pidx]) * pvj
        bidx= [k for k in range(numPar) if k != pidx]
        bestX_p = rn.prepPar(bestX, numPar, pval, pidx, bidx)
        bestChi2 = model.costFunction(bestX_p,retype='chi2')
        log.info(f'GA {pni}={pval} best found: cfCounter={model.cfCounter}, best CF={bestY}')
        log.info(f'{pni}={pval}, best found: par={bestX_p} ({len(bestX_p)}')
        bestLine = np.append(rn.cfiCounter, bestX_p)
        log.info(f'{pni}={pval}, best found: line={bestLine} ({len(bestLine)})')
        if chiCounter >0:
            bdf = pd.DataFrame(bestLine).T
            bdf['ParamName'] = pni
            bdf['ParamVal'] = pval
            bdf['Cost'] = bestY
            bdf['Chi2'] = bestChi2
            bdf.index = [chiCounter]
            log.info(f'{pni}={pval}, best found: df={bdf}')
            bdf.to_csv(f'{logFolder}/{modelPrefix}_{runPrefix}_res.csv', header=False, mode='a')
        else:
            cnames = ['Cnt'] + model.parnames
            bdf = pd.DataFrame(bestLine,index=cnames).T
            bdf['ParamName'] = pni
            bdf['ParamVal'] = pval
            bdf['Cost'] = bestY
            bdf['Chi2'] = bestChi2
            bdf.index = [chiCounter]
            log.info(f'{pni}={pval}, best found: df={bdf}')
            bdf.to_csv(f'{logFolder}/{modelPrefix}_{runPrefix}_res.csv', header=True, mode='w')
        chiCounter += 1
