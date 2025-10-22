
import numpy as np
import pandas as pd

import logging
FORMAT = '%(asctime)s :: %(levelname)s :: %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)
log = logging.getLogger("GA-logger")

import os
parN = int(os.environ['SGE_TASK_ID'])
valN = int(os.environ['valnum'])
log.info(f"parN={parN}, valN={valN}")

logdir = os.environ['odir']

import CA1_20reg_1dv_model as model
import Identifiability_PSO_GA_restart as rn

rn.Nvals = valN
rn.parvals = [1.0 * i/rn.Nvals for i in range(rn.Nvals+1)]

c = 0
for i in model.parnames :
    for j in rn.parvals :
        c += 1
        if c == parN:
            pni = i
            pvj = j
            break
    else:
        continue
    break
#pni = model.parnames[parN-1]
modelPrefix = f'{model.modelPrefix}_{pni}_{pvj}_{valN}'
model.modelPrefix = modelPrefix
runPrefix = 'identifiability_pso10_nm10_ga10'
rn.runPrefix = runPrefix
logFolder = logdir #'ident20240620'
# Check whether the specified path exists or not
isExist = os.path.exists(logFolder)
if not isExist:
    # Create a new directory because it does not exist
       os.makedirs(logFolder)
       log.info(f"Work folder {logFolder} is created.")

model.logFolder=logFolder
rn.logFolder=logFolder


log.info(f"Identifiability task for parameter '{pni}' with {valN} intervals saved to {logFolder}.")
chiCounter = 0
i = 0
jval = [1 + k for k in range(rn.Nvals)]

numPar = len(model.parnames)
rn.cfiCounter = 0
bestX, bestY = rn.profileChiSq(model, pni, pvj, pso_iter=20, nm_iter=50, ga_cycles=50)
pidx=[k for k in range(numPar) if model.parnames[k] == pni][0]
pval= model.lowb[pidx] + (model.upbga[pidx] - model.lowb[pidx]) * pvj
bidx= [k for k in range(numPar) if k != pidx]
bestX_p = rn.prepPar(bestX, numPar, pval, pidx, bidx)
bestChi2 = model.costFunction(bestX_p,retype='chi2')
log.info(f'GA {pni}={pval} best found: cfCounter={model.cfCounter}, best CF={bestY}')
log.info(f'{pni}={pval}, best found: par={bestX_p} ({len(bestX_p)}')
bestLine = np.append(rn.cfiCounter, bestX_p)
log.info(f'{pni}={pval}, best found: line={bestLine} ({len(bestLine)})')
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
