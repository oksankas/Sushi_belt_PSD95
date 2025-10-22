# If in Jupyter or within ipython run both following lines
# If in the terminal run nrnivmodl to load Neuron engine
# %%bash
# nrnivmodl

import numpy as np

import logging
FORMAT = '%(asctime)s :: %(levelname)s :: %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)
log = logging.getLogger("GA-logger")

from sko.GA import RCGA
from sko.PSO import PSO
from scipy.optimize import minimize, Bounds

import pandas as pd
import os

Nvals = 4 #10
parvals = [1.0 * i/Nvals for i in range(Nvals+1)]
logFolder = '.'
runPrefix = 'no_model_identifiability'
cfiCounter = 0

def prepPar(cpar,numPar,pval,pidx,bidx):
    lpar = np.zeros(numPar)
    lpar[pidx] = pval
    for j in range(numPar - 1):
        lpar[bidx[j]] = cpar[j]
    return lpar

#Number of GA rounds, which ends up in the same solution.
runLength = 0

def profileChiSq(model,pn,pv=0,pso_iter=15,nm_iter=100,ga_cycles=200):
    numPar = len(model.parnames)
    pidx=[i for i in range(numPar) if model.parnames[i]==pn][0]
    pvidx = [i for i in range(Nvals+1) if parvals[i] == pv][0]
    pval=model.lowb[pidx]+(model.upbga[pidx]-model.lowb[pidx])*pv
    bidx= [i for i in range(numPar) if i != pidx]
    cnames = ['Cnt','Nga','Nrun'] + model.parnames
    csv_name = f'{logFolder}/{model.modelPrefix}_{runPrefix}_res0.csv'
    dmp_name = f'{logFolder}/{model.modelPrefix}_{runPrefix}_dump.pkl'
    log.info(f'ChiSq starts: numPar={numPar}, pn={pn}, pidx={pidx}, pv={pv}, pval={pval}.')
    clowb=model.lowb[bidx]
    cupbga=model.upbga[bidx]
    cbnds=Bounds(lb=clowb,ub=cupbga)
    def costFunctionC(cpar,model=model):
        global cfiCounter
        cfiCounter += 1
        lpar = prepPar(cpar,numPar,pval,pidx,bidx)
        cost = model.costFunction(lpar)
        if cfiCounter > 1:
            pDF = pd.DataFrame(lpar).T
            pDF.index = [cfiCounter]
            pDF['ParamName'] = pn
            pDF['ParamVal'] = pval
            pDF['Cost'] = cost
            pDF.to_csv(f'{logFolder}/{model.modelPrefix}_{runPrefix}_{pn}_{pvidx}_cf_idnt.csv', header=False, mode='a')
        else:
            pDF = pd.DataFrame(lpar, index=model.parnames).T
            pDF.index = [cfiCounter]
            pDF['ParamName'] = pn
            pDF['ParamVal'] = pval
            pDF['Cost'] = cost
            pDF.to_csv(f'{logFolder}/{model.modelPrefix}_{runPrefix}_{pn}_{pvidx}_cf_idnt.csv', header=True, mode='w')
        return cost
    if not os.path.exists(dmp_name):
        log.info(f'{pn}={pval}, Start PSO')
        # Population size for the PSO should be at least 2*numPar and in ideal situation 5*numPar
        npop = max(5 * numPar, 15)
        pso = PSO(func=lambda x: costFunctionC(x,model), n_dim=(numPar - 1), pop=npop, max_iter=pso_iter, lb=clowb,
                ub=cupbga, w=0.8, c1=0.5, c2=0.5)
        pso.run()
        log.info(f'{pn}={pval}, best_x is {pso.gbest_x}, best_y is {pso.gbest_y}')
        log.info(f'{pn}={pval}, Run Nelder-Mead on PSO result')
        result = minimize(lambda x: costFunctionC(x,model), pso.gbest_x.flatten(), method='nelder-mead',bounds=cbnds,options={'maxiter':nm_iter})
        log.info(f'{pn}={pval}, Prepare GA population')
        #use pso.gbest_x and pso.gbest_y to get access to the global optimum found
        psox = pso.pbest_x
        psoy=pso.pbest_y
        idx=np.argsort(psoy.flatten())
        gainit=np.vstack([psox[idx[:(2*(numPar-1)-1)]],result.x])
        #log.info(f'{pn}={pval}, psox: {psox.shape}, idx: {len(idx)}, pop_size={2*(numPar-1)}, gainit: {gainit.shape}')
        log.info(f'{pn}={pval}, GA starts, population dimensions: {gainit.shape}')
        ga = RCGA(func=lambda x: costFunctionC(x,model),n_dim=(numPar-1), size_pop=2*(numPar-1), max_iter=50000, prob_mut=0.01, lb=clowb,
                ub=cupbga)
        ga.Chrom = (gainit-clowb)/(cupbga-clowb)
        bestX=result.x
        bestY=result.fun
    else:
        log.info(f'{pn}={pval}, GA starts from previously dumped population')
        chDF = pd.read_pickle(dmp_name)
        ga = RCGA(func=lambda x: costFunctionC(x,model),n_dim=(numPar-1), size_pop=2*(numPar-1), max_iter=50000, prob_mut=0.01, lb=clowb,
                ub=cupbga)
        ga.Chrom = np.array(chDF)
        best_x, best_y = ga.run(2)
        bestX = ga.best_x
        bestY = best_y

    for cnt in range(ga_cycles):
        log.info(f'{pn}={pval}, Continue GA {cnt}')
        best_x, best_y = ga.run(10)
        best_x_p=prepPar(best_x,numPar,pval,pidx,bidx)
        log.info(f'GA {pn}={pval} done {cnt}: cfCounter={model.cfCounter}, best CF={best_y}')
        log.info(f'{pn}={pval}, best par={best_x} ({len(best_x)})')
        #best_line = np.append(cnt,best_x_p)
        #log.info(f'{pn}={pval}, best line={best_line} ({len(best_line)})')
        chrom = ga.Chrom
        bpar_dist = np.sqrt(np.sum((bestX-ga.best_x) ** 2))/(numPar-1)
        if bpar_dist > 1e-7 :
            runLength = 0
            log.info(f'{pn}={pval}, Run Nelder-Mead on the best GA result {cnt}, best par distance = {bpar_dist}')
            result = minimize(lambda x: costFunctionC(x,model), ga.best_x, method='nelder-mead', bounds=cbnds, options={'maxiter': nm_iter})
            chrom[np.argmax(ga.Y), :] = (result.x - clowb) / (cupbga - clowb)
            ga.Chrom=chrom
            bestX = result.x
            bestY = result.fun
        else :
            runLength += 1
            bestX = ga.best_x
            bestY = best_y
            log.info(f'{pn}={pval}, best par distance = {bpar_dist} run= {runLength} and Nelder-Mead run on the best GA result is omitted.')
        bestChi2 = model.costFunction(best_x_p, retype='chi2')
        bestLine = np.append([cfiCounter,cnt,runLength], best_x_p)
        pDF = pd.DataFrame(bestLine,index=cnames).T
        pDF.index = [model.cfCounter]
        pDF['ParamName'] = pn
        pDF['ParamVal'] = pval
        pDF['Cost'] = bestY
        pDF['Chi2'] = bestChi2
        pDF.to_csv(csv_name, header=True, mode='w')
        chPD = pd.DataFrame(chrom)
        chPD.to_pickle(dmp_name)
        log.info(f'GA {pn}={pval} completed {cnt}: cfCounter={model.cfCounter}, best CF={bestY}')
    return bestX,bestY

