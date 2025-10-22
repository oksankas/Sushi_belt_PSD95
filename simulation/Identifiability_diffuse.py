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
from scipy.stats import chi2

import pandas as pd
import os

Nvals = 4 #10
parvals = [1.0 * i/Nvals for i in range(Nvals+1)]
logFolder = '.'
runPrefix = 'no_model_identifiability'
cfiCounter = 0
alpha = 0.05

def prepPar(cpar,numPar,pval,pidx,bidx):
    lpar = np.zeros(numPar)
    lpar[pidx] = pval
    for j in range(numPar - 1):
        lpar[bidx[j]] = cpar[j]
    return lpar

#Number of GA rounds, which ends up in the same solution.
#runLength = 0

def profileChiSq(model,opt_par,pn,step=0.001,nm_iter=100):
    """
    Perform stepwise profiling of the model likelihood
    :param model: model to simulate
    :param opt_par: vector of optimal parameters
    :param pn: name of the parameter to profile
    :param step: step along profiling parameter
    :param nm_iter: number of iterations in optimization
    :return: DataFrame of the likelihood profile
    """
    numPar = len(model.parnames)
    pidx=[i for i in range(numPar) if model.parnames[i]==pn][0]
    bidx= [i for i in range(numPar) if i != pidx]
    cnames = ['Cnt','Nsteps'] + model.parnames
    csv_name = f'{logFolder}/{model.modelPrefix}_{runPrefix}_res0.csv'
    dmp_name = f'{logFolder}/{model.modelPrefix}_{runPrefix}_dump.pkl'
    opt_chi = model.costFunction(opt_par,retype='chi2')
    delta = chi2.ppf(1 - alpha, len(model.parnames))
    chi_fin = opt_chi + delta
    clowb=model.lowb[bidx]
    cupbga=model.upbga[bidx]
    cbnds=Bounds(lb=clowb,ub=cupbga)
    cnt = 0
    def costFunctionC(cpar,model=model):
        global cfiCounter
        cfiCounter += 1
        lpar = prepPar(cpar,numPar,pval,pidx,bidx)
        cost = model.costFunction(lpar,retype='chi2')
        if cfiCounter > 1:
            pDF = pd.DataFrame(lpar).T
            pDF.index = [cfiCounter]
            pDF['ParamName'] = pn
            pDF['ParamVal'] = pval
            pDF['Chi2'] = cost
            pDF.to_csv(f'{logFolder}/{model.modelPrefix}_{runPrefix}_{pn}_cf_idnt.csv', header=False, mode='a')
        else:
            pDF = pd.DataFrame(lpar, index=model.parnames).T
            pDF.index = [cfiCounter]
            pDF['ParamName'] = pn
            pDF['ParamVal'] = pval
            pDF['Chi2'] = cost
            pDF.to_csv(f'{logFolder}/{model.modelPrefix}_{runPrefix}_{pn}_cf_idnt.csv', header=True, mode='w')
        return cost
    pval=opt_par[pidx]
    bestX = np.array(opt_par)[bidx]
    bestY = opt_chi
    def optimizeParamVal(bestX, bidx, model, nm_iter, numPar, pidx, pn, pval):
        result = minimize(lambda x: costFunctionC(x, model), bestX, method='nelder-mead', bounds=cbnds,
                          options={'maxiter': nm_iter})
        bestX = result.x
        best_x_p = prepPar(bestX, numPar, pval, pidx, bidx)
        bestY = result.fun
        bestCost = model.costFunction(best_x_p)
        bestLine = np.append([cfiCounter, cnt], best_x_p)
        pDF = pd.DataFrame(bestLine, index=cnames).T
        pDF.index = [model.cfCounter]
        pDF['ParamName'] = pn
        pDF['ParamVal'] = pval
        pDF['Cost'] = bestCost
        pDF['Chi2'] = bestY
        if cnt == 0:
            pDF.to_csv(csv_name, header=True, mode='w')
        else:
            pDF.to_csv(csv_name, header=False, mode='a')
        log.info(f'GA {pn}={pval} completed {cnt}: cfCounter={model.cfCounter}, best CF={bestY}')
        return bestX, bestY
    resList = []
    while bestY < chi_fin:
        log.info(f'ChiSq starts: numPar={numPar}, pn={pn}, pidx={pidx}, pval={pval}.')
        bestX, bestY = optimizeParamVal(bestX, bidx, model, nm_iter, numPar, pidx, pn, pval)
        resList.append([bestX, bestY])
        pval += step
        cnt += 1

    pval=opt_par[pidx]
    bestX = np.array(opt_par)[bidx]
    bestY = opt_chi
    while bestY < chi_fin:
        pval -= step
        log.info(f'ChiSq starts: numPar={numPar}, pn={pn}, pidx={pidx}, pval={pval}.')
        bestX, bestY = optimizeParamVal(bestX, bidx, model, nm_iter, numPar, pidx, pn, pval)
        resList.append([bestX, bestY])
        cnt += 1

    return resList


