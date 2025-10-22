# If in Jupyter or within ipython run both following lines
# If in the terminal run nrnivmodl to load Neuron engine
# %%bash
# nrnivmodl

import numpy as np
import matplotlib.pyplot as plt
import math

import logging
FORMAT = '%(asctime)s :: %(levelname)s :: %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)
log = logging.getLogger("GA-logger")

from sko.GA import RCGA
from sko.PSO import PSO
from scipy.optimize import minimize

import pandas as pd
import os

log.info("function defined")

logFolder = '.'
runPrefix = 'no_model'
def optimizeCF(model,pso_iter=15,nm_iter=100,ga_cycles=200, prob_mut=0.01, prob_cros=0.9):
    """
    Function to optimize model parameters
    :param model: package that contains model, its costFunction, list of parameter names, their boundaries
    :param pso_iter: number of iterations of PSO to run
    :param nm_iter: number of Nelder-Mead iterations
    :param ga_cycles: number of GA chunks of 10 iterations to run
    :return: three vectors -- bestX, bestY, and bestChi2
    """
    Ndim = len(model.lowb)
    csv_name = f'{logFolder}/{model.modelPrefix}_{runPrefix}_best_pso_ga.csv'
    dmp_name = f'{logFolder}/{model.modelPrefix}_{runPrefix}_dump.pkl'
    log.info('Start PSO')
    runLength = 0
    if not os.path.exists(dmp_name):
        npop = max(5 * Ndim, 15)
        pso = PSO(func=lambda x: model.costFunction(x,retype='err'), n_dim=Ndim, pop=npop, max_iter=pso_iter, lb=model.lowb,
                ub=model.upbga, w=0.8, c1=0.5, c2=0.5)
        pso.run()
        log.info(f'best_x is {pso.gbest_x}, best_y is {pso.gbest_y}')
        plt.plot(pso.gbest_y_hist)
        plt.savefig(f'{logFolder}/{model.modelPrefix}_{runPrefix}_PSO.png')
        plt.savefig(f'{logFolder}/{model.modelPrefix}_{runPrefix}_PSO.pdf')
        cnt = 0
        bestX = pso.gbest_x
        bestY = pso.gbest_y
        bestChi2 = model.costFunction(bestX,retype='chi2')
        bestY, cnames = prepareCnames(bestChi2, bestX, bestY,model)
        log.info(f'bestChi2={bestChi2}, bestY={bestY}, array={[bestY,bestChi2]}')
        best_line = np.append(cnt, bestX)
        best_line = np.append(best_line, bestY)
        best_line = np.append(best_line, bestChi2)
        log.info(f'best line={best_line} ({len(best_line)})')
        df = pd.DataFrame(best_line,index=cnames).T
        df.index = [model.cfCounter]
        df.to_csv(csv_name,header=True,mode='w')
        log.info('Run Nelder-Mead on PSO result')
        result = minimize(lambda x: model.costFunction(x,retype='err'), pso.gbest_x, method='nelder-mead',bounds=model.bnds,options={'maxiter':nm_iter})
        bestX = result.x
        bestY = result.fun
        bestChi2 = model.costFunction(bestX,retype='chi2')
        bestY, cnames = prepareCnames(bestChi2, bestX, bestY,model)
        best_line = np.append(cnt, bestX)
        best_line = np.append(best_line, bestY)
        best_line = np.append(best_line, bestChi2)
        log.info(f'best line={best_line} ({len(best_line)})')
        df = pd.DataFrame(best_line).T
        df.index = [model.cfCounter]
        df.to_csv(csv_name,header=False,mode='a')

        log.info('Prepare GA population')
        psox=pso.pbest_x
        psoy=pso.pbest_y
        idx=np.argsort(psoy.flatten())
        gainit=np.vstack([psox[idx[:(2*Ndim-1)]],result.x])
        log.info('GA starts')
        ga = RCGA(func=lambda x: model.costFunction(x,retype='err'), n_dim=Ndim, size_pop=2*Ndim, max_iter=50000,
                  prob_mut=prob_mut, prob_cros=prob_cros,lb=model.lowb, ub=model.upbga)
        ga.Chrom = (gainit-model.lowb)/(model.upbga-model.lowb)
    else:
        log.info(f'GA starts from previously dumped population')
        chDF = pd.read_pickle(dmp_name)
        ga = RCGA(func=lambda x: model.costFunction(x,retype='err'), n_dim=Ndim, size_pop=2*Ndim, max_iter=50000,
                  prob_mut=prob_mut, prob_cros=prob_cros,lb=model.lowb, ub=model.upbga)
        ga.Chrom = np.array(chDF)
        best_x, best_y = ga.run(2)
        bestX = ga.best_x
        bestY = best_y

    for cnt in range(ga_cycles):
        log.info(f'Continue GA {cnt}')
        best_x, best_y = ga.run(10)
        log.info(f'GA done {cnt}: cfCounter={model.cfCounter}, best CF={best_y}')
        log.info(f'best par={best_x} ({len(best_x)}')
        Y_history = pd.DataFrame(ga.all_history_Y)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
        Y_history.min(axis=1).cummin().plot(kind='line')
        plt.savefig(f'{logFolder}/{model.modelPrefix}_{runPrefix}_GA.png')
        plt.savefig(f'{logFolder}/{model.modelPrefix}_{runPrefix}_GA.pdf')
        chrom = ga.Chrom
        bpar_dist = np.sqrt(np.sum((bestX - ga.best_x) ** 2))/Ndim
        if bpar_dist > 1e-7 or math.fabs(bestY - best_y) > 1e-7:
            log.info('Run Nelder-Mead on the best GA result {cnt}')
            runLength = 0
            bestX = ga.best_x
            bestY = best_y
            bestChi2 = model.costFunction(bestX, retype='chi2')
            bestY, cnames = prepareCnames(bestChi2, bestX, bestY,model)
            best_line = np.append(cnt, bestX)
            best_line = np.append(best_line, bestY)
            best_line = np.append(best_line, bestChi2)
            df = pd.DataFrame(best_line).T
            df.index = [model.cfCounter]
            df.to_csv(csv_name, header=False, mode='a')
            result = minimize(lambda x: model.costFunction(x,retype='err'), ga.best_x, method='nelder-mead', bounds=model.bnds, options={'maxiter': nm_iter})
            bestX = result.x
            bestY = result.fun
            bestChi2 = model.costFunction(bestX, retype='chi2')
            bestY, cnames = prepareCnames(bestChi2, bestX, bestY,model)
            best_line = np.append(cnt, bestX)
            best_line = np.append(best_line, bestY)
            best_line = np.append(best_line, bestChi2)
            log.info(f'best line={best_line} ({len(best_line)})')
            df = pd.DataFrame(best_line).T
            df.index = [model.cfCounter]
            df.to_csv(csv_name, header=False, mode='a')
            chrom[np.argmax(ga.Y), :] = (result.x - model.lowb) / (model.upbga - model.lowb)
            ga.Chrom=chrom
        else :
            runLength += 1
            bestX = ga.best_x
            bestY, cnames = prepareCnames(bestChi2, bestX, bestY,model)
            best_line = np.append(cnt, bestX)
            best_line = np.append(best_line, bestY)
            best_line = np.append(best_line, bestChi2)
            df = pd.DataFrame(best_line).T
            df.index = [model.cfCounter]
            df.to_csv(csv_name, header=False, mode='a')
            log.info(f'best par distance = {bpar_dist} and Nelder-Mead run on the best GA result is omitted, runLength = {runLength}.')
        chPD = pd.DataFrame(chrom)
        chPD.to_pickle(dmp_name)
    log.info(f'GA completed {cnt}: cfCounter={model.cfCounter}, best CF={bestY}')
    return bestX,bestY,bestChi2


def prepareCnames(bestChi2, bestX, bestY,model):
    if np.isscalar(bestChi2):
        cnames = ['Cnt'] + model.parnames + ['Cost', 'Chi2']
    else:
        #bestY = model.costFunction(bestX, retype='err')
        costnames = ['Cost'] #+ [f"Cost_{i}" for i in range(len(bestY)) if i > 0]
        chinames = [f"Chi2_{i}" for i in range(len(bestChi2))]
        cnames = ['Cnt'] + model.parnames + costnames + chinames
    return bestY, cnames
