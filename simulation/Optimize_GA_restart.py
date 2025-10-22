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
from scipy.optimize import minimize

import pandas as pd

log.info("function defined")

logFolder = '.'
runPrefix = 'no_model'
def optimizeCF(model,init_file,nm_iter=100,ga_cycles=200, prob_mut=0.01, prob_cros=0.9,plot=True):
    """
    Function to optimize model parameters
    :param model: package that contains model, its costFunction, list of parameter names, their boundaries
    :param init_file: name of the file, which contains results of previous optimization rounds
    :param nm_iter: number of Nelder-Mead iterations
    :param ga_cycles: number of GA chunks of 10 iterations to run
    :param plot: logical should progress plot be made
    :return: three vectors -- bestX, bestY, and bestChi2
    """
    Ndim = len(model.lowb)
    gaPop = 2 * Ndim
    log.info('Load previous data')
    initDF = pd.read_csv(init_file)
    bX = (initDF.loc[np.argmin(initDF['Cost']), model.parnames]- model.lowb) / (model.upbga - model.lowb)
    initDFu = initDF.copy()
    initDFu.loc[:, model.parnames] = initDFu[model.parnames].round(5)
    initDFu = initDFu.drop_duplicates(
        subset=model.parnames)
    log.info(f'initDF: {initDF.shape}, initDFu: {initDFu.shape}')
    #log.info(f'initDFX: {bX}, initDFY: {initDF.loc[np.argmin(initDF["Cost"]),"Cost"]}')
    #log.info(initDFu)
    old_x = np.array(initDFu.loc[:, model.parnames])
    old_y = np.array(initDFu['Cost'])
    idx = np.argsort(old_y)
    #log.info(f'gaPop={gaPop},costs={np.log10(old_y[idx[:gaPop]])}')
    if len(idx) < gaPop:
        log.info(f'gaPop={gaPop}, len(idx)={len(idx)}')
        chrom = np.concatenate([(old_x - model.lowb) / (model.upbga - model.lowb), np.random.rand(gaPop - len(idx), Ndim)])
    else:
        chrom =  (old_x[idx[:gaPop]] - model.lowb) / (model.upbga - model.lowb)
    #log.info(f'chrom: {chrom.shape}')
    chrom[-1, :] = bX
    #log.info(f'chrom: {chrom.shape}')
    cnt = 0
    log.info('Prelim GA starts')
    pga = RCGA(func=lambda x: model.costFunction(x,retype='err'), n_dim=Ndim, size_pop=gaPop, max_iter=50000,
              prob_mut=0.09, prob_cros=prob_cros,lb=model.lowb, ub=model.upbga)
    pga.Chrom = chrom
    best_x, best_y = pga.run(10)
    bestX = pga.best_x
    bestY = best_y
    if plot:
        Y_history = pd.DataFrame(ga.all_history_Y)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
        Y_history.min(axis=1).cummin().plot(kind='line')
        plt.savefig(f'{logFolder}/{model.modelPrefix}_{runPrefix}_GA.png')
        plt.savefig(f'{logFolder}/{model.modelPrefix}_{runPrefix}_GA.pdf')
    chrom = pga.Chrom
    bestChi2 = model.costFunction(bestX,retype='chi2')
    bestY, cnames = prepareCnames(bestChi2, bestX, bestY,model)
    log.info(f'bestChi2={bestChi2}, bestY={bestY}, array={[bestY,bestChi2]}')
    best_line = np.append(cnt, bestX)
    best_line = np.append(best_line, bestY)
    best_line = np.append(best_line, bestChi2)
    log.info(f'best line={best_line} ({len(best_line)})')
    df = pd.DataFrame(best_line,index=cnames).T
    df.index = [model.cfCounter]
    df.to_csv(f'{logFolder}/{model.modelPrefix}_{runPrefix}_best_ga_restart.csv',header=True,mode='w')
    log.info('Run Nelder-Mead on Prelim GA result')
    result = minimize(lambda x: model.costFunction(x,retype='err'), bestX, method='nelder-mead',bounds=model.bnds,options={'maxiter':nm_iter})
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
    df.to_csv(f'{logFolder}/{model.modelPrefix}_{runPrefix}_best_ga_restart.csv',header=False,mode='a')

    log.info('Prepare GA population')
    log.info('GA starts')
    ga = RCGA(func=lambda x: model.costFunction(x,retype='err'), n_dim=Ndim, size_pop=gaPop, max_iter=50000,
              prob_mut=prob_mut, prob_cros=prob_cros,lb=model.lowb, ub=model.upbga)
    chrom[np.argmax(pga.Y), :] = (result.x - model.lowb) / (model.upbga - model.lowb)
    ga.Chrom = chrom

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
            bestX = ga.best_x
            bestY = best_y
            bestChi2 = model.costFunction(bestX, retype='chi2')
            bestY, cnames = prepareCnames(bestChi2, bestX, bestY,model)
            best_line = np.append(cnt, bestX)
            best_line = np.append(best_line, bestY)
            best_line = np.append(best_line, bestChi2)
            df = pd.DataFrame(best_line).T
            df.index = [model.cfCounter]
            df.to_csv(f'{logFolder}/{model.modelPrefix}_{runPrefix}_best_ga_restart.csv', header=False, mode='a')
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
            df.to_csv(f'{logFolder}/{model.modelPrefix}_{runPrefix}_best_ga_restart.csv', header=False, mode='a')
            chrom[np.argmax(ga.Y), :] = (result.x - model.lowb) / (model.upbga - model.lowb)
            ga.Chrom=chrom
        else :
            bestX = ga.best_x
            bestY, cnames = prepareCnames(bestChi2, bestX, bestY,model)
            best_line = np.append(cnt, bestX)
            best_line = np.append(best_line, bestY)
            best_line = np.append(best_line, bestChi2)
            df = pd.DataFrame(best_line).T
            df.index = [model.cfCounter]
            df.to_csv(f'{logFolder}/{model.modelPrefix}_{runPrefix}_best_ga_restart.csv', header=False, mode='a')
            log.info(f'best par distance = {bpar_dist} and Nelder-Mead run on the best GA result is omitted.')
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
