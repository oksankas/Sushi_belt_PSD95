# If in Jupyter or within ipython run both following lines
# If in the terminal run nrnivmodl to load Neuron engine
# %%bash
# nrnivmodl

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

# Load morphology and other stuff
# --> SegLists: soma[2], dend[74], dend_5[37], apic[42], axon[1]
# --> Files from Migliore & Migliore (2012)
# --> CA1 pyramidal neuron
h.load_file('stdrun.hoc')
h.xopen('ri06.hoc')
h.xopen('fixnseg.hoc')
h.xopen('5a_nogui.hoc')
h.tstop = 700.0

from sko.GA import RCGA
from sko.PSO import PSO
from scipy.optimize import minimize, Bounds

import pandas as pd

import sushibelt
import time

def trafficking_solution(utarg):
    """ Solve the problem by tuning trafficking rates, like Figs 1 and 2. """
    x = []

    # Iterative traversal of dendritic tree in pre-order
    i = 0
    section = None
    parentStack = [(None, h.soma[0])]
    while len(parentStack) > 0:
        # Get next section to traverse
        #  --> p is parent index, section is h.Section object
        (p, section) = parentStack.pop()

        # Trafficking to/from parent
        if p is not None:
            mp = utarg[p]  # concentration in parent
            mc = utarg[i]  # concentration in child
            x.insert(0, mp / mc)

        # visit all segments in compartment
        for (j, seg) in enumerate(section):
            # Deal with out/into rates within compartment, just tridiag matrix
            if j > 0:
                mp = utarg[i - 1]
                mc = utarg[i]
                x.insert(0, mp / mc)

            # move onto next compartment
            i += 1

        # now visit children in pre-order
        child_list = list(h.SectionRef(sec=section).child)
        if child_list is not None:
            child_list.reverse()
        for c_sec in child_list:
            parentStack.append([i - 1, c_sec])  # append parent index and child

    # return calculated guesses (flip, up/down since get_deriv pops from start)
    return np.array(x)

#
# def get_sys_matrix(utarg, F=0.5, Ctau=1e-3, dscale=0.1, dv=1e-7):
#     # F is a mixing factor between 0 and 1
#     K = np.sum(utarg) / N
#     x = trafficking_solution(F * utarg + (1 - F) * K)
#     a = (1 / (1 + x))
#     a = list(a)
#     b = list((1 / (1 + x ** -1)))
#     l = list(np.ones(N) * dv)
#     c = list(Ctau * utarg / (F * utarg + (1 - F) * K))
#     d = list([ci * dscale for ci in c])
#     A = sushi_system(a, b, c, d, l)
#     return A
#
bgSignal = 1e-5
prDemand = [0.01385481065982917,-6.226265838840085,0.6709771115511489,-17.995913305482574,
            -16.696101588995024,0.21868548101204954,0.06211405206769383,0.048882353452234434,
            0.046632773486508856,1.9401449712242143e-06,0.7864443640945032,0.5632548318074286,
            0.14915702983161863,0.1491589398028194,0.13322800132469398,0.15668388533325328,
            0.17435550169553826,0.20436446923690443,0.37541096611836217,0.5260391371129066,
            0.043995168941356404,0.1970021740399707,0.19622524557390722,0.2758337024352678,0.9999999999613025]

def calcProtDemand(par,delta=bgSignal):
    F = par[0]
    Ctau = 10 ** par[1]
    mProp = par[2]
    dvA = par[3]
    dvB = par[4]
    dv = np.zeros(N)
    utarg = delta*np.ones(N)
    for k in range(N):
        if itarg[k] > 4:
            utarg[k] = par[itarg[k]]
            dv[k] = (10 ** dvA) + (10 ** dvB)*utarg[k]
    utarg /= np.sum(utarg)
    K = np.sum(utarg) / N
    x = trafficking_solution(F * utarg + (1 - F) * K)
    a = (1 / (1 + x))
    a = list(a)
    b = list((1 / (1 + x ** -1)))
    l = list(dv)
    c = list(Ctau * utarg / (F * utarg + (1 - F) * K))
    d = list(np.zeros(N))
    return a,b,c,d,l

def calcUtrace(par,delta=bgSignal,tstart=1e10):
    # Soma RNA traffick rates
    ar=ar_b.copy()
    br = br_b.copy()
    ar[0] = par[0] # Soma $a_r^s$
    ar[1] = par[0]
    br[0] = par[1] # Soma $b_r^s$
    br[1] = par[1]
    # Soma protein traffick rates
    ap=ap_b.copy()
    bp = bp_b.copy()
    ap[0] = par[2] # Soma $a_p^s$
    ap[1] = par[2]
    bp[0] = par[3] # Soma $b_p^s$
    bp[1] = par[3]
    # Soma protein degradation rate
    lp = lp_b.copy()
    lp[0] = 10 ** par[4] # Soma $l_p^s$
    lp[1] = 10 ** par[4] # Soma $l_p^s$
    # Transcription rate
    tr = 10 ** par[6]
    # Soma RNA reattachment
    dr = dr_b.copy()
    dr[0] = 10 ** par[8] # Soma $d_r^s$
    dr[1] = 10 ** par[8]
    # Soma protein reattachment
    dp = dp_b.copy()
    dp[0] = 10 ** par[9] # Soma $d_p^s$
    dp[1] = 10 ** par[9]
    # Soma protein detachment
    cp = cp_b.copy()
    cp[0] = 10 ** par[11] # Soma $c_p^s$
    cp[1] = 10 ** par[11]
    # RNA detachment
    cr = np.array(cr_b) * (10 ** par[12]) # Neurophil $f_{C_{\tau}} = \frac{C_{\tau R}}{C_{\tau p}}$
    cr[0] = 10 ** par[10] # Soma $c_r^s$
    cr[1] = 10 ** par[10]
    cr = list(cr)
    # RNA degradation
    lr = np.ones(N) * (10 ** par[13]) # Neurophil $l_r^n$
    lr[0] = 10 ** par[5] # Soma $l_r^s$
    lr[1] = 10 ** par[5]
    lr = list(lr)
    # RNA translation
    tp = np.ones(N) * (10 ** par[14]) # Neurophil $t_p^n$
    tp[0] = 10 ** par[7] # Soma $t_p^s$
    tp[1] = 10 ** par[7]
    tp = list(tp)


    A = sushibelt.full_sushi_system(h, ar, br, cr, dr, tr, lr, ap, bp, cp, dp, tp, lp)
    utrace,times = sushibelt.run_sim(h,A, npools=4,tmax=1.5,t0=tstart)
    return utrace.T


log.info("function defined")

##### Read data ######
#seglist in pre-order
sec_list = allsec_preorder(h)
seg_list = []
for sec in sec_list:
    locs = np.linspace(0,1,sec.nseg+2)[1:-1]
    for loc in locs:
        seg_list.append(sec(loc))
N = len(seg_list)
tdf=pd.read_csv('data/seg_mapping.csv')
abbCA1=tdf['abb']
abbT={}
segIdx={}
for i in range(N):
    abbT[abbCA1[i]] = 1+ abbT.get(abbCA1[i],0)
    ll=segIdx.get(abbCA1[i],[])
    ll.append(i)
    segIdx[abbCA1[i]] = ll

expD=pd.read_csv('data/CA1_gradient.csv')
intD=pd.read_csv('data/CA1_Day0_intensity.csv')
subreg = ['CA1so', 'CA1sr', 'CA1slm']

target = np.array(intD["mean_intensity"])/np.sum(intD["mean_intensity"]) #norm target to Day0 sum to take into accound degradation
tnorm = np.sum(target ** 2)
day7 = 7 * 24 * 3600 # final time point

itarg = np.ones(N, dtype=int)
for i in range(expD.shape[0]):
    abb = expD['Abbreviation'][i]
    sidx = segIdx[abb]
    itarg[sidx] *= (i+5)

log.info("data read")

ar_b,br_b,cr_b,dr_b,_ = calcProtDemand(prDemand)
ap_b,bp_b,cp_b,dp_b,lp_b = calcProtDemand(prDemand)

cfCounter = 0
dumpCSV = True
def costFunction(par):
    initTime = time.time()
    global cfCounter, dumpCSV
    cfCounter += 1
    log.info(f'Cost function starts: {cfCounter}')
    log.info(f'{par}')
    utrace = calcUtrace(par)
    resM, resF = sushibelt.aggregate_segments(utrace[2*N:, -1], segIdx, expD['Abbreviation'], fun=np.sum)
    #cost=np.sum((resF + resM - target) ** 2)/tnorm
    cost=np.sum((resF - target) ** 2)/tnorm
    FinalTime = time.time() - initTime
    if dumpCSV:
        best_line = np.append(cfCounter, par)
        best_line = np.append(best_line, cost)
        df = pd.DataFrame(best_line).T
        #log.info(f'best df={df}')
        df.to_csv('Edita_RNA_cf.csv',header=False,mode='a')
    log.info(f'Cost function done: cost={cost}. ({FinalTime})')
    return cost

lowb=np.array([0,0,0,0,-18,-18,-18,-18,-18,-18,-18,-18,-5,-28,-18])
upbga=np.array([50,10,50,50,1,1,1,-3,1,1,1,1,5,1,1])
bnds=Bounds(lb=lowb,ub=upbga)
Ndim=len(lowb)

log.info('Start PSO')
pso = PSO(func=costFunction, n_dim=Ndim, pop=100, max_iter=50, lb=lowb,
        ub=upbga, w=0.8, c1=0.5, c2=0.5)
# pso = PSO(func=costFunction, n_dim=Ndim, pop=2, max_iter=2, lb=lowb,
#         ub=upbga, w=0.8, c1=0.5, c2=0.5)
pso.run()
log.info(f'best_x is {pso.gbest_x}, best_y is {pso.gbest_y}')
plt.plot(pso.gbest_y_hist)
plt.savefig('Edita_RNA_PSO.png')
plt.savefig('Edita_RNA_PSO.pdf')

log.info('Run Nelder-Mead on PSO result')
result = minimize(costFunction, pso.gbest_x, method='nelder-mead',bounds=bnds,options={'maxiter':100})
#result = minimize(costFunction, pso.gbest_x, method='nelder-mead',bounds=bnds,options={'maxiter':100})

log.info('Prepare GA population')
psox=pso.pbest_x
psoy=pso.pbest_y
psoDF=pd.DataFrame(psox)
psoDF['Cost']=psoy
psoDF.to_csv('Edita_RNA_best_pso.csv')
idx=np.argsort(psoy.flatten())
gainit=np.vstack([psox[idx[:(2*Ndim-1)]],result.x])
# log.info('Read GA population')
# initDF=pd.read_csv('Edita_best.csv')
log.info('GA starts')
ga = RCGA(func=costFunction, n_dim=Ndim, size_pop=2*Ndim, max_iter=50000, prob_mut=0.01, lb=lowb,
        ub=upbga)
# ga = RCGA(func=costFunction, n_dim=Ndim, size_pop=2, max_iter=2, prob_mut=0.01, lb=lowb,
#         ub=upbga)
ga.Chrom = (gainit-lowb)/(upbga-lowb)

#for cnt in range(2):
for cnt in range(200):
    log.info(f'Continue GA {cnt}')
    best_x, best_y = ga.run(10)
    #best_x, best_y = ga.run(1)
    log.info(f'GA done {cnt}: cfCounter={cfCounter}, best CF={best_y}')
    log.info(f'best par={best_x} ({len(best_x)}')
    best_line = np.append(cnt,best_x)
    best_line = np.append(best_line,best_y)
    log.info(f'best line={best_line} ({len(best_line)})')
    df = pd.DataFrame(best_line).T
    log.info(f'best df={df}')
    df.to_csv('Edita_RNA_best_pso_ga.csv',header=False,mode='a')
    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    #plt.show()
    plt.savefig('Edita_RNA_GA.png')
    plt.savefig('Edita_RNA_GA.pdf')
    chrom = ga.Chrom
    log.info('Run Nelder-Mead on the best GA result {cnt}')
    result = minimize(costFunction, ga.best_x, method='nelder-mead', bounds=bnds, options={'maxiter': 140})
    #result = minimize(costFunction, ga.best_x, method='nelder-mead', bounds=bnds, options={'maxiter': 2})
    chrom[np.argmax(ga.Y), :] = (result.x - lowb) / (upbga - lowb)
    ga.Chrom=chrom

