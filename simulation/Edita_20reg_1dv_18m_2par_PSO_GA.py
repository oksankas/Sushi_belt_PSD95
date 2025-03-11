# If in Jupyter or within ipython run both following lines
# If in the terminal run nrnivmodl to load Neuron engine
# %%bash
# nrnivmodl

import sys
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

def sushi_system(a, b, c, d, l):
    """
    Returns a matrix A, such that dx/dt = A*x

    N = # of compartments
    A is (2N x 2N) matrix
    x is (2N x 1) vector.
      The first N elements correspond to concentrations of u (molecules in transit)
      The second half correspond to concentrations of u-star (active molecules)
    The trafficking rate constants along the microtubules are given by the vectors "a" and "b"
    The rate constants for u turning into u* is given by the vector "c"
    The rate constants for u* turning into u is given by the vector "d"
    The rate constants for the degradation of u* is given by the vector "l"
    """
    # number of compartments
    N = len(l)

    ## State-space equations
    #  dx/dt = Ax + Bu
    A = np.zeros((2 * N, 2 * N))

    # Trafficking along belt
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
            # Out of parent, into child
            ai = a.pop()
            A[p, p] += -ai
            A[i, p] += ai
            # Into parent, out of child
            bi = b.pop()
            A[p, i] += bi
            A[i, i] += -bi

        # visit all segments in compartment
        for (j, seg) in enumerate(section):
            # Deal with out/into rates within compartment, just tridiag matrix
            if j > 0:
                # Out of parent, into child
                ai = a.pop()
                A[i - 1, i - 1] += -ai
                A[i, i - 1] += ai
                # Into parent, out of child
                bi = b.pop()
                A[i - 1, i] += bi
                A[i, i] += -bi
            # move onto next compartment
            i += 1

        # now visit children in pre-order
        child_list = list(h.SectionRef(sec=section).child)
        if child_list is not None:
            child_list.reverse()
        for c_sec in child_list:
            parentStack.append([i - 1, c_sec])  # append parent index and child

    # Trafficking off the belt
    for i in range(N):
        A[i, i] += -c[i]
        A[i + N, i] += c[i]

    # Reattachment to belt
    # for i in range(N):
    #    # reattachment
    #    A[i, i + N] += d[i]
    #    A[i + N, i + N] += -d[i]

    # Degradation after being taken off the belt
    for i in range(N):
        A[i + N, i + N] = -l[i]

    return A


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


def get_sys_matrix(utarg, F=0.5, Ctau=1e-3, dscale=0.1, dv=1e-7):
    # F is a mixing factor between 0 and 1
    K = np.sum(utarg) / N
    x = trafficking_solution(F * utarg + (1 - F) * K)
    a = (1 / (1 + x))
    a = list(a)
    b = list((1 / (1 + x ** -1)))
    l = list(np.ones(N) * dv)
    c = list(Ctau * utarg / (F * utarg + (1 - F) * K))
    d = list([ci * dscale for ci in c])
    A = sushi_system(a, b, c, d, l)
    return A

def solve_u(u0,w,V,Vinv,t):
    D = np.diag(np.exp(w*t))          # diagonal matrix exponential
    PHI = np.real(V.dot(D.dot(Vinv))) # state transition matrix
    return PHI.dot(u0)                # calculate u(t)

def sim_time(A,u0,time,nframes=10):
    # Run a simulation (log time)
    # --> this is a linear system; thus, matrix exponential provides exact solution
    utrace = [u0]
    w,V = scipy.linalg.eig(A)
    Vinv = np.linalg.inv(V)
    t = np.logspace(-0.5,math.log10(time),nframes)
    for t_ in t: utrace.append(solve_u(u0,w,V,Vinv,t_))
    return np.array(utrace).T

bgSignal = 1e-5

bpar = [0.8861278052764772,-5.358123919558814,0.3127455791698551,
        -5.7356821635860165,-17.999032750284783,0.21382364663804718,
        0.06074280874523973,0.04781382417616123,0.04562340830020193,
        1.0000000000039887e-07,0.7733439929081787,0.5543045057407645,
        0.1468913931288651,0.14710118884157364,0.1316203146580693,
        0.15499357159791044,0.17296720163305693,0.20287137824434023,
        0.3735878952746845,0.5243019440900127,0.043864102155039865,
        0.1966480252602025,0.19591970216660096,0.2755074889018492,
        0.9999999999999992]
def calcUtrace(par,delta=bgSignal):
    F = bpar[0]
    Ctau = 10 ** bpar[1]
    mProp = bpar[2]
    dvA = par[0]
    dvB = par[1]
    dv = np.zeros(N)
    utarg = delta*np.ones(N)
    for k in range(N):
        if itarg[k] > 4:
            utarg[k] = bpar[itarg[k]]
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
    A = sushi_system(a, b, c, d, l)
    u0 = np.concatenate((mProp * dinit, (1 - mProp) * dinit))
    utrace = sim_time(A, u0, day7)
    return utrace


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
subreg = ['CA1so', 'CA1sr', 'CA1slm']

#cname0='D0W3'
#cname0='D0M3'
cname0='D0M318'
d0w = -1 * np.ones(N)
for i in range(expD.shape[0]):
    abb = expD['Abbreviation'][i]
    sidx= segIdx[abb]
    d0w[sidx] *= -1*expD[f"{cname0}_MEAN"][i]/len(sidx)
for i in range(N):
    if d0w[i]<0:
        d0w[i] = bgSignal
dinit = d0w/np.sum(d0w)

#cname7='D7W3'
#cname7='D7M3'
cname7='D7M318'
d7w = -1*np.ones(N)
for i in range(expD.shape[0]):
    abb = expD['Abbreviation'][i]
    sidx= segIdx[abb]
    d7w[sidx] *= -1 * expD[f"{cname7}_MEAN"][i]/len(sidx)
for i in range(N):
    if d7w[i]<0:
        d7w[i] = bgSignal
target = np.array(expD[f"{cname7}_MEAN"])/np.sum(expD[f"{cname0}_MEAN"]) #norm target to Day0 sum to take into accound degradation
targSD = np.array(expD[f"{cname7}_SD"])/np.sum(expD[f"{cname0}_MEAN"]) #results to fit to
tnorm = np.sum(target ** 2)
day7 = 7 * 24 * 3600 # final time point

itarg = np.ones(N, dtype=int)
for i in range(expD.shape[0]):
    abb = expD['Abbreviation'][i]
    sidx = segIdx[abb]
    itarg[sidx] *= (i+5)

log.info("data read")

cfCounter = 0
dumpCSV = True
def costFunction(par):
    initTime = time.time()
    global cfCounter, dumpCSV
    cfCounter += 1
    log.info(f'Cost function starts: {cfCounter}')
    log.info(f'{par}')
    mProp = bpar[2]
    utrace = calcUtrace(par)
    resM, resF = sushibelt.aggregate_segments(utrace[:, -1], segIdx, expD['Abbreviation'], fun=np.sum)
    #cost=np.sum((resF + resM - target) ** 2)/tnorm
    #cost=np.sum((resF/(1-mProp) - target) ** 2)#/tnorm
    if any(np.isnan(resM)):
        cost = np.sum([ any(np.isnan(utrace[:,i])) for i in range(utrace.shape[1])])/utrace.shape[1]*sys.float_info.max
    else:
        #cost = np.sum(((resF / (1 - mProp) - target) / targSD) ** 2)
        cost = np.sum((resF / (1 - mProp) - target) ** 2)  # /tnorm
    FinalTime = time.time() - initTime
    if dumpCSV:
        chi2 = np.sum(((resF / (1 - mProp) - target) / targSD) ** 2)
        best_line = np.append(cfCounter, par)
        best_line = np.append(best_line, [cost,chi2])
        df = pd.DataFrame(best_line).T
        #log.info(f'best df={df}')
        df.to_csv('Edita_20reg_1dv_18m_2par_cf.csv',header=False,mode='a')
    log.info(f'Cost function done: cost={cost}. ({FinalTime})')
    return cost

lowb=np.array([-18,-18])
upbga=np.array([1,1])
bnds=Bounds(lb=lowb,ub=upbga)
Ndim=len(lowb)

log.info('Start PSO')
pso = PSO(func=costFunction, n_dim=Ndim, pop=30, max_iter=50, lb=lowb,
        ub=upbga, w=0.8, c1=0.5, c2=0.5)
pso.run()
log.info(f'best_x is {pso.gbest_x}, best_y is {pso.gbest_y}')
plt.plot(pso.gbest_y_hist)
plt.savefig('Edita_20reg_1dv_18m_2par_PSO.png')
plt.savefig('Edita_20reg_1dv_18m_2par_PSO.pdf')

log.info('Run Nelder-Mead on PSO result')
result = minimize(costFunction, pso.gbest_x, method='nelder-mead',bounds=bnds,options={'maxiter':100})

log.info('Prepare GA population')
psox=pso.pbest_x
psoy=pso.pbest_y
psoDF=pd.DataFrame(psox)
psoDF['Cost']=psoy
psoDF.to_csv('Edita_20reg_1dv_18m_2par_best_pso.csv')
idx=np.argsort(psoy.flatten())
gainit=np.vstack([psox[idx[:(2*Ndim-1)]],result.x])
# log.info('Read GA population')
# initDF=pd.read_csv('Edita_best.csv')
log.info('GA starts')
ga = RCGA(func=costFunction, n_dim=Ndim, size_pop=2*Ndim, max_iter=50000, prob_mut=0.01, lb=lowb,
        ub=upbga)
ga.Chrom = (gainit-lowb)/(upbga-lowb)
bestX = result.x
bestY = result.fun

for cnt in range(200):
    log.info(f'Continue GA {cnt}')
    best_x, best_y = ga.run(10)
    log.info(f'GA done {cnt}: cfCounter={cfCounter}, best CF={best_y}')
    log.info(f'best par={best_x} ({len(best_x)}')
    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    #plt.show()
    plt.savefig('Edita_20reg_1dv_18m_2r_GA.png')
    plt.savefig('Edita_20reg_1dv_18m_2par_GA.pdf')
    chrom = ga.Chrom
    bpar_dist = np.sum((bestX - ga.best_x) ** 2)
    if bpar_dist > 1e-7:
        log.info('Run Nelder-Mead on the best GA result {cnt}')
        result = minimize(costFunction, ga.best_x, method='nelder-mead', bounds=bnds, options={'maxiter': 140})
        chrom[np.argmax(ga.Y), :] = (result.x - lowb) / (upbga - lowb)
        ga.Chrom=chrom
        bestX = result.x
        bestY = result.fun
        best_line = np.append(cnt, bestX)
        best_line = np.append(best_line, bestY)
        log.info(f'best line={best_line} ({len(best_line)})')
        df = pd.DataFrame(best_line).T
        df.index = [cfCounter]
        #log.info(f'best df={df}')
        df.to_csv('Edita_20reg_1dv_18m_2par_best_pso_ga.csv', header=False, mode='a')
    else :
        bestX = ga.best_x
        bestY = best_y
        log.info(f'best par distance = {bpar_dist} and Nelder-Mead run on the best GA result is omitted.')

