# If in Jupyter or within ipython run both following lines
# If in the terminal run nrnivmodl to load Neuron engine
# %%bash
# nrnivmodl

import sys
from neuron import h
import numpy as np
import math
import scipy.linalg
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

from scipy.optimize import minimize, Bounds

import pandas as pd

import sushibelt
import time


def sushi_system(a, b, c, d, l,tr):
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
    The soma translation coefficient define the proportionality between translation rate and np.mean(l)
    """
    # number of compartments
    N = len(l)

    ## State-space equations
    #  dx/dt = Ax + Bu
    A = np.zeros((2 * N+1, 2 * N+1))

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
    # Soma translation
    A[0, 2*N] += tr*np.mean(l)

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


def get_sys_matrix(utarg, F=0.5, Ctau=1e-3, dscale=0.1, dv=1e-7, tr=1.0):
    # F is a mixing factor between 0 and 1
    K = np.sum(utarg) / N
    x = trafficking_solution(F * utarg + (1 - F) * K)
    a = (1 / (1 + x))
    a = list(a)
    b = list((1 / (1 + x ** -1)))
    l = list(np.ones(N) * dv)
    c = list(Ctau * utarg / (F * utarg + (1 - F) * K))
    d = list([ci * dscale for ci in c])
    A = sushi_system(a, b, c, d, l,tr)
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
    dV = np.linalg.det(V)
    #log.info(f'Time simulations: dimA={A.shape}, dimV={V.shape}, detV={dV}.')
    if np.abs(dV) > 1e-308 :
        #log.info(f'Time simulations NORMAL.')
        Vinv = np.linalg.inv(V)
        t = np.logspace(-0.5,math.log10(time),nframes)
        for t_ in t: utrace.append(solve_u(u0,w,V,Vinv,t_))
    else:
        log.info(f'Time simulations SINGULAR: dimA={A.shape}, dimV={V.shape}, detV={dV}.')
        for t_ in range(nframes): utrace.append(np.nan*u0)
    return np.array(utrace).T


bgSignal = 1e-5
day7 = 7 * 24 * 3600 # final time point
day90 = 90 * 24 * 3600 # final time point

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

#cname='D0W3'
cname0='D0M3'
d0w = -1 * np.ones(N)
for i in range(expD.shape[0]):
    abb = expD['Abbreviation'][i]
    sidx= segIdx[abb]
    d0w[sidx] *= -1*expD[f"{cname0}_MEAN"][i]/len(sidx)
for i in range(N):
    if d0w[i]<0:
        d0w[i] = bgSignal
dinit = d0w/np.sum(d0w)

#cname='D7W3'
cname7='D7M3'
d7w = -1*np.ones(N)
for i in range(expD.shape[0]):
    abb = expD['Abbreviation'][i]
    sidx= segIdx[abb]
    d7w[sidx] *= -1 * expD[f"{cname7}_MEAN"][i]/len(sidx)
for i in range(N):
    if d7w[i]<0:
        d7w[i] = bgSignal

itarg = np.ones(N, dtype=int)
for i in range(expD.shape[0]):
    abb = expD['Abbreviation'][i]
    sidx = segIdx[abb]
    itarg[sidx] *= (i+6)

log.info("data read")


def calcUtrace(par,delta=bgSignal):
    F = par[0]
    Ctau = 10 ** par[1]
    mProp = par[2]
    dvA = par[3]
    dvB = par[4]
    tr = 10 ** par[5]
    dv = np.zeros(N)
    utarg = delta*np.ones(N)
    for k in range(N):
        if itarg[k] > 5:
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
    A = sushi_system(a, b, c, d, l,tr)
    u1 = np.concatenate((mProp * dinit, (1 - mProp) * dinit,[1]))
    utrace1 = sim_time(A, u1, day90)
    #resM, resF = sushibelt.aggregate_segments(utrace1[:, -1], segIdx, expD['Abbreviation'], fun=np.sum)
    utrace0 = sim_time(A[:-1,:-1], u1[:-1], day7)
    utrace = np.vstack([utrace0,utrace1])
    return utrace


log.info("calcUtrace function defined")


####### Parameters to set by invocation script #######
dinit = d0w/np.sum(d0w)
target0 = np.array(expD[f"{cname0}_MEAN"])/np.sum(expD[f"{cname0}_MEAN"]) #norm target to Day0 sum to take into accound degradation
targ0SD = np.array(expD[f"{cname0}_SD"])/np.sum(expD[f"{cname0}_MEAN"]) #results to fit to
tnorm0 = np.sum(target0)
target7 = np.array(expD[f"{cname7}_MEAN"])/np.sum(expD[f"{cname0}_MEAN"]) #norm target to Day0 sum to take into accound degradation
targ7SD = np.array(expD[f"{cname7}_SD"])/np.sum(expD[f"{cname0}_MEAN"]) #results to fit to
tnorm7 = np.sum(target7)
logFolder = '.'
modelPrefix = 'Edita_20reg_1dv_dual_soma'
parnames=['F','Ctau','mProp','dvA','dvB','tr',
          'demand_CA1so_1','demand_CA1so_2','demand_CA1so_3','demand_CA1so_4','demand_CA1so_5',
          'demand_CA1sr_1','demand_CA1sr_2','demand_CA1sr_3','demand_CA1sr_4','demand_CA1sr_5',
          'demand_CA1sr_6','demand_CA1sr_7','demand_CA1sr_8','demand_CA1sr_9','demand_CA1sr_10',
          'demand_CA1slm_1','demand_CA1slm_2','demand_CA1slm_3','demand_CA1slm_4','demand_CA1slm_5']
lowb=np.array([0,-18,0.1,-7,-18,-3,1.0e-07,1.0e-07,1.0e-07,1.0e-07,1.0e-07,1.0e-07,1.0e-07,1.0e-07,1.0e-07,
               1.0e-07,1.0e-07,1.0e-07,1.0e-07,1.0e-07,1.0e-07,1.0e-07,1.0e-07,1.0e-07,1.0e-07,1.0e-07])
upbga=np.array([1,-1,0.9,1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
bnds=Bounds(lb=lowb,ub=upbga)
Ndim=len(lowb)
costWeight=[1,1,.05,.01]
######################################################

cfCounter = 0

def costFunction(par,retype='err'):
    initTime = time.time()
    global cfCounter, dumpCSV
    cfCounter += 1
    log.info(f'Cost function starts: {cfCounter}')
    log.info(f'{par}')
    mProp = par[2]
    utrace = calcUtrace(par)
    resM0, resF0 = sushibelt.aggregate_segments(utrace[:2*N, -1], segIdx, expD['Abbreviation'], fun=np.sum)
    resM1, resF1 = sushibelt.aggregate_segments(utrace[2 * N:, -1], segIdx, expD['Abbreviation'], fun=np.sum)
    if any(np.isnan(resM0)):
        cost0 = np.sum([ any(np.isnan(utrace[:,i])) for i in range(utrace.shape[1])])/utrace.shape[1]*sys.float_info.max/2
        cost2 = 0
    else:
        cost0 = np.sum((resF0 / (1 - mProp) - target7) ** 2)
        cost2 = np.abs(np.sum(resF0) + np.sum(resM0) - tnorm7)
    if any(np.isnan(resM1)):
        cost1 = np.sum([ any(np.isnan(utrace[:,i])) for i in range(utrace.shape[1])])/utrace.shape[1]*sys.float_info.max/2
        cost3 = 0
    else:
        cost1 = np.sum((resF1 / (1 - mProp) - target0) ** 2)
        cost3 = np.abs(np.sum(resF1) + np.sum(resM1) - tnorm0)
    FinalTime = time.time() - initTime
    costT = costWeight[0]*cost0 + costWeight[1]*cost1 + costWeight[2]*cost2 + costWeight[3]*cost3
    chi2_0 = np.sum(((resF0 / (1 - mProp) - target7) / targ7SD) ** 2)
    chi2_1 = np.sum(((resF1 / (1 - mProp) - target0) / targ0SD) ** 2)
    best_line = np.append(par, [costT,cost0,cost1,cost2,cost3,chi2_0,chi2_1])
    if cfCounter > 1:
        df = pd.DataFrame(best_line).T
        df.index = [cfCounter]
        df.to_csv(f'{logFolder}/{modelPrefix}_cf.csv', header=False, mode='a')
    else:
        cnames = parnames + ['Cost','CostD7','CostD0', 'PenaltyD7','PenaltyD0','Chi2D7', 'Chi2D0']
        df = pd.DataFrame(best_line, index=cnames).T
        df.index = [cfCounter]
        df.to_csv(f'{logFolder}/{modelPrefix}_cf.csv', header=True, mode='w')
    log.info(f'Cost function done: cost={(costT,cost0,cost1)}, chi2={(chi2_0,chi2_1)}. ({FinalTime})')
    if 'chi2' == retype:
        return (chi2_0,chi2_1)
    return costT


