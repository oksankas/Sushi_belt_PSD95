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
log = logging.getLogger("GA-logger")

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

def calcUtrace(par,delta=bgSignal):
    F = par[0]
    Ctau = 10 ** par[1]
    mProp = par[2]
    dv = np.zeros(N)
    utarg = delta*np.ones(N)
    for k in range(N):
        if itarg[k] > 2:
            dv[k] = 10 ** par[itarg[k]]
            utarg[k] = par[itarg[k]+3]
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
tdf=pd.read_csv('../data/seg_mapping.csv')
abbCA1=tdf['abb']
abbT={}
segIdx={}
for i in range(N):
    abbT[abbCA1[i]] = 1+ abbT.get(abbCA1[i],0)
    ll=segIdx.get(abbCA1[i],[])
    ll.append(i)
    segIdx[abbCA1[i]] = ll

expD=pd.read_csv('../data/CA1_gradient.csv')
subreg = ['CA1so', 'CA1sr', 'CA1slm']

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

cname7='D7M3'
d7w = -1*np.ones(N)
for i in range(expD.shape[0]):
    abb = expD['Abbreviation'][i]
    sidx= segIdx[abb]
    d7w[sidx] *= -1 * expD[f"{cname7}_MEAN"][i]/len(sidx)
for i in range(N):
    if d7w[i]<0:
        d7w[i] = bgSignal
target = np.array(expD[f"{cname7}_MEAN"])/np.sum(expD[f"{cname0}_MEAN"]) #norm target to Day0 sum to take into accound degradation
targSD = np.array(expD[f"{cname7}_SD"])/np.sum(expD[f"{cname0}_MEAN"]) #measurement errors
tnorm = np.sum(target ** 2)
day7 = 7 * 24 * 3600 # final time point

itarg = np.ones(N, dtype=int)
for i in range(expD.shape[0]):
    abb = expD['Abbreviation'][i]
    sidx = segIdx[abb]
    itarg[sidx] *= [j + 3 for j in range(len(subreg)) if subreg[j] == expD['Subregion'][i]][0]

log.info("data read")

cfiCounter = 0
cfCounter = 0
dumpCSV = True
def costFunction(par):
    initTime = time.time()
    global cfCounter, dumpCSV
    cfCounter += 1
    log.info(f'Cost function starts: {cfCounter}')
    log.info(f'{par}')
    mProp = par[2]
    utrace = calcUtrace(par)
    resM, resF = sushibelt.aggregate_segments(utrace[:, -1], segIdx, expD['Abbreviation'], fun=np.sum)
    cost=np.sum(((resF/(1-mProp) - target)/targSD) ** 2)
    FinalTime = time.time() - initTime
    if dumpCSV:
        best_line = np.append(cfCounter, par)
        best_line = np.append(best_line, cost)
        df = pd.DataFrame(best_line).T
        #log.info(f'best df={df}')
        df.to_csv('CA1_3dv_cfi.csv',header=False,mode='a')
    log.info(f'Cost function done: cost={cost}. ({FinalTime})')
    return cost

lowb=np.array([0, -18, 1e-7, -18, -18, -18, 1e-3, 1e-3, 1e-3])
upbga=np.array([1, -1, 1-1e-7, 1, 1, 1, 1, 1, 1])
bnds=Bounds(lb=lowb,ub=upbga)

parnames=['F','Ctau','mProp','dv_CA1so','dv_CA1sr','dv_CA1slm','demand_CA1so','demand_CA1sr','demand_CA1slm']
Nvals = 4 #10
parvals = [1.0 * i/Nvals for i in range(Nvals+1)]
numPar=len(parnames)

def prepPar(cpar,pn,pval,pidx,bidx):
    lpar = np.zeros(numPar)
    lpar[pidx] = pval
    for j in range(numPar - 1):
        lpar[bidx[j]] = cpar[j]
    return lpar

def profileChiSq(pn,pv=0,pso_iter=15,nm_iter=100,ga_cycles=200):
    pidx=[i for i in range(numPar) if parnames[i]==pn][0]
    pval=lowb[pidx]+(upbga[pidx]-lowb[pidx])*pv
    bidx= [i for i in range(numPar) if i != pidx]
    log.info(f'ChiSq starts: numPar={numPar}, pn={pn}, pidx={pidx}, pv={pv}, pval={pval}.')
    clowb=lowb[bidx]
    cupbga=upbga[bidx]
    cbnds=Bounds(lb=clowb,ub=cupbga)
    def costFunctionC(cpar):
        global cfiCounter
        cfiCounter += 1
        lpar = prepPar(cpar,pn,pval,pidx,bidx)
        pDF = pd.DataFrame(lpar).T
        pDF.index = [cfiCounter]
        pDF['ParamName'] = pn
        pDF['ParamVal'] = pval
        cost = costFunction(lpar)
        pDF['Cost'] = cost
        pDF.to_csv('CA1_3dv_cf_idnt.csv',header=False,mode='a')
        return cost

    log.info(f'{pn}={pval}, Start PSO')
    # Population size for the PSO should be at least 2*numPar and in ideal situation 5*numPar
    pso = PSO(func=costFunctionC, n_dim=(numPar - 1), pop=2*numPar, max_iter=pso_iter, lb=clowb,
            ub=cupbga, w=0.8, c1=0.5, c2=0.5)
    pso.run()
    log.info(f'{pn}={pval}, best_x is {pso.gbest_x}, best_y is {pso.gbest_y}')
    log.info(f'{pn}={pval}, Run Nelder-Mead on PSO result')
    result = minimize(costFunctionC, pso.gbest_x, method='nelder-mead',bounds=cbnds,options={'maxiter':nm_iter})

    log.info(f'{pn}={pval}, Prepare GA population')
    #use pso.gbest_x and pso.gbest_y to get access to the global optimum found
    psox = pso.pbest_x
    psoy=pso.pbest_y
    idx=np.argsort(psoy.flatten())
    gainit=np.vstack([psox[idx[:(2*(numPar-1)-1)]],result.x])
    #log.info(f'{pn}={pval}, psox: {psox.shape}, idx: {len(idx)}, pop_size={2*(numPar-1)}, gainit: {gainit.shape}')
    log.info(f'{pn}={pval}, GA starts, population dimensions: {gainit.shape}')
    ga = RCGA(func=costFunctionC, n_dim=(numPar-1), size_pop=2*(numPar-1), max_iter=50000, prob_mut=0.01, lb=clowb,
            ub=cupbga)
    ga.Chrom = (gainit-clowb)/(cupbga-clowb)
    bestX=result.x
    bestY=result.fun

    for cnt in range(ga_cycles):
        log.info(f'{pn}={pval}, Continue GA {cnt}')
        best_x, best_y = ga.run(10)
        best_x_p=prepPar(best_x,pn,pval,pidx,bidx)
        log.info(f'GA {pn}={pval} done {cnt}: cfCounter={cfCounter}, best CF={best_y}')
        log.info(f'{pn}={pval}, best par={best_x} ({len(best_x)})')
        best_line = np.append(cnt,best_x_p)
        #log.info(f'{pn}={pval}, best line={best_line} ({len(best_line)})')
        chrom = ga.Chrom
        bpar_dist = np.sum((bestX-ga.best_x) ** 2)
        if bpar_dist > 1e-7 :
            log.info(f'{pn}={pval}, Run Nelder-Mead on the best GA result {cnt}, best par distance = {bpar_dist}')
            result = minimize(costFunctionC, ga.best_x, method='nelder-mead', bounds=cbnds, options={'maxiter': nm_iter})
            chrom[np.argmax(ga.Y), :] = (result.x - clowb) / (cupbga - clowb)
            ga.Chrom=chrom
            bestX = result.x
            bestY = result.fun
        else :
            bestX = ga.best_x
            bestY = best_y
            log.info(f'{pn}={pval}, best par distance = {bpar_dist} and Nelder-Mead run on the best GA result is omitted.')
        log.info(f'GA {pn}={pval} completed {cnt}: cfCounter={cfCounter}, best CF={bestY}')
    return bestX, bestY

chiCounter = 0
i = 0
jval = [1 + k for k in range(Nvals)]

for pni in parnames :
    for pvj in parvals :
        cfiCounter = 0
        #pni = parnames[i]
        bestX, bestY = profileChiSq(pni,pvj,pso_iter=10,nm_iter=10,ga_cycles=10)
        pidx=[k for k in range(numPar) if parnames[k]==pni][0]
        pval=lowb[pidx]+(upbga[pidx]-lowb[pidx])*pvj
        bidx= [k for k in range(numPar) if k != pidx]
        bestX_p = prepPar(bestX, pni, pval, pidx, bidx)
        log.info(f'GA {pni}={pval} best found: cfCounter={cfCounter}, best CF={bestY}')
        log.info(f'{pni}={pval}, best found: par={bestX_p} ({len(bestX_p)}')
        bestLine = np.append(cfiCounter, bestX_p)
        log.info(f'{pni}={pval}, best found: line={bestLine} ({len(bestLine)})')
        bdf = pd.DataFrame(bestLine).T
        bdf['ParamName'] = pni
        bdf['ParamVal'] = pval
        bdf['Cost'] = bestY
        bdf.index = [chiCounter]
        log.info(f'{pni}={pval}, best found: df={bdf}')
        bdf.to_csv('CA1_3dv_ident.csv', header=False, mode='a')
        chiCounter += 1
