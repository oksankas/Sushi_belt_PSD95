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


def calcUtrace(par,delta=1e-7):
    F = par[0]
    Ctau = 10 ** par[1]
    dv = 10 ** par[2]
    mProp = par[3]
    utarg = delta*np.ones(N)
    for k in range(N):
        if itarg[k] > 3:
            utarg[k] = par[itarg[k]]
    utarg /= np.sum(utarg)
    K = np.sum(utarg) / N
    x = trafficking_solution(F * utarg + (1 - F) * K)
    a = (1 / (1 + x))
    a = list(a)
    b = list((1 / (1 + x ** -1)))
    l = list(np.ones(N) * dv)
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
bgSignal = 1e-5

cname='D0W3'
d0w = -1 * np.ones(N)
for i in range(expD.shape[0]):
    abb = expD['Abbreviation'][i]
    sidx= segIdx[abb]
    d0w[sidx] *= -1*expD[f"{cname}_MEAN"][i]/len(sidx)
for i in range(N):
    if d0w[i]<0:
        d0w[i] = bgSignal
dinit = d0w/np.sum(d0w)

cname='D7W3'
d7w = -1*np.ones(N)
for i in range(expD.shape[0]):
    abb = expD['Abbreviation'][i]
    sidx= segIdx[abb]
    d7w[sidx] *= -1 * expD[f"{cname}_MEAN"][i]/len(sidx)
for i in range(N):
    if d7w[i]<0:
        d7w[i] = bgSignal
target = np.array(expD[f"{cname}_MEAN"])/np.sum(expD[f"{cname}_MEAN"]) #results to fit to
day7 = 7 * 24 * 3600 # final time point

itarg = np.ones(N, dtype=int)
for i in range(expD.shape[0]):
    abb = expD['Abbreviation'][i]
    sidx = segIdx[abb]
    itarg[sidx] *= [j + 4 for j in range(len(subreg)) if subreg[j] == expD['Subregion'][i]][0]

log.info("data read")

cfCounter = 0
dumpCSV = True
def costFunction(par):
    initTime = time.time()
    global cfCounter, dumpCSV
    cfCounter += 1
    log.info(f'Cost function starts: {cfCounter}')
    log.info(f'{par}')
    utrace = calcUtrace(par)
    resM, resF = sushibelt.aggregate_segments(utrace[:, -1], segIdx, expD['Abbreviation'], fun=np.sum)
    cost=np.sum((resF - target) ** 2)
    FinalTime = time.time() - initTime
    if dumpCSV:
        best_line = np.append(cfCounter, par)
        best_line = np.append(best_line, cost)
        df = pd.DataFrame(best_line).T
        log.info(f'best df={df}')
        df.to_csv('Edita_cf.csv',header=False,mode='a')
    log.info(f'Cost function done: cost={cost}. ({FinalTime})')
    return cost

lowb=np.array([0, -18, -18, 1e-3, 1e-3, 1e-3, 1e-3])
upbga=np.array([1, -1, -1, 1, 1, 1, 1])
gaPop=2*len(lowb)

log.info('Read GA population')
initDF=pd.read_csv('Edita_best.csv')
initDFu=initDF.drop_duplicates(subset=['F','Ctau','dv','mProp','demand_CA1so','demand_CA1sr','demand_CA1slm'])
old_x=np.array(initDFu.loc[:,['F','Ctau','dv','mProp','demand_CA1so','demand_CA1sr','demand_CA1slm']])
old_y=np.array(initDFu['Cost'])
idx=np.argsort(old_y)
if len(idx)< gaPop:
    chrom=np.concatenate([(old_x-lowb)/(upbga-lowb),np.random.rand(gaPop-len(idx),7)])
else:
    chrom=(old_x[idx[:gaPop]]-lowb)/(upbga-lowb)
log.info('GA starts')
ga = RCGA(func=costFunction, n_dim=7, size_pop=gaPop, max_iter=50000, prob_mut=0.01, lb=lowb,
        ub=upbga)
ga.Chrom = chrom

for cnt in range(200):
    log.info(f'Continue GA {cnt}')
    best_x, best_y = ga.run(10)
    log.info(f'GA done {cnt}: cfCounter={cfCounter}, best CF={best_y}')
    log.info(f'best par={best_x} ({len(best_x)}')
    best_line = np.append(cnt,best_x)
    best_line = np.append(best_line,best_y)
    log.info(f'best line={best_line} ({len(best_line)})')
    df = pd.DataFrame(best_line).T
    log.info(f'best df={df}')
    df.to_csv('Edita_best_pso_ga.csv',header=False,mode='a')
    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    #plt.show()
    plt.savefig('EditaGArest.png')
    plt.savefig('EditaGArest.pdf')

