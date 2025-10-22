# If in Jupyter or within ipython run both following lines
# If in the terminal run nrnivmodl to load Neuron engine
# %%bash
# nrnivmodl

from neuron import h
import numpy as np

np.random.seed(123456789)

import sushibelt
import time

# Load morphology and other stuff
# --> SegLists: soma[2], dend[74], dend_5[37], apic[42], axon[1]
# --> Files from Migliore & Migliore (2012)
# --> CA1 pyramidal neuron
h.load_file('stdrun.hoc')
h.xopen('ri06.hoc')
h.xopen('fixnseg.hoc')
h.xopen('5a_nogui.hoc')
h.tstop = 700.0

from sko.GA import GA

import pandas as pd

fitdt = pd.read_csv('../data/Distr.csv')
seg_idx = sushibelt.prepare_seg_index(h, fitdt, delta=10)

initTime = time.time()
A = sushibelt.make_uniform_reattachment_matrix(h, 0.1, 5, 50.0, 1e-8, 1e-5)
FinalTime = time.time() - initTime
print(f"Make matrix {FinalTime}")


def costFunction(par):
    initTime = time.time()
    A = sushibelt.make_uniform_reattachment_matrix(h, par[0], par[1], par[2], par[3], par[4])
    u, t = sushibelt.run_sim(h, A)
    dvec = sushibelt.make_dist_calc(seg_idx, u, fitdt, useAmount=True)
    FinalTime = time.time() - initTime
    print(f"Eval time {FinalTime}; parameters: {par}, result {dvec[-1]}")
    return dvec[-1]


ga = GA(func=costFunction, n_dim=5, size_pop=10, max_iter=500, prob_mut=0.01, lb=[0, 0, 0, 0, 0],
        ub=[2, 10, 10, 1e-5, 1e-5], precision=[1e-3, 1e-3, 1e-3, 1e-9, 1e-9])
best_x, best_y = ga.run(3)
