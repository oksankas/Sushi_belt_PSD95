
import numpy as np
import math
import pandas as pd
import os
#np.random.seed(123456789)

import logging
FORMAT = '%(asctime)s :: %(levelname)s :: %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)
log = logging.getLogger("Edita_GA-logger")

import Edita_20reg_1dv_dual_model_soma as model
import Optimize_PSO_GA as rn

runPrefix = 'optimization_pso10_nm10_ga10'
rn.runPrefix = runPrefix
#logFolder = 'dual_soma_PSO_GA_20240717'
logFolder = os.environ['odir']

# Check whether the specified path exists or not
isExist = os.path.exists(logFolder)
if not isExist:
    # Create a new directory because it does not exist
       os.makedirs(logFolder)
       log.info(f"Work folder {logFolder} is created.")

model.logFolder=logFolder
#model.lowb[0]=0.8
rn.logFolder=logFolder

bestX,bestY,bestChi2 = rn.optimizeCF(model,pso_iter=50, nm_iter=100, ga_cycles=200,prob_mut=0.01)