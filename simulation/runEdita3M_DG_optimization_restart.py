
import numpy as np
import math
import pandas as pd
import os
#np.random.seed(123456789)

import logging
FORMAT = '%(asctime)s :: %(levelname)s :: %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)
log = logging.getLogger("Edita_GA-logger")

#import Edita_DG_10reg_10dv_model as model
import Edita_DG_10reg_1dv_model as model
import Optimize_GA_restart as rn

runPrefix = 'optimization_nm10_ga200_restart'
rn.runPrefix = runPrefix
logFolder = 'DG_1dv_PSO_GA_20240902'
initFile = '/Users/anatolii-sorokin/Documents/Projects/neuron-model/DG_1dv_PSO_GA__20240821/Edita_DG_10reg_1dv_3M_optimization_pso10_nm10_ga10_best_pso_ga.csv'
#logFolder = os.environ['odir']
#initFile = os.environ['ifile']

# Check whether the specified path exists or not
isExist = os.path.exists(logFolder)
if not isExist:
    # Create a new directory because it does not exist
       os.makedirs(logFolder)
       log.info(f"Work folder {logFolder} is created.")

model.logFolder=logFolder
#model.lowb[0]=0.8
rn.logFolder=logFolder

bestX,bestY,bestChi2 = rn.optimizeCF(model,initFile, nm_iter=100, ga_cycles=200,prob_mut=0.01)
#quick test setup
#bestX,bestY,bestChi2 = rn.optimizeCF(model,pso_iter=1, nm_iter=5, ga_cycles=2,prob_mut=0.01)