
import numpy as np
import math
import pandas as pd
import os
#np.random.seed(123456789)

import logging
FORMAT = '%(asctime)s :: %(levelname)s :: %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)
log = logging.getLogger("GA-logger")

#import DG_10reg_10dv_model as model
import DG_10reg_1dv_model as model
import Optimize_PSO_GA as rn

runPrefix = 'optimization_pso10_nm10_ga10'
rn.runPrefix = runPrefix
logFolder = 'DG_1dv_PSO_GA__20240821'
#logFolder = os.environ['odir']

# Check whether the specified path exists or not
isExist = os.path.exists(logFolder)
if not isExist:
    # Create a new directory because it does not exist
       os.makedirs(logFolder)
       log.info(f"Work folder {logFolder} is created.")

model.logFolder=logFolder
#model.lowb[0]=0.8
rn.logFolder=logFolder

bestX,bestY,bestChi2 = rn.optimizeCF(model,pso_iter=150, nm_iter=100, ga_cycles=200,prob_mut=0.01)
#quick test setup
#bestX,bestY,bestChi2 = rn.optimizeCF(model,pso_iter=1, nm_iter=5, ga_cycles=2,prob_mut=0.01)