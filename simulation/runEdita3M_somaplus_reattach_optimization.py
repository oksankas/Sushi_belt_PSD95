
import numpy as np
import math
import pandas as pd
import os
#np.random.seed(123456789)

import logging
FORMAT = '%(asctime)s :: %(levelname)s :: %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)
log = logging.getLogger("Edita_GA-logger")

import Edita_20reg_1dv_model_somaplus_translation_reattach as model
import Optimize_PSO_GA as rn

runPrefix = 'optimization_pso10_nm10_ga10'
rn.runPrefix = runPrefix
logFolder = 'somaplus_reattach_PSO_GA_20240725'
# Check whether the specified path exists or not
isExist = os.path.exists(logFolder)
if not isExist:
    # Create a new directory because it does not exist
       os.makedirs(logFolder)
       log.info(f"Work folder {logFolder} is created.")

model.logFolder=logFolder
rn.logFolder=logFolder

bestX,bestY,bestChi2 = rn.optimizeCF(model,pso_iter=10, nm_iter=10, ga_cycles=10,prob_mut=0.9)