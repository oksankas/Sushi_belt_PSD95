
import numpy as np
import pandas as pd

import logging
FORMAT = '%(asctime)s :: %(levelname)s :: %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)
log = logging.getLogger("GA-logger")

import os
chunk = int(os.environ['SGE_TASK_ID'])
chunkSize = int(os.environ['chunkSize'])
log.info(f"chunk={chunk}, chunkSize={chunkSize}")
seed = int(os.environ['seed'])
log.info(f'seed:{seed}')

logdir = os.environ['odir']
log.info(f'output folder: {logdir}')
import CA1_20reg_1dv_model as model
import Sample_Sobol as rn

logFolder = logdir #'sobol20240621'
# Check whether the specified path exists or not
isExist = os.path.exists(logFolder)
if not isExist:
    # Create a new directory because it does not exist
       os.makedirs(logFolder)
       log.info(f"Work folder {logFolder} is created.")

model.logFolder=logFolder
rn.logFolder=logFolder

log.info(f"Sobol sampling task for chunk '{chunk}' of size {chunkSize} saved to {logFolder}.")

res = rn.sampleModel(model,chunk=chunk,chunkSize=chunkSize,seed=seed)