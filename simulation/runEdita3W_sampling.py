import logging
FORMAT = '%(asctime)s :: %(levelname)s :: %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)
log = logging.getLogger("Edita_GA-logger")

import os
chunk = 0 #int(os.environ['SGE_TASK_ID'])
chunkSize = 256 #int(os.environ['chunkSize'])
log.info(f"chunk={chunk}, chunkSize={chunkSize}")
seed = 255 #int(os.environ['seed'])
log.info(f'seed:{seed}')

import Edita_model as model
#import Edita_20reg_1dv_model as model
import Sample_Sobol as rn

logFolder = 'sobol20240621'
log.info(f'output folder: {logFolder}')
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