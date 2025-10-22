
import numpy as np
import pandas as pd

import logging
FORMAT = '%(asctime)s :: %(levelname)s :: %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)
log = logging.getLogger("GA-logger")

import os

#logdir = os.environ['odir']
logdir = 'ident_diff_CA1_20reg_1dv_3M_20240926'#'ident_diff_CA1_20reg_1dv_3M_20240925'

import CA1_20reg_1dv_model as model
import Identifiability_diffuse as rn

opt_par = [0.8861278052764772,-5.358123919558814,0.3127455791698551,-5.7356821635860165,-17.999032750284783,
           0.21382364663804718,0.06074280874523973,0.04781382417616123,0.04562340830020193,1.0000000000039887e-07,
           0.7733439929081787,0.5543045057407645,0.1468913931288651,0.14710118884157364,0.1316203146580693,
           0.15499357159791044,0.17296720163305693,0.20287137824434023,0.3735878952746845,0.5243019440900127,
           0.043864102155039865,0.1966480252602025,0.19591970216660096,0.2755074889018492,0.9999999999999992]

niter = 50
#niter = 100

pni = 'mProp'
modelPrefix = f'{model.modelPrefix}_{pni}'
model.modelPrefix = modelPrefix
runPrefix = 'identifiability_pso10_nm10_ga10'
rn.runPrefix = runPrefix
logFolder = logdir #'ident20240620'
# Check whether the specified path exists or not
isExist = os.path.exists(logFolder)
if not isExist:
    # Create a new directory because it does not exist
       os.makedirs(logFolder)
       log.info(f"Work folder {logFolder} is created.")

model.logFolder=logFolder
rn.logFolder=logFolder


log.info(f"Identifiability diffusing along parameter '{pni}' saved to {logFolder}.")
chiCounter = 0

numPar = len(model.parnames)
rn.cfiCounter = 0
log.info(f"pni={pni}, numPar={numPar}")

resList = rn.profileChiSq(model, opt_par=opt_par,pn=pni,step=0.01,nm_iter=niter)
