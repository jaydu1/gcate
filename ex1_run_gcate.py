import os
import sys
os.environ["OMP_NUM_THREADS"] = "11"
os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "11" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "11" # export NUMEXPR_NUM_THREADS=6
os.environ["NUMBA_THREADING_LAYER"] = "omp"
os.environ["NUMBA_CACHE_DIR"]='/tmp/numba_cache'


from gcate import *
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


family = 'poisson'
kwargs_glm = {'family':family}

data_family = 'poisson'
path_data = 'data/ex1/{}/'.format(data_family)
path_result = 'result/ex1/{}/'.format(data_family)
os.makedirs(path_result, exist_ok=True)

intercept = 1
offset = 0

p = 3000
n = 250; r = 2

for seed in range(100):
    if os.path.exists(path_result+'gcate_{}_{}_{}_{}.csv'.format(family,n,r,seed)):
        pass
    print('n:{}, r:{}, seed:{}'.format(n,r,seed))
    
    file_name = path_data + 'n_{}_r_{}_seed_{}.npz'.format(n,r,seed)
    with open(file_name, 'rb') as f:
        tmp = np.load(f)
        X = tmp['X']
        X = X[:,1:]
        B_true = tmp['B']
        Y = tmp['Y'].astype(float)
        n, d = X.shape
    print(X.shape, Y.shape)
    id_genes = np.any(Y, axis=0)
    p_all = Y.shape[1]
    Y = Y[:,id_genes]
    
    lams = np.r_[
            np.linspace(0,0.01,11)[1:],
            np.linspace(0,0.1,11)[2:],
            np.linspace(0,1,11)[2:]
            ]
    
    for ratio_infer in np.linspace(0,1,6)[1:-1]:
        df_res_raw, df_res = fit_gcate(
            Y, X, r, d-1,
            kwargs_glm=kwargs_glm, 
            intercept=intercept, offset=offset,
            kwargs_ls_2={'alpha':0.01},
            kwargs_es_1={'max_iters': 600},
            kwargs_es_2={'max_iters': 600},
            c1=0.02,
            c2=lams,
            ratio_infer = ratio_infer
        )

        df_res_raw['signals'] = np.tile(B_true[:,-1][id_genes]!=0, len(lams)).astype(np.float32)    
        df_res_raw.to_csv(path_result+'gcate_raw_{}_{}_{}_{:.01f}_{}.csv'.format(family,n,r,ratio_infer,seed))

        df_res['signals'] = np.tile(B_true[:,-1][id_genes]!=0, len(lams)).astype(np.float32)
        df_res.to_csv(path_result+'gcate_{}_{}_{}_{:.01f}_{}.csv'.format(family,n,r,ratio_infer,seed))
