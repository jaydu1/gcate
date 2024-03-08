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
path_data = 'data/ex2/{}/'.format(data_family)
path_result = 'result/ex2/{}/'.format(data_family)
os.makedirs(path_result, exist_ok=True)

intercept = 1
offset = 0

p = 3000


n_list = [100,250]
r_list = [2,10]
n = n_list[int(sys.argv[1])]
r = r_list[int(sys.argv[2])]

i_seed = int(sys.argv[3])
for seed in range(i_seed,i_seed+1):
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
    

    r_max = 12
    df_r = estimate_r(Y, X, r_max, 
        kwargs_glm=kwargs_glm, 
        kwargs_es_1={'max_iters': 600},
        kwargs_es_2={'max_iters': 600},
        kwargs_ls_2={'alpha':0.01},
        c1=0.02,
        intercept=intercept, offset=offset, C=1e5, verbose=False
    )
    
    r_hat = int(df_r.iloc[df_r['JIC'].idxmin()]['r'])
    print(df_r, r_hat)

    lams = np.r_[
            np.linspace(0,0.01,11)[1:],
            np.linspace(0,0.1,11)[2:],
            np.linspace(0,1,11)[2:]
            ]
    df_res_raw, df_res = fit_gcate(
        Y, X, r_hat, d-1,
        kwargs_glm=kwargs_glm, 
        intercept=intercept, offset=offset,
        kwargs_ls_2={'alpha':0.01},
        kwargs_es_1={'max_iters': 600},
        kwargs_es_2={'max_iters': 600},
        c1=0.02,
        c2=lams
    )
    
    df_res_raw['signals'] = np.tile(B_true[:,-1][id_genes]!=0, len(lams)).astype(np.float32)    
    df_res_raw.to_csv(path_result+'gcate_raw_{}_{}_{}_{}.csv'.format(family,n,r,seed))
    
    df_res['signals'] = np.tile(B_true[:,-1][id_genes]!=0, len(lams)).astype(np.float32)
    df_res.to_csv(path_result+'gcate_{}_{}_{}_{}.csv'.format(family,n,r,seed))
