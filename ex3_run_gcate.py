import os
import sys
os.environ["OMP_NUM_THREADS"] = "11"
os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "11" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "11" # export NUMEXPR_NUM_THREADS=6
os.environ["NUMBA_CACHE_DIR"]='/tmp/numba_cache'


from gcate import *
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

path_data = 'data/ex3/'
path_result = 'result/ex3/'
os.makedirs(path_result, exist_ok=True)

family = 'negative_binomial'

p = 2500
d = 3
r = 3
n_list = [100,200]

for n in n_list:
    for seed in range(100):
        if os.path.exists(path_result+'gcate_adj_{}_{}_{}_{}.csv'.format(family,n,r,seed)):
            pass
        print('n:{}, r:{}, seed:{}'.format(n,r,seed))
        
        Y = pd.read_csv(path_data+'{}_{}_Y.csv'.format(n,seed)).values.astype(np.float32)

        id_genes = np.any(Y, axis=0) & (np.sum(Y>0, axis=0)>=10)
        p_all = Y.shape[1]
        Y = Y[:,id_genes]

        B_true = pd.read_csv(path_data+'{}_{}_B.csv'.format(n,seed)).values[:,0]
        X = pd.read_csv(path_data+'{}_{}_X.csv'.format(n,seed)).values

        if family=='negative_binomial':
            disp = estimate_disp(Y, X)
            
            kwargs_glm = {
                'family':'negative_binomial', 
                'nuisance':disp
            }
        else:
            kwargs_glm = {'family':family}


        r_max = 5
        df_r = estimate_r(Y, X, r_max, 
            kwargs_glm=kwargs_glm, 
            kwargs_ls_1={'alpha':0.1}, kwargs_es_1={'max_iters': 2000},
            kwargs_ls_2={'alpha':0.01}, kwargs_es_2={'max_iters': 2000,'patience':25},
            c1=0.01, intercept=1, offset=0, C=1e3, verbose=False
        )
        
        r_hat = int(df_r.iloc[df_r['JIC'].idxmin()]['r'])
        print(df_r, r_hat)

            
        lams = np.r_[
                np.linspace(0,0.1,11)[2:],
                np.linspace(0,1,11)[2:],
                np.linspace(1,2,11)[1:], 5., 10.
                ]
        df_res, df_res_adj = fit_gcate(
            Y, X, r_hat, d-1,
            kwargs_glm=kwargs_glm, 
            kwargs_ls_1={'alpha':0.1}, kwargs_es_1={'max_iters':2000},
            kwargs_ls_2={'alpha':0.01}, kwargs_es_2={'max_iters':2000,'patience':25},
            intercept=1, offset=0, num_d=1,
            c1=0.01, c2=lams, C=1e3
        )

        df_res['signals'] = np.tile(B_true[id_genes]!=0, len(lams)).astype(np.float32)
        df_res.to_csv(path_result+'gcate_raw_{}_{}_{}_{}.csv'.format(family,n,r,seed))

        df_res_adj['signals'] = np.tile(B_true[id_genes]!=0, len(lams)).astype(np.float32)
        df_res_adj.to_csv(path_result+'gcate_adj_{}_{}_{}_{}.csv'.format(family,n,r,seed))
