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


family = 'poisson'
kwargs_glm = {'family':family}

data_family = 'poisson'
path_data = 'data/ex2/{}_2/'.format(data_family)
path_result = 'result/ex2/{}_2/'.format(data_family)
os.makedirs(path_result, exist_ok=True)


p = 3000
d = 2

n_list = [100,250]
r_list = [2,10]
n = n_list[int(sys.argv[1])]
r = r_list[int(sys.argv[2])]

glm_family = sm.families.Poisson() if family=='poisson' else sm.families.NegativeBinomial()
for seed in range(100):
    print('n:{}, r:{}, seed:{}'.format(n,r,seed))
    
    file_name = path_data + 'n_{}_r_{}_seed_{}.npz'.format(n,r,seed)
    with open(file_name, 'rb') as f:
        tmp = np.load(f)
        X = tmp['X']
        B_true = tmp['B']
        Z = tmp['Z']
        Y = tmp['Y'].astype(float)

    print(X.shape, Y.shape)
    id_genes = np.any(Y, axis=0)
    p_all = Y.shape[1]
    Y = Y[:,id_genes]

    p = Y.shape[1]
    tvals = []
    pvals = []
    B = []
    for j in range(p):
        mod = sm.GLM(Y[:,j], X[:,1:], offset=X[:,0], family=glm_family).fit()
        B.append(mod.params[d-1])
        tvals.append(mod.tvalues[d-1])
        pvals.append(mod.pvalues[d-1])
    B = np.array(B)
    tvals = np.array(tvals)
    pvals = np.array(pvals)
    df_res = pd.DataFrame({
        'signals':(B_true[:,-1][id_genes]!=0).astype(np.float32),
        'beta_hat':B,
        'z_scores':tvals, 
        'p_values':pvals,
        'q_values':multipletests(pvals, alpha=0.05, method='fdr_bh')[1]
    })

    df_res.to_csv(path_result+'glm_{}_{}_{}_{}.csv'.format(family,n,r,seed))

    
    p = Y.shape[1]
    tvals = []
    pvals = []
    B = []
    for j in range(p):
        mod = sm.GLM(Y[:,j], np.c_[X[:,1:],Z], offset=X[:,0], family=glm_family).fit()
        B.append(mod.params[d-1])
        tvals.append(mod.tvalues[d-1])
        pvals.append(mod.pvalues[d-1])
    B = np.array(B)
    tvals = np.array(tvals)
    pvals = np.array(pvals)
    df_res = pd.DataFrame({
        'signals':(B_true[:,-1][id_genes]!=0).astype(np.float32),
        'beta_hat':B,
        'z_scores':tvals, 
        'p_values':pvals,
        'q_values':multipletests(pvals, alpha=0.05, method='fdr_bh')[1]
    })

    df_res.to_csv(path_result+'glm_oracle_{}_{}_{}_{}.csv'.format(family,n,r,seed))