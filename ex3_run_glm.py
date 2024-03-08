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


family = 'negative_binomial'
kwargs_glm = {'family':family}

path_data = 'data/ex3/'
path_result = 'result/ex3/'
os.makedirs(path_result, exist_ok=True)

n_list = [100,200]
n = n_list[int(sys.argv[1])]
r = 3
d = 3

for seed in range(100):
    print('n:{}, seed:{}'.format(n,seed))
    
    Y = pd.read_csv(path_data+'{}_{}_Y.csv'.format(n,seed)).values.astype(np.float32)
    id_genes = np.any(Y, axis=0) & (np.sum(Y>0, axis=0)>=10)
    p_all = Y.shape[1]
    Y = Y[:,id_genes]

    B_true = pd.read_csv(path_data+'{}_{}_B.csv'.format(n,seed)).values[:,0]
    X = pd.read_csv(path_data+'{}_{}_X.csv'.format(n,seed)).values
    Z = pd.read_csv(path_data+'{}_{}_Z.csv'.format(n,seed)).values
    
    for _type in ['oracle']:
        if _type == 'oracle':
            _X = np.c_[X,Z]
            method = 'glm_oracle'
        else:
            _X = X
            method = 'glm'
            
        family = 'poisson'
        p = Y.shape[1]
        tvals = []
        pvals = []
        B = []
        for j in range(p):
            mod = sm.GLM(Y[:,j], _X, family=sm.families.Poisson()).fit()
            B.append(mod.params)
            tvals.append(mod.tvalues[d-1])
            pvals.append(mod.pvalues[d-1])
        B = np.array(B)
        tvals = np.array(tvals)
        pvals = np.array(pvals)
        df_res = pd.DataFrame({
            'signals':(B_true[id_genes]!=0).astype(np.float32),
            'beta_hat':B[:,d-1],
            'z_scores':tvals, 
            'p_values':pvals,
            'q_values':multipletests(pvals, alpha=0.05, method='fdr_bh')[1]
        })

        df_res.to_csv(path_result+'{}_{}_{}_{}_{}.csv'.format(method,family,n,r,seed))



        family = 'negative_binomial'
        mu_glm = np.mean(np.exp(_X @ B.T), axis=0)
        disp_glm = (np.mean((Y - mu_glm[None,:])**2, axis=0) - mu_glm) / mu_glm**2
        disp_glm = np.clip(disp_glm, 0.01, 100.)

        tvals = []
        pvals = []
        B_glm_nb = []
        for j in range(Y.shape[1]):
            try:
                mod = sm.GLM(Y[:,j], _X, family=sm.families.NegativeBinomial(alpha=disp_glm[j])).fit()
                B_glm_nb.append(mod.params)
                tvals.append(mod.tvalues[d-1])
                pvals.append(mod.pvalues[d-1])
            except:
                print(seed,j)
                B_glm_nb.append(np.zeros((d,)))
                tvals.append(0.)
                pvals.append(1.)

        B_glm_nb = np.array(B_glm_nb)
        tvals = np.array(tvals)
        pvals = np.array(pvals)
        df_res = pd.DataFrame({
            'signals':(B_true[id_genes]).astype(np.float32),
            'beta_hat':B_glm_nb[:,d-1],
            'z_scores':tvals, 
            'p_values':pvals,
            'q_values':multipletests(pvals, alpha=0.05, method='fdr_bh')[1]
        })

        df_res.to_csv(path_result+'{}_{}_{}_{}_{}.csv'.format(method,family,n,r,seed))
