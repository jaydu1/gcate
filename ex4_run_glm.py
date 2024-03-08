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

import dill


path_data = 'data/lupus/'
path_result = 'result/lupus/'
os.makedirs(path_result, exist_ok=True)

cell_types = ['T4', 'cM', 'B', 'T8', 'NK']
for cell_type in cell_types:
    print(cell_type)
    
    file_name = path_data + 'data_lupus_{}.npz'.format(cell_type)
    with open(file_name, 'rb') as f:
        tmp = np.load(f)
        X = tmp['X']
        Y = tmp['Y'].astype(float)
        X = X * 2 - 1
        X = np.c_[np.ones(X.shape[0]), np.log(np.sum(Y, axis=1)), X]
        X = X / np.sqrt(np.sum(X**2, 0, keepdims=True)) * np.sqrt(X.shape[0])
    X_sub = np.c_[X[:,:2],X[:,-1:]]
    id_genes = np.any(Y, axis=0) & (np.sum(Y>0, axis=0)>=10)
    p_all = Y.shape[1]
    Y = Y[:,id_genes]
    print(X.shape, X_sub.shape, Y.shape)
    
    p = Y.shape[1]
    d = X.shape[1]

    family = 'poisson'
    tvals = []
    pvals = []
    B_glm = []
    se_glm = []
    # d = 1
    for j in range(Y.shape[1]):
        try:
            mod = sm.GLM(Y[:,j], X, family=sm.families.Poisson()).fit()
            B_glm.append(mod.params)
            se_glm.append(mod.bse)
            tvals.append(mod.tvalues[-1])
            pvals.append(mod.pvalues[-1])
        except:
            print(j)
            B_glm.append(np.full((X.shape[1],), np.nan))
            tvals.append(np.nan)
            pvals.append(np.nan)
    B_glm = np.array(B_glm)
    se_glm = np.array(se_glm)
    tvals = np.array(tvals)
    pvals = np.array(pvals)
    _df_glm = pd.DataFrame({
        'beta_hat':B_glm[:,-1],
        'z_scores':tvals, 
        'p_values':pvals,
        'q_values':multipletests(pvals, alpha=0.05, method='fdr_bh')[1]
    })
    _df_glm.to_csv(path_result+'glm_{}_{}.csv'.format(family, cell_type))
    


    mu_glm = np.mean(np.exp(X @ B_glm.T), axis=0)
    disp_glm = (np.mean((Y - mu_glm[None,:])**2, axis=0) - mu_glm) / mu_glm**2
    disp_glm = np.clip(disp_glm, 0.01, 100.)
    
    family = 'negative_binomial'
    tvals_glm_nb = []
    pvals_glm_nb = []
    mu_glm_nb = []
    B_glm_nb = []
    d = X[:,:].shape[1]
    for j in range(Y.shape[1]):
        try:
            mod = sm.GLM(Y[:,j], X, family=sm.families.NegativeBinomial(alpha=disp_glm[j])).fit()
            B_glm_nb.append(mod.params)
            mu_glm_nb.append(mod.predict(X[:,:]))
            tvals_glm_nb.append(mod.tvalues[-1])
            pvals_glm_nb.append(mod.pvalues[-1])
        except:
            print(j)
            B_glm_nb.append(B_glm[j,:])
            tvals_glm_nb.append(tvals[j])
            pvals_glm_nb.append(pvals[j])

    B_glm_nb = np.array(B_glm_nb)
    tvals_glm_nb = np.array(tvals_glm_nb)
    pvals_glm_nb = np.array(pvals_glm_nb)
    _df_glm_nb = pd.DataFrame({
        'beta_hat':B_glm_nb[:,-1],
        'z_scores':tvals_glm_nb, 
        'p_values':pvals_glm_nb,
        'q_values':multipletests(pvals_glm_nb, alpha=0.05, method='fdr_bh')[1]
    })
    _df_glm_nb.to_csv(path_result+'glm_{}_{}.csv'.format(family, cell_type))
    
    
    
    family = 'poisson'
    tvals_sub = []
    pvals_sub = []
    B_glm_sub = []
    se_glm_sub = []
    # d = 1
    for j in range(Y.shape[1]):
        mod = sm.GLM(Y[:,j], X_sub, family=sm.families.Poisson()).fit()
        B_glm_sub.append(mod.params)
        se_glm_sub.append(mod.bse)
        tvals_sub.append(mod.tvalues[-1])
        pvals_sub.append(mod.pvalues[-1])
    B_glm_sub = np.array(B_glm_sub)
    se_glm_sub = np.array(se_glm_sub)
    tvals_sub = np.array(tvals_sub)
    pvals_sub = np.array(pvals_sub)
    _df_glm_sub = pd.DataFrame({
        'beta_hat':B_glm_sub[:,-1],
        'z_scores':tvals_sub, 
        'p_values':pvals_sub,
        'q_values':multipletests(pvals_sub, alpha=0.05, method='fdr_bh')[1]
    })
    _df_glm_sub.to_csv(path_result+'glm_{}_{}_sub.csv'.format(family, cell_type))
    
    
    mu_glm_sub = np.mean(np.exp(X_sub @ B_glm_sub.T), axis=0)    
    disp_glm_sub = (np.mean((Y - mu_glm_sub[None,:])**2, axis=0) - mu_glm_sub) / mu_glm_sub**2
    disp_glm_sub = np.clip(disp_glm_sub, 0.01, 100.)
    
    family = 'negative_binomial'
    tvals_glm_nb_sub = []
    pvals_glm_nb_sub = []
    mu_glm_nb_sub = []
    B_glm_nb_sub = []
    d = X_sub[:,:].shape[1]
    for j in range(Y.shape[1]):
        try:
            mod = sm.GLM(Y[:,j], X_sub, family=sm.families.NegativeBinomial(alpha=disp_glm_sub[j])).fit()
            B_glm_nb_sub.append(mod.params)
            mu_glm_nb_sub.append(mod.predict(X_sub))
            tvals_glm_nb_sub.append(mod.tvalues[-1])
            pvals_glm_nb_sub.append(mod.pvalues[-1])
        except:
            print(j)
            B_glm_nb_sub.append(B_glm_sub[j,:])
            tvals_glm_nb_sub.append(tvals_sub[j])
            pvals_glm_nb_sub.append(pvals_sub[j])

    B_glm_nb_sub = np.array(B_glm_nb_sub)
    tvals_glm_nb_sub = np.array(tvals_glm_nb_sub)
    pvals_glm_nb_sub = np.array(pvals_glm_nb_sub)
    _df_glm_nb_sub = pd.DataFrame({
        'beta_hat':B_glm_nb_sub[:,-1],
        'z_scores':tvals_glm_nb_sub, 
        'p_values':pvals_glm_nb_sub,
        'q_values':multipletests(pvals_glm_nb_sub, alpha=0.05, method='fdr_bh')[1]
    })
    _df_glm_nb_sub.to_csv(path_result+'glm_{}_{}_sub.csv'.format(family, cell_type))
    
    
    with open('result/lupus/dill_{}.pkl'.format(cell_type), "wb") as file_object:
        dill.dump([
            B_glm,tvals,pvals,_df_glm,disp_glm,
            B_glm_nb,tvals_glm_nb,pvals_glm_nb,_df_glm_nb,
            B_glm_sub,tvals_sub,pvals_sub,_df_glm_sub,disp_glm_sub,
            B_glm_nb_sub,tvals_glm_nb_sub,pvals_glm_nb_sub,_df_glm_nb_sub
        ], file_object)   