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

import dill


path_data = 'data/lupus/'
path_result = 'result/lupus/'
os.makedirs(path_result, exist_ok=True)

cell_types =  ['T4', 'cM', 'B', 'T8', 'NK']
for ic, cell_type in enumerate(cell_types[:1]):
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

    
    with open('result/lupus/dill_{}.pkl'.format(cell_type), "rb") as file_object:
        [
            B_glm,tvals,pvals,_df_glm,disp_glm,
            B_glm_nb,tvals_glm_nb,pvals_glm_nb,_df_glm_nb,
            B_glm_sub,tvals_sub,pvals_sub,_df_glm_sub,disp_glm_sub,
            B_glm_nb_sub,tvals_glm_nb_sub,pvals_glm_nb_sub,_df_glm_nb_sub
        ] = dill.load(file_object)
        
        
    kwargs_glm = {
        'family':'negative_binomial',
        'nuisance':1/disp_glm_sub[None,:]
    }

    params = {'kwargs_glm':kwargs_glm, 
        'kwargs_ls_1':{'alpha':0.1}, 'kwargs_es_1':{'max_iters': 2000},
        'kwargs_ls_2':{'alpha':0.01}, 'kwargs_es_2':{'max_iters': 2000, 'patience':25},
        'c1':0.01, 'intercept':1, 'offset':0, 'C':1e3}

    n_infer = int(X_sub.shape[0])
    X_test = X_sub.copy()
    Y_test = Y.copy()

    n, d = X_sub.shape
    _, p = Y.shape

    # Select the number of factors r
    # r_list = [7, 6, 6, 6, 5]
    # r_hat = r_list[ic]

    r_max = 10
    df_r = estimate_r(Y, X_sub, r_max, 0.25, verbose=False, **params)

    r_hat = int(df_r.iloc[df_r['JIC'].idxmin()]['r'])
    print(df_r, r_hat)

    c2 = np.r_[
            np.linspace(0,0.01,11)[1:],
            np.linspace(0,1,51)[1:]
            ]

    df_res, _ = fit_gcate(
        Y, X_sub, r_hat, d-1, c2=c2, **params
    )
    
    df_res.to_csv(path_result+'gcate_{}.csv'.format(cell_type), index=False)


    ############################################################################
    #
    # Sensitivity analysis in r
    #
    ############################################################################

    if cell_type == 'T4':

        for r in range(1,11):
        
            A01, A02, info = alter_min(Y, r, X=X_sub, P1=True, kwargs_ls={'alpha':.1}, kwargs_es={'max_iters': 2000},
                                kwargs_glm=kwargs_glm, C=1e3
                                )
            Q, _ = sp.linalg.qr(A02[:,d:], mode='economic')
            P_Gamma = np.identity(p) - Q @ Q.T

            A1, A2, info = alter_min(Y, r, X=X_sub, P2=P_Gamma, A=A01.copy(), B=A02.copy(),
                                    lam=.01*np.sqrt(np.log(p)/n), kwargs_ls={'alpha':.01},
                                    kwargs_es={'max_iters': 2000, 'patience':25
                                            },
                                    kwargs_glm=kwargs_glm, C=1e3
                                    )
            B = A2[:, :d]
            print(np.mean(np.abs(B[:,-1])<1e-3))

            B_de, se = debias(
                Y, A1, A2, P_Gamma, 
                d, i=d-1, lam=0.0883057516886606, kwargs_glm=kwargs_glm
            )

            with open('result/lupus/dill_{}_gcate_r_{}.pkl'.format(cell_type, r), "wb") as file_object:
                dill.dump([
                    A01, A02, A1, A2, P_Gamma, B_de, se
                ], file_object)    

    