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


path_data = 'data/ex5/'
os.makedirs(path_data, exist_ok=True)
path_result = 'result/ex5/'
os.makedirs(path_result, exist_ok=True)

intercept = 1
offset = 0


d = 2
r = 2

n_list = [100,250,500,750]
r_list = [2,10]

for p in [1500, 3000]:
    for n in [100, 250, 500, 750]:
        print(p, n, r)
        for seed in range(100):
            np.random.seed(seed)
            
            Gamma = sp.stats.ortho_group.rvs(dim=p)[:,:r]
            Lam = np.sqrt(p/2) * np.diag(2 - np.arange(r) / (r-1)) / 2
            Gamma = Gamma @ Lam 

            X = 2 * np.random.binomial(1, 0.5, size=(n,1)) - 1
            X = np.c_[np.ones((X.shape[0],1)), X]
            
            Q, _ = sp.linalg.qr(X[:,1:], mode='economic')
            PX_c = np.identity(n) - Q @ Q.T
            
            D = Q.T @ np.random.normal(size=(n,r))
            UD, SD, VDT = np.linalg.svd(D, full_matrices=False)
            if len(SD)>1:
                SD = (2 - np.arange(len(SD)) / (len(SD)-1)) / n**(3/2)
            else:
                SD = np.array([1. / n**(3/2)])
            D = UD @ np.diag(SD) @ VDT        
            
            W = PX_c @ np.random.normal(size=(n,r))
            UW, SW, VWT = np.linalg.svd(W, full_matrices=False)
            W = np.sqrt(n/2) *  UW @ np.diag(2 - np.arange(r) / (r-1)) @ VWT     
            Z = X[:,1:] @ D + W
            Z /= 2

            B = - np.random.uniform(0.25, 1., size=(p,d))
            B[:,0] = 0.5
            B[:,-1] = 0.2
            B[:,-1] *= (2 * np.random.binomial(1, 0.5, size=(p,)) - 1)
            B[np.random.rand(p) > 0.05, -1] = 0.

            Theta = X @ B.T + Z @ Gamma.T
            lib_size = np.sum(np.exp(Theta), axis=1, keepdims=False)
            scale_factor = np.ones(n)

            X = np.c_[np.log(scale_factor), X]
            B = np.c_[np.ones(p), B]            
            Theta = X @ B.T + Z @ Gamma.T
            
            
            Y = np.random.poisson(np.exp(Theta))
                
            file_name = path_data + 'p_{}_n_{}_r_{}_seed_{}.npz'.format(p, n,r,seed)
            with open(file_name, 'wb') as f:
                np.savez(f, X=X, B=B, Y=Y, Gamma=Gamma, Z=Z, W=W, D=D)


            with open(file_name, 'rb') as f:
                tmp = np.load(f)
                X = tmp['X']
                X = X[:,1:]
                B = tmp['B']
                Y = tmp['Y'].astype(float)
                n, d = X.shape
            print(X.shape, Y.shape)
            id_genes = np.any(Y, axis=0)
            p_all = Y.shape[1]
            Y = Y[:,id_genes]

            c1=0.02
            lam1 = c1 * np.sqrt(np.log(p_all)/n)

            X_infer, Y_infer = X, Y
            n_infer = Y.shape[0]
            num_missing = np.zeros(2)
            
            r_hat = 2

            _, _, P_Gamma, A1, A2 = estimate(Y, X, r_hat, 
                intercept=intercept, offset=offset,
                num_d=1, C=None, num_missing = np.zeros(2),
                lam1=lam1, 
                kwargs_glm=kwargs_glm, 
                
                kwargs_ls_1={},
                kwargs_ls_2={'alpha':0.01},
                kwargs_es_1={'max_iters': 600},
                kwargs_es_2={'max_iters': 600},
                )

            df_res_raw = pd.DataFrame(
                {
                    'B_true': B[:,-1][id_genes],
                    'B_hat': A2[:,d-1]
                }
            )
            df_res_raw.to_csv(path_result+'gcate_raw_{}_{}_{}_{}_{}.csv'.format(family,p,n,r,seed))

