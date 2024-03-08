import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
from tqdm import tqdm


p = 3000
d = 2

path_data = 'data/ex2/poisson/'

os.makedirs(path_data, exist_ok=True)

for n in [100,250]:
    for r in [2,10]:
        print(n,r)
        for seed in tqdm(range(100)):
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
            B[:,0] = 0.5 if n==250 else .5
            B[:,-1] = 0.2 if n==100 else 0.2
            B[:,-1] *= (2 * np.random.binomial(1, 0.5, size=(p,)) - 1)
            B[np.random.rand(p) > 0.05, -1] = 0.

            Theta = X @ B.T + Z @ Gamma.T
            lib_size = np.sum(np.exp(Theta), axis=1, keepdims=False)
            scale_factor = np.ones(n)

            X = np.c_[np.log(scale_factor), X]
            B = np.c_[np.ones(p), B]            
            Theta = X @ B.T + Z @ Gamma.T
            
            
            Y = np.random.poisson(np.exp(Theta))
                
            file_name = path_data + 'n_{}_r_{}_seed_{}.npz'.format(n,r,seed)
            with open(file_name, 'wb') as f:
                np.savez(f, X=X, B=B, Y=Y, Gamma=Gamma, Z=Z, W=W, D=D)
            