import os
import sys
os.environ["OMP_NUM_THREADS"] = "11"
os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "11" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "11" # export NUMEXPR_NUM_THREADS=6
os.environ["NUMBA_CACHE_DIR"]='/tmp/numba_cache'


import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats

import anndata as ad
import scanpy as sc
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


########################################################################
#
# Load raw data
#
########################################################################
adata = sc.read_h5ad('data/lupus/GSE174188_CLUES1_adjusted.h5ad')

# only keep ential info to save memory
X = adata.raw.X
obs = pd.DataFrame()
obs['SLE_status'] = adata.obs['SLE_status'].tolist()
obs['ind_cov'] = adata.obs['ind_cov'].tolist()
obs['batch_cov'] = adata.obs['batch_cov'].tolist()
obs['pop_cov'] = adata.obs['pop_cov'].tolist()
obs['cg_cov'] = adata.obs['cg_cov'].tolist()
obs['ct_cov'] = adata.obs['ct_cov'].tolist()
obs['Age'] = adata.obs['Age'].tolist()
obs['Sex'] = adata.obs['Sex'].tolist()
obs['Processing_Cohort'] = adata.obs['Processing_Cohort'].tolist()
obs['L3'] = adata.obs['L3'].tolist()
var_names = adata.raw.var_names.tolist()
var = pd.DataFrame(index=var_names)
cdata = ad.AnnData(X, obs=obs, var=var, dtype='int32')
cdata.raw = cdata
del adata
cdata.write_h5ad("data/lupus/raw.h5ad")



########################################################################
#
# Preprocess raw data
#
########################################################################
adata = sc.read_h5ad('data/lupus/raw.h5ad')
print(adata)
# AnnData object with n_obs × n_vars = 1263676 × 32738
#     obs: 'SLE_status', 'ind_cov', 'batch_cov', 'pop_cov', 'cg_cov', 'ct_cov', 'Age', 'Sex', 'Processing_Cohort', 'L3'


adata.obs.groupby(['ind_cov']).count().shape[0]
# 261 individuals
adata = adata[adata.obs['pop_cov'].isin(['Asian', 'European']),:]
# 256 individuals
adata.obs['cg_pop_cov'] = adata.obs['cg_cov'].astype(str) + '-' + adata.obs['pop_cov'].astype(str)
adata.obs['cg_pop_cov'] = adata.obs['cg_pop_cov'].astype("category")

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.obs.groupby(['ind_cov']).count().shape[0]
# 256


########################################################################
#
# Save data with 2000 highly variable genes
#
########################################################################

col_covariate = ['Sex','Processing_Cohort','pop_cov','SLE_status']
path_data = 'data/lupus/'

for celltype in adata.obs['cg_cov'].unique()[:5]:
    idx = adata.obs['cg_cov'] == celltype
    _adata = adata[idx].copy()
    
    sc.pp.highly_variable_genes(_adata, n_top_genes=2000)
    gene_names = _adata.var.index[_adata.var['highly_variable']]

    count = _adata.raw.X[:,np.in1d(_adata.raw.var_names.tolist(), gene_names)]
    
    _df = _adata.obs
    uni_ind = np.unique(_df['ind_cov'])
    
    print(celltype)
    print('\t shape:', _adata.raw.X.shape, len(_adata.raw.var_names))
    print('\t ind:', len(uni_ind))    
    
    arr = []
    meta = []
    for ind in uni_ind:
        arr.append(count[_df['ind_cov']==ind].sum(axis=0))
        meta.append(_df[_df['ind_cov']==ind].iloc[:1,:])

    Y = np.concatenate(arr, axis=0)
    meta = pd.concat(meta, axis=0).reset_index(drop=True)
    
    enc = OrdinalEncoder()    
    X = enc.fit_transform(meta[['Sex','pop_cov','SLE_status']])
        
    enc = OneHotEncoder(drop='first')
    _X = enc.fit_transform(meta[['Processing_Cohort']])
    
    X = np.c_[X[:,:-1], _X.toarray(), X[:,-1:]]
    
    file_name = path_data + 'data_lupus_{}.npz'.format(celltype)
    with open(file_name, 'wb') as f:
        np.savez(f, X=X, Y=Y, gene_names=gene_names)

# T4
# 	 shape: (375885, 32738) 32738
# 	 ind: 256
# cM
# 	 shape: (302649, 32738) 32738
# 	 ind: 256
# B
# 	 shape: (149861, 32738) 32738
# 	 ind: 254
# T8
# 	 shape: (242677, 32738) 32738
# 	 ind: 256
# NK
# 	 shape: (90190, 32738) 32738
# 	 ind: 256    




########################################################################
#
# Save data with 250 highly variable genes
#
########################################################################
n_hvg = 250
for celltype in adata.obs['cg_cov'].unique()[:5]:
    idx = adata.obs['cg_cov'] == celltype
    _adata = adata[idx].copy()
    
    sc.pp.highly_variable_genes(_adata, n_top_genes=n_hvg)
    gene_names = _adata.var.index[_adata.var['highly_variable']]
    
    count = _adata.raw.X[:,np.in1d(_adata.raw.var_names.tolist(), gene_names)]
    
    _df = _adata.obs
    uni_ind = np.unique(_df['ind_cov'])
    
    print(celltype)
    print('\t shape:', _adata.raw.X.shape, len(_adata.raw.var_names))
    print('\t ind:', len(uni_ind))    
    
    arr = []
    meta = []
    for ind in uni_ind:
        arr.append(count[_df['ind_cov']==ind].sum(axis=0))
        meta.append(_df[_df['ind_cov']==ind].iloc[:1,:])

    Y = np.concatenate(arr, axis=0)
    meta = pd.concat(meta, axis=0).reset_index(drop=True)
    
    enc = OrdinalEncoder()    
    X = enc.fit_transform(meta[['Sex','pop_cov','SLE_status']])
        
    enc = OneHotEncoder(drop='first')
    _X = enc.fit_transform(meta[['Processing_Cohort']])
    
    X = np.c_[X[:,:-1], _X.toarray(), X[:,-1:]]
    
    file_name = path_data + 'data_lupus_{}_{}.npz'.format(celltype,n_hvg)
    with open(file_name, 'wb') as f:
        np.savez(f, X=X, Y=Y, gene_names=gene_names)

# T4
# 	 shape: (375885, 32738) 32738
# 	 ind: 256
# cM
# 	 shape: (302649, 32738) 32738
# 	 ind: 256
# B
# 	 shape: (149861, 32738) 32738
# 	 ind: 254
# T8
# 	 shape: (242677, 32738) 32738
# 	 ind: 256
# NK
# 	 shape: (90190, 32738) 32738
# 	 ind: 256