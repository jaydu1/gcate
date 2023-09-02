import numpy as np
import statsmodels.api as sm



def init_inv_link(Y, family, disp):
    if family=='gaussian':
        val = Y/disp
    elif family=='poisson':
        val = np.log1p(Y)
    elif family=='negative_binomial':
        val = np.log1p(Y/disp / (Y/disp + 1))
    elif family=='binomial':
        eps = (np.mean(Y, axis=0) + np.mean(Y, axis=1)) / 2 
        val = np.log((Y + eps)/(disp - Y + eps))
    else:
        raise ValueError('Family not recognized')
    return val


def glm(y, X, offset, family):
    try:
        b = sm.GLM(y, X, offset=offset, family=family).fit().params
        return b
    except:
        pass
    
    try:
        b = sm.GLM(y, X, offset=offset, family=family).fit(method='bfgs').params
        return b
    except:
        pass
    
    alpha = 1e-6
    b = sm.GLM(y, X, offset=offset, family=family).fit_regularized(alpha=alpha, L1_wt=0.).params
    return b


def fit_glm(Y, X, offset, family, disp):
    p = Y.shape[1]
    d = X.shape[1]
    B = np.zeros((p,d))
    
    if family=='gaussian':
        sm_family = sm.families.Gaussian()
    elif family=='poisson':
        sm_family = sm.families.Poisson()
    elif family=='negative_binomial':
        pass
    elif family=='binomial':
        sm_family = sm.families.Binomial()
    else:
        raise ValueError('Family not recognized')
        
    for j in range(p):
        if family=='negative_binomial':
#             sm_family=sm.families.NegativeBinomial(
#                 link=sm.genmod.families.links.NegativeBinomial(alpha=1./disp[j]), alpha=1./disp[j])
            assert offset is None
            sm_family = sm.families.Poisson()
            B[j, :] = glm(Y[:,j], X, offset, sm_family)
            Yp = 1 - 1 / (np.exp(X @ B[j, :])/disp[j] + 1.)
            B[j, :] = glm(Yp, X, offset, sm_family)
            
            tmp = np.max(X @ B[j, :].T)
            if tmp>-1e-4:
                B[j, 0] = B[j, 0] - tmp - 1e-4
        else:
            B[j, :] = glm(Y[:,j], X, offset, sm_family)

#         # this does not guarantee theta<0
#         if family=='negative_binomial' and np.max(np.abs(B[j,:])) > 1e1:
#             if offset is not None:
#                 B[j,:] = np.r_[np.minimum(- np.max(offset), 0.) - 1e-2, np.zeros(d-1)]
#             else:
#                 B[j,:] = np.r_[- 1e-2, np.zeros(d-1)]
#                 raise ValueError("Optimization infeasible for gene {} during " \
#                 "initialization with GLM. Consider remove it.".format(j))

#         B_p = np.zeros((p,d))
#         sm_family = sm.families.Poisson()
#         for j in range(p):
#             B_p[j, :] = glm(Y[:,j], X, offset, sm_family)
#         idx = np.max(np.abs(B), axis=-1) > 1e2
#         B[idx,:] = B_p[idx,:]
    return B


def estimate_disp(Y, X):
    B_glm = []
    for j in range(Y.shape[1]):
        mod = sm.GLM(Y[:,j], X, family=sm.families.Poisson()).fit()
        B_glm.append(mod.params)
    B_glm = np.array(B_glm)

    mu_glm = np.mean(np.exp(X @ B_glm.T), axis=0)
    disp_glm = (np.mean((Y - mu_glm[None,:])**2, axis=0) - mu_glm) / mu_glm**2
    disp_glm = 1./np.clip(disp_glm, 0.01, 100.)
    return disp_glm[None,:]