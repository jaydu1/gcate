from gcate.opt import *
from gcate.inference import *
from statsmodels.stats.multitest import multipletests



def fit_gcate(Y, X, r, i,
    kwargs_glm={}, kwargs_ls_1={}, kwargs_ls_2={}, kwargs_es={}, 
    # lam1=None, lam2=None, 
    c1=None, c2=None, ratio_infer=None, intercept=0, offset=0, C=1e4,
    load_model=False, save_model=False, path_model='gcate.npz'
):
    if not (X.ndim == 2 and Y.ndim == 2):
        raise ValueError("Input must have ndim of 2. Y.ndim: {}, X.ndim: {}.".format(Y.ndim, X.ndim))

    if np.sum(np.any(Y!=0., axis=0))<Y.shape[1]:
        raise ValueError("Y contains non-expressed features.")
                
    # preprocess
    scale_factor = np.ones((1, X.shape[1]))
#     scale_factor = np.sqrt(np.sum(X**2, 0, keepdims=True)) / np.sqrt(X.shape[0])
#     X[:,offset:] = X[:,offset:] / scale_factor[:,offset:]
    
    if ratio_infer is None:
        X_infer, Y_infer = X, Y
        n_infer = Y.shape[0]
    else:
        n_infer = int(X.shape[0] * ratio_infer)
        X_infer, Y_infer = X[-n_infer:,:], Y[-n_infer:,:]
        X, Y = X[:-n_infer,:], Y[:-n_infer,:]
        
    d = X.shape[1]
    p = Y.shape[1]
    n = Y.shape[0]
    
    if load_model:
        with open(path_model, 'rb') as f:
            tmp = np.load(f)
            A1, A2 = tmp['A1'], tmp['A2']
    else:
        A01, A02, info = alter_min(
            Y, r, X=X, P1=True, intercept=intercept, offset=offset, C=C,
            kwargs_glm=kwargs_glm, kwargs_ls=kwargs_ls_1, kwargs_es=kwargs_es)
        Q, _ = sp.linalg.qr(A02[:,d:], mode='economic')
        P_Gamma = np.identity(p) - Q @ Q.T


        c1 = 0.1 if c1 is None else c1
        lam1 = c1 * np.sqrt(np.log(p)/n)
        A1, A2, info = alter_min(
            Y, r, X=X, P2=P_Gamma, A=A01.copy(), B=A02.copy(), lam=lam1, intercept=intercept, offset=offset, C=C,
            kwargs_glm=kwargs_glm, kwargs_ls=kwargs_ls_2, kwargs_es=kwargs_es)
        
        
    if not load_model and save_model:
        model = {'init':{'A01':A01, 'A02':A02},
                 'c1-{}'.format(c1):{'lam':lam1, 'A1':A1, 'A2':A2}
                }
        with open(path_model, 'wb') as f:
            np.savez(f, model)
        
    c2 = 0.1 if c2 is None else c2
    c2 = np.array([c2]) if np.isscalar(c2) else c2
    lam2 = c2 * np.sqrt(np.log(p)/n_infer)
    if ratio_infer is None:
        B_de, se = debias(
            Y_infer, A1, A2, P_Gamma, d, i=i, kwargs_glm=kwargs_glm, intercept=intercept, offset=offset,
            lam=lam2)
    else:
        pass
    
    df_res_raw = pd.DataFrame()
    df_res = pd.DataFrame()    
    for j,lam in enumerate(lam2):
        z_scores = B_de[:,j] / se[:,j]
        pvals = sp.stats.norm.sf(np.abs(z_scores))*2

        _df = pd.DataFrame({
            'beta_hat':B_de[:,j]/scale_factor[0,i],
            'z_scores':z_scores, 
            'p_values':pvals,
            'q_values':multipletests(pvals, alpha=0.05, method='fdr_bh')[1]
        })
        _df['c2'] = c2[j]
        df_res_raw = pd.concat([df_res_raw, _df], axis=0)

        sigma = sp.stats.median_abs_deviation(
            z_scores, scale="normal", nan_policy='omit')
        z_scores = np.nan_to_num(z_scores, nan=np.nanmedian(z_scores))
        z_scores = (z_scores - np.median(z_scores)) / sigma
        pvals = sp.stats.norm.sf(np.abs(z_scores))*2

        _df = pd.DataFrame({
            'beta_hat':B_de[:,j]/scale_factor[0,i],
            'z_scores':z_scores, 
            'p_values':pvals,
            'q_values':multipletests(pvals, alpha=0.05, method='fdr_bh')[1]
        })
        _df['c2'] = c2[j]
        df_res = pd.concat([df_res, _df], axis=0)
        
    return df_res_raw, df_res
