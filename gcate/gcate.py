from gcate.opt import *
from gcate.inference import *
from statsmodels.stats.multitest import multipletests



def fit_gcate(Y, X, r, i,
    kwargs_glm={}, kwargs_ls_1={}, kwargs_ls_2={}, kwargs_es_1={}, kwargs_es_2={},
    c1=None, c2=None, ratio_infer=None, intercept=0, offset=0, num_d=1, C=1e5, w_type=0
):
    '''
    Parameters
    ----------
    Y : array-like, shape (n, p)
        The response variable.
    X : array-like, shape (n, d)
        The covariate matrix.
    r : int
        The number of unmeasured confounders.
    i : int
        The index of the feature to be tested. Integer from 0 to d-1.
    kwargs_glm : dict
        Keyword arguments for the GLM solver.
    kwargs_ls_1 : dict
        Keyword arguments for the line search solver in the first phrase.
    kwargs_ls_2 : dict
        Keyword arguments for the line search solver in the second phrase.
    kwargs_es_1 : dict
        Keyword arguments for the early stopper in the first phrase.
    kwargs_es_2 : dict
        Keyword arguments for the early stopper in the second phrase.
    c1 : float
        The regularization constant in the first phrase. Default is 0.1.
    c2 : float or array-like, shape (n_c2,)
        The regularization constant in the second phrase. Default is 0.1.
    ratio_infer : float
        The ratio of samples splitted for inference. Default is None.
    intercept : int
        Whether to include intercept in the model. Default is 0.
    offset : int
        Whether to use offset in the model. Default is 0.
    num_d : int
        The number of covariates to be regularized. Assume the last num_d covariates are to be regularized. Default is 1.
    C : float
        The constant for maximum l2 norm of the coefficients. Default is 1e5.
    w_type : int
        The type of weights used for inference. Default is 0.
    '''
    if not (X.ndim == 2 and Y.ndim == 2):
        raise ValueError("Input must have ndim of 2. Y.ndim: {}, X.ndim: {}.".format(Y.ndim, X.ndim))

    if np.sum(np.any(Y!=0., axis=0))<Y.shape[1]:
        raise ValueError("Y contains non-expressed features.")
                
    # preprocess
    scale_factor = np.ones((1, X.shape[1]))
        
    d = X.shape[1]
    p = Y.shape[1]
    n = Y.shape[0]

    c1 = 0.1 if c1 is None else c1
    lam1 = c1 * np.sqrt(np.log(p)/n)

    c2 = 0.1 if c2 is None else c2
    c2 = np.array([c2]) if np.isscalar(c2) else c2
    lam2 = c2 * np.sqrt(np.log(p)/p)

    if ratio_infer is None:
        X_infer, Y_infer = X, Y
        n_infer = Y.shape[0]
        num_missing = np.zeros(2)

        _, _, P_Gamma, A1, A2 = estimate(Y, X, r, intercept, offset, num_d, C, num_missing,
            lam1, kwargs_glm, kwargs_ls_1, kwargs_es_1, kwargs_ls_2, kwargs_es_2)

        B_de, se = debias(
            Y_infer, A1, A2, P_Gamma, d, i=i, kwargs_glm=kwargs_glm, intercept=intercept, offset=offset,
            lam=lam2, w_type=w_type)
    else:
        n_infer = int(X.shape[0] * ratio_infer)
        p_fold = int(Y.shape[1]//2)
        num_missing = np.array([n_infer, p_fold], dtype=int)

        B_de, se = [], []
        for fold in range(2):
            if fold==0:
                Y_infer = np.c_[Y[:,-num_missing[1]:], Y[:,:-num_missing[1]]]
                p_infer = p - num_missing[1]
            else:
                Y_infer = np.c_[Y[:,:-num_missing[1]], Y[:,-num_missing[1]:]]
                p_infer = num_missing[1]
            A01, A02, _, A1, A2 = estimate(Y_infer, X, r, intercept, offset, num_d, C, num_missing,
                lam1, kwargs_glm, kwargs_ls_1, kwargs_es_1, kwargs_ls_2, kwargs_es_2)
            Q, _ = sp.linalg.qr(A02[-p_infer:,d:], mode='economic')
            P_Gamma = np.identity(p_infer) - Q @ Q.T
            _B_de, _se = debias(
                Y_infer[-n_infer:,-p_infer:], A1[-n_infer:,:], A2[-p_infer:,], P_Gamma, d, i=i, kwargs_glm=kwargs_glm, intercept=intercept, offset=offset,
                lam=lam2, w_type=w_type)
            B_de.append(_B_de)
            se.append(_se)
        B_de, se = np.concatenate(B_de, axis=0), np.concatenate(se, axis=0)               

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


def estimate(Y, X, r, intercept, offset, num_d, C, num_missing,
    lam1, kwargs_glm, kwargs_ls_1, kwargs_es_1, kwargs_ls_2, kwargs_es_2):
    d = X.shape[1]
    p = Y.shape[1]

    A01, A02, info = alter_min(
        Y, r, X=X, P1=True, intercept=intercept, offset=offset, C=C, num_missing=num_missing,
        kwargs_glm=kwargs_glm, kwargs_ls=kwargs_ls_1, kwargs_es=kwargs_es_1)
    Q, _ = sp.linalg.qr(A02[:,d:], mode='economic')
    P_Gamma = np.identity(p) - Q @ Q.T

    A1, A2, info = alter_min(
        Y, r, X=X, P2=P_Gamma, A=A01.copy(), B=A02.copy(), lam=lam1, 
        intercept=intercept, offset=offset, num_d=num_d, C=C, num_missing=num_missing,
        kwargs_glm=kwargs_glm, kwargs_ls=kwargs_ls_2, kwargs_es=kwargs_es_2)
    return A01, A02, P_Gamma, A1, A2