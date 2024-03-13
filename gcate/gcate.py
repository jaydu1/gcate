from gcate.opt import *
from gcate.inference import *
from gcate.likelihood import *
from statsmodels.stats.multitest import multipletests



def fit_gcate(Y, X, r, i,
    kwargs_glm={}, kwargs_ls_1={}, kwargs_ls_2={}, kwargs_es_1={}, kwargs_es_2={},
    c1=None, c2=None, ratio_infer=None, intercept=0, offset=0, num_d=1, C=None, w_type=0, **kwargs
):
    '''
    Fit GCATE model.
    It consists of three optimization problems:
    (1) Joint estimation of the marginal effects X@F^T and uncorrelated latent effects W@Gamma^T.
    (2) Decouple the primary effects X@B^T and Z@Gamma^T, with l1 regularization on B.
    (3) Debias the primary effects B for inference.

    Parameters
    ----------
    Y : array-like, shape (n, p)
        Response matrix.
    X : array-like, shape (n, d)
        Observed covariate matrix.
    r : int
        Number of latent variables.
    i : int
        Index of the feature of interest.
    kwargs_glm : dict
        Keyword arguments for the GLM.
    kwargs_ls_1 : dict
        Keyword arguments of the line search algorithm for the first optimization problem.
    kwargs_ls_2 : dict
        Keyword arguments of the line search algorithm for the second optimization problem.
    kwargs_es_1 : dict
        Keyword arguments of the early stopping monitor for the first optimization problem.
    kwargs_es_2 : dict
        Keyword arguments of the early stopping monitor for the second optimization problem.
    c1 : float
        Regularization parameter for the second optimization problem.
    c2 : float or array-like, shape (n_c2,)
        Regularization parameter for inference.
    ratio_infer : float
        Ratio of the data used for inference.
    intercept : int
        Whether to include intercept in the model.
    offset : int
        Whether to include offset in the model.
    num_d : int
        The number of columns to be regularized. Assume the last 'num_d' columns of the covariates are the regularized coefficients. If 'num_d' is None, it is set to be 'd-offset-intercept' by default.
    C : float
        The gradients are preojected to the L2-norm ball with radius 2C for two optimization problems.
    w_type : int
        Type of the weight function.

    Returns
    -------
    df_res_raw : DataFrame
        Raw results.
    df_res : DataFrame
        Results after multiple testing correction.
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
    lam2 = c2 * np.sqrt(np.log(n)/n)

    if ratio_infer is None:
        X_infer, Y_infer = X, Y
        n_infer = Y.shape[0]
        num_missing = np.zeros(2)

        _, _, P_Gamma, A1, A2 = estimate(Y, X, r, intercept, offset, num_d, C, num_missing,
            lam1, kwargs_glm, kwargs_ls_1, kwargs_es_1, kwargs_ls_2, kwargs_es_2, **kwargs)

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
                lam1, kwargs_glm, kwargs_ls_1, kwargs_es_1, kwargs_ls_2, kwargs_es_2, **kwargs)
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
            'q_values':multipletests(np.nan_to_num(pvals, nan=1.), alpha=0.05, method='fdr_bh')[1]
        })
        _df['c2'] = c2[j]
        df_res_raw = pd.concat([df_res_raw, _df], axis=0)

        sigma = sp.stats.median_abs_deviation(
            z_scores, scale="normal", nan_policy='omit')
        z_scores = np.nan_to_num(z_scores, nan=np.nanmedian(z_scores))
        z_scores = (z_scores - np.median(z_scores)) / sigma
        pvals = sp.stats.norm.sf(np.abs(z_scores)) * 2

        _df = pd.DataFrame({
            'beta_hat':B_de[:,j]/scale_factor[0,i],
            'z_scores':z_scores, 
            'p_values':pvals,
            'q_values':multipletests(pvals, alpha=0.05, method='fdr_bh')[1]
        })
        _df['c2'] = c2[j]
        df_res = pd.concat([df_res, _df], axis=0)
    
    df_res_raw['lam'] = df_res_raw['c2'] * np.sqrt(np.log(n)/n)
    df_res['lam'] = df_res['c2'] * np.sqrt(np.log(n)/n)
    return df_res_raw, df_res


def estimate(Y, X, r, intercept, offset, num_d, C, num_missing,
    lam1, kwargs_glm, kwargs_ls_1, kwargs_es_1, kwargs_ls_2, kwargs_es_2, **kwargs):
    '''
    Two-stage estimation of the GCATE model.

    Parameters
    ----------
    Y : array-like, shape (n, p)
        Response matrix.
    X : array-like, shape (n, d)
        Observed covariate matrix.
    r : int
        Number of latent variables.
    intercept : int
        Whether to include intercept in the model.
    offset : int
        Whether to include offset in the model.
    num_d : int
        The number of columns to be regularized. Assume the last 'num_d' columns of the covariates are the regularized coefficients. If 'num_d' is None, it is set to be 'd-offset-intercept' by default.
    C : float
        The gradients are preojected to the L2-norm ball with radius 2C for two optimization problems.
    num_missing : array-like, shape (2,)
        The number of missing data for the first and second optimization problems.
    lam1 : float
        Regularization parameter for the first optimization problem.
    kwargs_glm : dict
        Keyword arguments for the GLM.
    kwargs_ls_1 : dict
        Keyword arguments of the line search algorithm for the first optimization problem.
    kwargs_ls_2 : dict
        Keyword arguments of the line search algorithm for the second optimization problem.
    kwargs_es_1 : dict
        Keyword arguments of the early stopping monitor for the first optimization problem.
    kwargs_es_2 : dict
        Keyword arguments of the early stopping monitor for the second optimization problem.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    A01 : array-like, shape (n, d+r)
        The observed covaraite and unobserved uncorrelated latent factors.
    A02 : array-like, shape (p, d+r)  
        The estimated marginal effects and latent coefficients.
    P_Gamma : array-like, shape (p, p)
        The projection matrix for the second optimization problem.
    A1 : array-like, shape (n, d+r)
        The observed covaraite and unobserved latent factors.
    A2 : array-like, shape (p, d+r)
        The estimated primary effects and latent coefficients.
    '''
    d = X.shape[1]
    p = Y.shape[1]

    A01, A02, info = alter_min(
        Y, r, X=X, P1=True, intercept=intercept, offset=offset, C=C, num_missing=num_missing,
        kwargs_glm=kwargs_glm, kwargs_ls=kwargs_ls_1, kwargs_es=kwargs_es_1, **kwargs)
    Q, _ = sp.linalg.qr(A02[:,d:], mode='economic')
    P_Gamma = np.identity(p) - Q @ Q.T

    A1, A2, info = alter_min(
        Y, r, X=X, P2=P_Gamma, A=A01.copy(), B=A02.copy(), lam=lam1, 
        intercept=intercept, offset=offset, num_d=num_d, C=C, num_missing=num_missing,
        kwargs_glm=kwargs_glm, kwargs_ls=kwargs_ls_2, kwargs_es=kwargs_es_2, **kwargs)
    return A01, A02, P_Gamma, A1, A2



def estimate_r(Y, X, r_max, c=1.,
    kwargs_glm={}, kwargs_ls_1={}, kwargs_ls_2={}, kwargs_es_1={}, kwargs_es_2={},
    c1=None, intercept=0, offset=0, num_d=1, C=None, **kwargs
):
    '''
    Estimate the number of latent factors for the GCATE model.

    Parameters
    ----------
    Y : array-like, shape (n, p)
        Response matrix.
    X : array-like, shape (n, d)
        Observed covariate matrix.
    r_max : int
        Number of latent variables.
    c : float
        The constant factor for the complexity term.
    kwargs_glm : dict
        Keyword arguments for the GLM.
    kwargs_ls_1 : dict
        Keyword arguments of the line search algorithm for the first optimization problem.
    kwargs_ls_2 : dict
        Keyword arguments of the line search algorithm for the second optimization problem.
    kwargs_es_1 : dict
        Keyword arguments of the early stopping monitor for the first optimization problem.
    kwargs_es_2 : dict
        Keyword arguments of the early stopping monitor for the second optimization problem.
    c1 : float
        Regularization parameter for the second optimization problem.
    intercept : int
        Whether to include intercept in the model.
    offset : int
        Whether to include offset in the model.
    num_d : int
        Number of latent variables.
    C : float
        The gradients are preojected to the L2-norm ball with radius 2C for two optimization problems.

    Returns
    -------
    df_r : DataFrame
        Results of the number of latent factors.
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

    kwargs_glm = {**{'family':'gaussian', 'nuisance':np.ones((1,p))}, **kwargs_glm}

    c1 = 0.1 if c1 is None else c1
    lam1 = c1 * np.sqrt(np.log(p)/n)

    res = []
    if np.isscalar(r_max):
        r_list = np.arange(1, r_max+1)
    else:
        r_list = np.array(r_max)
    
    num_missing = np.zeros(2)
    for r in r_list:
        _, _, _, A1, A2 = estimate(Y, X, r, intercept, offset, num_d, C, num_missing,
            lam1, kwargs_glm, kwargs_ls_1, kwargs_es_1, kwargs_ls_2, kwargs_es_2, **kwargs)

        logh = log_h(Y, kwargs_glm['family'], kwargs_glm['nuisance'])
        
        ll = 2 * ( 
            nll(Y, A1, A2, kwargs_glm['family'], kwargs_glm['nuisance']) / p 
            - np.sum(logh) / (n*p) ) 
        nu = (d + r) * np.maximum(n,p) * np.log(n * p / np.maximum(n,p)) / (n*p)
        jic = ll + c * nu
        res.append([r, ll, nu, jic])

    df_r = pd.DataFrame(res, columns=['r', 'deviance', 'nu', 'JIC'])
    return df_r