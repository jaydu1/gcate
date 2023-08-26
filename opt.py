from gcate.likelihood import *
from gcate.utils import *
import cvxpy as cp
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
import pprint


@njit
def line_search(Y, A, x0, g, d, lam=0., alpha=1., beta=0.5, max_iters=100, tol=1e-3, 
                family='gaussian', nuisance=1., intercept=1):
    """
    Performs line search to find the step size that minimizes a given function.
    
    Parameters:
        f (callable): A scalar-valued function of a vector argument.
        x0 (ndarray): The starting point for the line search.
        g (ndarray): The search direction vector.
        alpha (float, optional): The initial step size. Default is 10.
        beta (float, optional): The shrinkage factor used to reduce the step size. Default is 0.5.
        max_iters (int, optional): The maximum number of iterations. Default is 100.
        tol (float, optional): The tolerance for the step size. Default is 1e-3.
    
    Returns:
        float: The step size that minimizes the function along the search direction.
    """

    # Evaluate the function at the starting point.
    f0 = nll(Y, A, x0, family, nuisance)
    if lam>0.:
        f0 += lam * np.sum(np.abs(x0[intercept:d]))
    
    # Initialize the step size.    
    alpha = type_f(alpha)
    beta = type_f(beta)
    t = alpha
    norm_g = np.linalg.norm(g)

    # Iterate until the maximum number of iterations is reached or the step size is small enough.
    for i in range(max_iters):
        
        # Compute the new point.
        x1 = x0 - t*g
        if lam>0.:
            x1[intercept:d] = np.sign(x1[intercept:d]) * np.maximum(np.abs(x1[intercept:d]) - lam, type_f(0.))
        
        # Evaluate the function at the new point.
        f1 = nll(Y, A, x1, family, nuisance) 
        if lam>0.:
            f1 += lam * np.sum(np.abs(x1[intercept:d]))
        if i==0:
            f01 = f1

        # Check if the function has decreased sufficiently.
        if f1 < f0 - tol*t*norm_g:
            return x1
        if t<1e-4:
            break
        t *= beta

    # Return the maximum step size.
    if f01<2*f1:
        x1 = x0 - alpha*g
        if lam>0.:
            x1[intercept:d] = np.sign(x1[intercept:d]) * np.maximum(np.abs(x1[intercept:d]) - lam, type_f(0.))
    return x1



@njit
def project_l2_ball(X, radius):
    """
    Projects a vector X to the l2 norm ball of a given radius.

    Parameters:
        X (ndarray): The vector to be projected.
        radius (float): The radius of the l2 norm ball.

    Returns:
        ndarray: The projected matrix.
    """
#     if np.max(np.abs(X)):
#         X = np.clip(X, -radius, radius)
    norms = np.linalg.norm(X)
    if norms > radius:
        X *= radius / norms

    return X

# @njit
# def project_l2_ball(X, radius, axis=0):
#     """
#     Projects a matrix X to the l2 norm ball of a given radius along a specified axis.

#     Parameters:
#         X (ndarray): The matrix to be projected.
#         radius (float): The radius of the l2 norm ball.
#         axis (int, optional): The axis along which to perform the normalization.
#             If axis=0 (default), each column of x is normalized.
#             If axis=1, each row of x is normalized.

#     Returns:
#         ndarray: The projected matrix.
#     """
#     # norms = np.linalg.norm(X, axis=axis)
#     norms = np.sum(X**2, axis=axis)**(1./2)
#     mask = norms > radius
    
#     if np.any(mask):
#         if axis==1:
#             X[mask,:] *= radius / np.expand_dims(norms[mask], axis)
#         else:
#             X[:,mask] *= radius / np.expand_dims(norms[mask], axis)

#     return X


# @njit
# def prox_gd(x, g, eta, C, lam=0.):
#     """
#     Projected gradient descent algorithm for optimization.

#     Parameters:
#         x (ndarray): Initial point for the optimization.
#         g (ndarray): Gradient.
#         eta (float): Step size for the gradient descent.

#     Returns:
#         ndarray: Optimal point found by the algorithm.
#     """
#     x = x - eta * g
#     x = project_l2_ball(x, C, axis=1)

#     if lam>0.:
#         x = np.sign(x) * np.maximum(np.abs(x) - lam, 0)
    
#     return x


@njit(parallel=True)
def update(Y, A, B, d, lam, P1, P2,
          family, nuisance, C,
          alpha, beta, max_iters, tol, offset, intercept):
    n, p = Y.shape
    
    g = grad(Y.T, B, A, family, nuisance)
    g[:, :d] = 0.
#     if P1 is not None:
#         g[:, d:] = P1 @ g[:, d:]
    for i in prange(n):
        g[i, d:] = project_l2_ball(g[i, d:], 2*C)
        A[i, :] = line_search(Y[i, :].T, B, A[i, :], g[i, :], d, type_f(0.),
                          alpha, beta, max_iters, tol,
                          family, nuisance, intercept)

#     A[:, d:] = prox_gd(A[:, d:], g[:, d:], eta, C, lam=0.)
#     A[:, d:] = np.clip(A[:, d:], -C, C)
#     if P1 is not None:
#         
    for j in prange(d, B.shape[1]):
        mu = np.mean(A[:, j])
        B[:,0] += mu * B[:, j]
        A[:,j] -= mu
#         mu = np.sum(A[:, d:], axis=0) / n
#         B[:,offset] = B[:,offset] + B[:, d:] @ mu # assuming there is an intercept
#         A[:,d:] =  A[:,d:] - np.expand_dims(mu, 0)
    
    g = grad(Y, A, B, family, nuisance)
    g[:, :offset+intercept] = 0.
    if P2 is not None:
        g[:, offset+intercept:d] = P2 @ g[:, offset+intercept:d]
        g[:, d:] = 0.

    for j in prange(p):
        if P2 is None:
            g[j, :] = project_l2_ball(g[j, :], 2*C)
        else:
            g[j, :d] = project_l2_ball(g[j, :d], 2*C)
        B[j, :] = line_search(Y[:, j], A, B[j, :], g[j, :], d, lam,
                          alpha, beta, max_iters, tol,
                          family, nuisance, offset+intercept)
#     if P2 is not None:
#         B[:, offset:d] = P2 @ B[:, offset:d]
#     B = prox_gd(B, g, eta, C, lam=lam)
#     if P2 is None:
#         B[:, d:] = np.clip(B[:, d:], -C, C)
    func_val = nll(Y, A, B, family, nuisance)

    return func_val, A, B


def alter_min(
    Y, r, X=None, P1=None, P2=None, C=1e4, lam=0.,
    A=None, B=None,
    kwargs_glm={}, kwargs_ls={}, kwargs_es={}, intercept=1, offset=1, verbose=True):
    '''
    Alternative minimization of latent factorization for generalized linear models.

    Parameters:
        X (ndarray): The matrix to be projected.
        radius (float): The radius of the l2 norm ball.
        axis (int, optional): The axis along which to perform the normalization.
            If axis=0 (default), each column of x is normalized.
            If axis=1, each row of x is normalized.

    Returns:
        ndarray: The projected matrix.
    '''
    kwargs_glm = {**{'family':'gaussian', 'nuisance':1.}, **kwargs_glm}
    kwargs_ls = {**{'alpha':0.1, 'beta':0.5, 'max_iters':20, 'tol':1e-4}, **kwargs_ls}
    kwargs_es = {**{'max_iters':200, 'warmup':5, 'patience':20, 'tolerance':1e-4}, **kwargs_es}
    
    
    if verbose:
        pprint.pprint({'kwargs_glm':kwargs_glm,'kwargs_ls':kwargs_ls,'kwargs_es':kwargs_es})

    n, p = Y.shape
    
#     if C is None:
#         C = 5 * np.sqrt(r)
        
    if X is None:
        d = 0
        P1 = None
    else:
        d = X.shape[1]
        if P1 is True:
            Q, _ = sp.linalg.qr(X[:,offset:], mode='economic')
            P1 = np.identity(n) - Q @ Q.T
            P1 = P1.astype(type_f)

    # to do : check X has col norm <= C

    # initialization for Theta = A @ B^T
    if A is None or B is None:
        A = np.empty((n, r), dtype=type_f)
        B = np.empty((p, d+r), dtype=type_f)
        E = np.log(Y+1)
        
        if d>0:
            A = np.c_[X, A]
            
            offset_arr = None if offset==0 else X[:,0]
            if kwargs_glm['family']=='gaussian':
                B[:, offset:d] = np.c_[[sm.GLM(Y[:,j], X[:,offset:], offset=offset_arr, family=sm.families.Gaussian()
                                        ).fit_regularized(alpha=1e-5,L1_wt=0.).params for j in range(p)]]
            elif kwargs_glm['family']=='poisson' or kwargs_glm['family']=='negative_binomial':
                B[:, offset:d] = np.c_[[sm.GLM(Y[:,j], X[:,offset:], offset=offset_arr, family=sm.families.Poisson()
                                        ).fit_regularized(alpha=1e-5,L1_wt=0.).params for j in range(p)]]
            elif kwargs_glm['family']=='binomial':
                B[:, offset:d] = np.c_[[sm.GLM(Y[:,j], X[:,offset:], offset=offset_arr, family=sm.families.Binomial()
                                        ).fit_regularized(alpha=1e-5,L1_wt=0.).params for j in range(p)]]
            else:
                raise ValueError('Family not recognized')
            # E = E - X@B[:, :d].T
            E = P1 @ E / np.sqrt(n * p)
        u, s, vh = sp.sparse.linalg.svds(E, k=r)        
        A[:, d:] = u * s[None,:]**(1/2) * np.sqrt(n)
        B[:, d:] = vh.T * s[None,:]**(1/2) * np.sqrt(p)
        del E, u, s, vh
#         if intercept==1:
#             B[:,0] += np.mean(A[:, d:], axis=0) @ B[:, d:].T

        if offset==1:
            scale = np.sqrt(np.median(np.abs(X[:,0])))#np.sqrt(np.linalg.norm(X[:,offset-1]))
            B[:, :offset] = scale
            A[:, :offset] /= scale 
            
        if kwargs_glm['family']=='negative_binomial':
            B[:, offset] = -(np.max(A @ B.T, axis=0) + 1e-2)

    if P2 is not None:  
        P2 = P2.astype(type_f)
        B[:,offset+intercept:d] = P2 @ B[:,offset+intercept:d]
        
    Y = Y.astype(type_f)
    A = A.astype(type_f)
    B = B.astype(type_f)
    lam = type_f(lam)
    
    family, nuisance = kwargs_glm['family'], kwargs_glm['nuisance']
#     g1 = grad(Y.T, B, A, family, nuisance)[:, d:]
#     g2 = grad(Y, A, B, family, nuisance)[:, d:]
#     eta_2 = (np.std(g2) / np.std(g1))
#     if lam>0:
#         g = grad(Y, A, B, kwargs_glm['family'], kwargs_glm['nuisance'])[:,offset+intercept:d]
# #         lams = np.maximum(lam * np.abs(g), 
# #                           np.maximum(lam * np.quantile(np.abs(g), 0.05), 1e-5)
# #                          )
#         lams = np.ones((p,d-offset-intercept), dtype=type_f) * lam
#         lams = lams.astype(type_f)
#     else:
#         lams = np.zeros((p,d-offset-intercept), dtype=type_f)
#     lams = lam * np.ones((p,d-offset-intercept), dtype=type_f)
    
    func_val_pre = nll(Y, A, B, family, nuisance)/p + lam * np.mean(np.abs(B[:,offset+intercept:d]))
    hist = [func_val_pre]
    es = Early_Stopping(**kwargs_es)
    with tqdm(np.arange(kwargs_es['max_iters'])) as pbar:
        for t in pbar:
            func_val, A, B = update(
                Y, A, B, d, lam, P1, P2,
                family, nuisance, C,
                kwargs_ls['alpha'], kwargs_ls['beta'], kwargs_ls['max_iters'], kwargs_ls['tol'], 
                offset,intercept
            )
            func_val = func_val/p + lam * np.mean(np.abs(B[:,offset+intercept:d]))
            hist.append(func_val)
            if not np.isfinite(func_val) or func_val>np.maximum(1e3*np.abs(func_val_pre),1e3):
                print('Encountered large or infinity values. Try to decrease the value of C for the norm constraints.')
                break
            elif es(func_val):
                print('Early stopped.')
                break
            elif lam>0. and np.mean(np.abs(B[:,offset+intercept:d])<1e-4)>0.95:
                print('Stopped because coefficients are too sparse.')
                break
            else:
                func_val_pre = func_val
            pbar.set_postfix(nll='{:.02f}'.format(func_val))
    
    info = {'n_iter':t, 'func_val':func_val, 'resid':func_val_pre - func_val,
           'hist':hist}
    return A, B, info



