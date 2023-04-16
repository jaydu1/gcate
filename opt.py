from gcate.likelihood import *
import cvxpy as cp
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict
import statsmodels.api as sm


@njit
def line_search(Y, A, x0, g, d, lam=0., alpha=1., beta=0.5, max_iters=100, tol=1e-3, 
                family='gaussian', nuisance=1.):
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
    
    # Initialize the step size.
    t = alpha# * np.ones(x0.shape[:-1])[:, np.newaxis]
    norm_g = np.linalg.norm(g)
    #norm_g = np.linalg.norm(dx, axis=1, keepdims=True)

    # Iterate until the maximum number of iterations is reached or the step size is small enough.
    for i in range(max_iters):
        
        # Compute the new point.
        x1 = x0 - t*g
        if lam>0.:
            x1[:d] = np.sign(x1[:d]) * np.maximum(np.abs(x1[:d]) - lam, 0.)
        
        # Evaluate the function at the new point.
        f1 = nll(Y, A, x1, family, nuisance)

        # Check if the function has decreased sufficiently.
        if f1 < f0 - tol*t*norm_g:
            return x1
        if t<1e-4:
            break
        t *= beta

    # Return the maximum step size.
    # t[ind] = alpha
    x1 = x0 - alpha*g
    if lam>0.:
        x1[:d] = np.sign(x1[:d]) * np.maximum(np.abs(x1[:d]) - lam, 0.)
    return x1



@njit
def project_l2_ball(X, radius, axis=0):
    """
    Projects a matrix X to the l2 norm ball of a given radius along a specified axis.

    Parameters:
        X (ndarray): The matrix to be projected.
        radius (float): The radius of the l2 norm ball.
        axis (int, optional): The axis along which to perform the normalization.
            If axis=0 (default), each column of x is normalized.
            If axis=1, each row of x is normalized.

    Returns:
        ndarray: The projected matrix.
    """
    # norms = np.linalg.norm(X, axis=axis)
    norms = np.sum(X**2, axis=axis)**(1./2)
    mask = norms > radius

    if np.any(mask):
        if axis==1:
            X[mask,:] *= radius / np.expand_dims(norms[mask], axis)
        else:
            X[:,mask] *= radius / np.expand_dims(norms[mask], axis)

    return X


@njit
def prox_gd(x, g, eta, C, lam=0.):
    """
    Projected gradient descent algorithm for optimization.

    Parameters:
        x (ndarray): Initial point for the optimization.
        g (ndarray): Gradient.
        eta (float): Step size for the gradient descent.

    Returns:
        ndarray: Optimal point found by the algorithm.
    """
    x = x - eta * g
    x = project_l2_ball(x, C, axis=1)

    if lam>0.:
        x = np.sign(x) * np.maximum(np.abs(x) - lam, 0)
    
    return x


@njit(parallel=True)
def update(Y, A, B, d, lam, P1, P2,
          family, nuisance, C,
          alpha, beta, max_iters, tol):
    n, p = Y.shape
    
    g = grad(Y.T, B, A, family, nuisance)
    g[:, :d] = 0.
    if P1 is not None:
        g[:, d:] = P1 @ g[:, d:]
    for i in prange(n):
        A[i, :] = line_search(Y[i, :].T, B, A[i, :], g[i, :], d, 0.,
                          alpha, beta, max_iters, tol,
                          family, nuisance)
#     A[:, d:] = prox_gd(A[:, d:], g[:, d:], eta, C, lam=0.)

    g = grad(Y, A, B, family, nuisance)
    if P2 is not None:
        g[:,:d] = P2 @ g[:,:d]
    for j in prange(p):
        B[j, :] = line_search(Y[:, j], A, B[j, :], g[j, :], d, lam,
                          alpha, beta, max_iters, tol,
                          family, nuisance)
#     B = prox_gd(B, g, eta, C, lam=lam)

    func_val = nll(Y, A, B, family, nuisance)

    return func_val, A, B


def alter_min(
    Y, r, X=None, P2=None, C=None, lam=0.,
    A=None, B=None,
    kwargs_glm={},
    kwargs_ls={}, max_iters=100, eps=1e-4):
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
    kwargs_ls = {**{'alpha':1., 'beta':0.5, 'max_iters':100, 'tol':1e-3}, **kwargs_ls}
    
    n, p = Y.shape
    
    if C is None:
        C = 5 * np.sqrt(r)
        
    if X is None:
        d = 0
        P1 = None
    else:
        d = X.shape[1]
        Q, _ = sp.linalg.qr(X, mode='economic')
        P1 = np.identity(n) - Q @ Q.T

    # to do : check X has col norm <= C

    # to do: svd start
    # initialization
    # Theta = A @ B^T
    if A is None or B is None:
        A = np.empty((n, r))
        B = np.empty((p, d+r))
        E = np.log(Y+1)
        
        if d>0:
            A = np.c_[X, A]
            if kwargs_glm['family']=='gaussian':
                B[:, :d] = np.c_[[sm.GLM(Y[:,j], X, family=sm.families.Gaussian()).fit().params for j in range(p)]]
            elif kwargs_glm['family']=='poisson' or kwargs_glm['family']=='negative_binomial':
                B[:, :d] = np.c_[[sm.GLM(Y[:,j], X, family=sm.families.Poisson()).fit().params for j in range(p)]]
            else:
                raise ValueError('Family not recognized')
            # E = E - X@B[:, :d].T
            E = P1 @ E
        u, s, vh = sp.sparse.linalg.svds(E, k=r)        
        A[:, d:] = u * s[None,:]**(1/2) / np.sqrt(n)
        B[:, d:] = vh.T * s[None,:]**(1/2) / np.sqrt(p)
        del E, u, s, vh
            
#     if P1 is not None:
#         A[:,d:] = P1 @ A[:,d:] / np.sqrt(n)
    if P2 is not None:        
        B[:,:d] = P2 @ B[:,:d] / np.sqrt(p)


    func_val_pre = nll(Y, A, B, kwargs_glm['family'], kwargs_glm['nuisance'])
    hist = [func_val_pre]
    # to do: parallel
    with tqdm(np.arange(max_iters)) as pbar:
        for t in pbar:
            func_val, A, B = update(
                Y, A, B, d, lam, P1, P2,
                kwargs_glm['family'], kwargs_glm['nuisance'], C,
                kwargs_ls['alpha'], kwargs_ls['beta'], kwargs_ls['max_iters'], kwargs_ls['tol']
            )
            hist.append(func_val)
            if not np.isfinite(func_val) or np.abs(func_val_pre - func_val)<eps*p:
                break
            else:
                func_val_pre = func_val
            pbar.set_postfix(nll='{:.02f}'.format(func_val))

    info = {'n_iter':t, 'func_val':func_val, 'resid':func_val_pre - func_val,
           'hist':hist}
    return A, B, info



