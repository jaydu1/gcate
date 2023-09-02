from gcate.likelihood import *
from gcate.utils import *
from gcate.glm import *
import cvxpy as cp
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import pprint


@njit
def proj_qp(A, b, max_iterations=1000, L=1.0, tolerance=1e-4):
    m, n = A.shape
    w = np.zeros(m)
    y = np.zeros(m)
    t = 1.0
    L_inv = 1.0 / L

    for i in range(max_iterations):
        
        w_prev = w.copy()
        y_prev = y.copy()

        u = A.T @ w
        v = np.minimum(A @ u - L * w, b)

        y = w - L_inv * (A @ u - v)
        
        
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t**2))
        w = y + (1.0 / L) * (t - 1) / t_new * (y - y_prev)

        if np.linalg.norm(w - w_prev) < tolerance:
            break

        t = t_new
    return A.T @ w


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


@njit
def line_search(Y, A, x0, g, d, family, nuisance, intercept,
                lam=0., alpha=1., beta=0.5, max_iters=100, tol=1e-3, 
                ):
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






@njit(parallel=True)
def update(Y, A, B, d, lam, P1, P2,
          family, nuisance, C,
          alpha, beta, max_iters, tol, offset, intercept, num_d):
    n, p = Y.shape
    
    g = grad(Y.T, B, A, family, nuisance.T)
    g[:, :d] = 0.
    if family=='negative_binomial':
        A_prev = A.copy()
    for i in prange(n):
        g[i, d:] = project_l2_ball(g[i, d:], 2*C)
        A[i, :] = line_search(Y[i, :].T, B, A[i, :], g[i, :], d, 
                              family, nuisance[0], intercept,
                          type_f(0.), alpha, beta, max_iters, tol)
    if P1 is not None:
        A[:, d:] = P1 @ A[:, d:]

    if family=='negative_binomial':
        for i in prange(n):
            max_val = np.max(B @ A[i,:])
            if max_val>0:
                delta = A[i, d:] - A_prev[i, d:]
                factor = - np.max(B[:,d:] @ delta) * (np.max(B @ A_prev[i,:]) - 1e-4)
                A[i, d:] = A_prev[i, d:] + factor * delta
    
    
    g = grad(Y, A, B, family, nuisance)
    g[:, :d-num_d] = 0.
#     g[:, :offset] = 0.
    if P1 is not None:
        g[:, :d] = 0.
    elif P2 is not None:
        g[:, d-num_d:d] = P2 @ g[:, d-num_d:d]
        g[:, d:] = 0.
    
    if family=='negative_binomial':
        B_prev = B.copy()
    for j in prange(p):
        
        if P2 is None:
            g[j, :] = project_l2_ball(g[j, :], 2*C)
        else:
            g[j, :d] = project_l2_ball(g[j, :d], 2*C)

        B[j, :] = line_search(Y[:, j], A, B[j, :], g[j, :], d, 
                              family, nuisance[:,j], d-num_d,
                              lam, alpha, beta, max_iters, tol
                             )

        if family=='negative_binomial':
            max_val = np.max(A @ B[j,:])
            if max_val>0:
                if P2 is None:
                    delta = B[j, d:] - B_prev[i, d:]
                    factor = - np.max(A[:,d:] @ delta) * (np.max(A @ B_prev[j, :]) - 1e-4)
                    B[j, d:] = B_prev[j, d:] + factor * delta
                else:
                    delta = B[j, :d] - B_prev[i, :d]
                    factor = - np.max(A[:,:d] @ delta) * (np.max(A @ B_prev[j, :]) - 1e-4)
                    B[j, :d] = B_prev[j, :d] + factor * delta

#     if P2 is not None:
#         B[:, offset:d] = P2 @ B[:, offset:d]
#     B = prox_gd(B, g, eta, C, lam=lam)
#     if P2 is None:
#         B[:, d:] = np.clip(B[:, d:], -C, C)
    func_val = nll(Y, A, B, family, nuisance)

    return func_val, A, B


def alter_min(
    Y, r, X=None, P1=None, P2=None, C=1e5, lam=0.,
    A=None, B=None,
    kwargs_glm={}, kwargs_ls={}, kwargs_es={}, 
    intercept=1, offset=1, num_d=None, verbose=True):
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
    n, p = Y.shape
    
    kwargs_glm = {**{'family':'gaussian', 'nuisance':np.ones((1,p))}, **kwargs_glm}
    kwargs_ls = {**{'alpha':0.1, 'beta':0.5, 'max_iters':20, 'tol':1e-4}, **kwargs_ls}
    kwargs_es = {**{'max_iters':200, 'warmup':5, 'patience':20, 'tolerance':1e-4}, **kwargs_es}
    
    
    if verbose:
        pprint.pprint({'kwargs_glm':kwargs_glm,'kwargs_ls':kwargs_ls,'kwargs_es':kwargs_es})
    family, nuisance = kwargs_glm['family'], kwargs_glm['nuisance'].astype(type_f)
    
    
    
#     if C is None:
#         C = 5 * np.sqrt(r)
    d = X.shape[1]
    assert d>0
    if P1 is True:
        Q, _ = sp.linalg.qr(X[:,offset:], mode='economic')
        P1 = np.identity(n) - Q @ Q.T
        P1 = P1.astype(type_f)
            
    if num_d is None:
        num_d = d-offset-intercept
        
    if verbose:
        pprint.pprint({'n':n,'p':p,'d':d,'r':r})
    # to do : check X has col norm <= C

    # initialization for Theta = A @ B^T
    if A is None or B is None:
        A = np.empty((n, r), dtype=type_f)
        B = np.empty((p, d+r), dtype=type_f)
        E = init_inv_link(Y, family, nuisance)
        
        if d>0:
            A = np.c_[X, A]
            
            offset_arr = None if offset==0 else X[:,0]
            alpha = np.full(d-offset, 1e-8)
            alpha[:intercept] = 0.
            B[:, offset:d] = fit_glm(Y, X[:,offset:], offset_arr, family, nuisance[0])
            
            # E = E - X@B[:, :d].T
            E = P1 @ E #/ np.sqrt(n * p)
        u, s, vh = sp.sparse.linalg.svds(E, k=r)        
        A[:, d:] = u * s[None,:]**(1/2) #* np.sqrt(n)
        B[:, d:] = vh.T * s[None,:]**(1/2) #* np.sqrt(p)
        del E, u, s, vh

        if offset==1:
            scale = np.sqrt(np.median(np.abs(X[:,0])))#np.sqrt(np.linalg.norm(X[:,offset-1]))
            B[:, :offset] = scale
            A[:, :offset] /= scale 
            
        if family=='negative_binomial':
#             assert intercept==1
            factor = np.sqrt(- np.max(A[:,d:] @ B[:,d:].T) * (np.max(A[:,:d] @ B[:,:d].T) - 1e-4))
            A[:, d:] *= factor
            B[:, d:] *= factor
            
#             L = np.linalg.norm(A[:, d:]) ** 2
#             for j in prange(p):
#                 B[j, d:] += proj_qp(A[:, d:], - A @ B[j, :], L=L)
#             for i in prange(n):
#                 A[i, d:] += proj_qp(B[:, d:], - B @ A[i, :], L=L)
                
#             B[:, offset] = np.minimum(0., -(np.max(A @ B.T, axis=0) + 1e-2))

    if P2 is not None:  
        P2 = P2.astype(type_f)
        E = A[:,d-num_d:] @ B[:,d-num_d:].T @ (np.identity(p) - P2)
        u, s, vh = sp.sparse.linalg.svds(E, k=r)
        B[:, d-num_d:d] = P2 @ B[:, d-num_d:d]
        A[:, d:] = u * s[None,:]**(1/2)
        B[:, d:] = vh.T * s[None,:]**(1/2)
        del E, u, s, vh
                
    Y = Y.astype(type_f)
    A = A.astype(type_f)
    B = B.astype(type_f)
    lam = type_f(lam)
    
    assert ~np.any(np.isnan(A))
    assert ~np.any(np.isnan(B))
    
    func_val_pre = nll(Y, A, B, family, nuisance)/p + lam * np.mean(np.abs(B[:,d-num_d:d]))
    hist = [func_val_pre]
    es = Early_Stopping(**kwargs_es)
    with tqdm(np.arange(kwargs_es['max_iters'])) as pbar:
        for t in pbar:
            func_val, A, B = update(
                Y, A, B, d, lam, P1, P2,
                family, nuisance, C,
                kwargs_ls['alpha'], kwargs_ls['beta'], kwargs_ls['max_iters'], kwargs_ls['tol'], 
                offset,intercept,num_d
            )
            func_val = func_val/p + lam * np.mean(np.abs(B[:,d-num_d:d]))
            hist.append(func_val)
            if not np.isfinite(func_val) or func_val>np.maximum(1e3*np.abs(func_val_pre),1e3):
                print('Encountered large or infinity values. Try to decrease the value of C for the norm constraints.')
                break
            elif es(func_val):
                print('Early stopped.')
                break
#             elif lam>0. and np.mean(np.abs(B[:,d-num_d:d])<1e-4)>0.95:
#                 print('Stopped because coefficients are too sparse.')
#                 break
            else:
                func_val_pre = func_val
            pbar.set_postfix(nll='{:.02f}'.format(func_val))
    
    info = {'n_iter':t, 'func_val':func_val, 'resid':func_val_pre - func_val,
           'hist':hist}
    return A, B, info



