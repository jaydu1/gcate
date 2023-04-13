from gcate.likelihood import *
import cvxpy as cp
from tqdm import tqdm
from joblib import Parallel, delayed



def line_search(f, x0, g, alpha=10, beta=0.5, max_iters=100, tol=1e-3):
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
    f0 = f(x0)
    
    # Initialize the step size.
    t = alpha# * np.ones(x0.shape[:-1])[:, np.newaxis]
    norm_g = np.linalg.norm(g, ord='fro')
    #norm_g = np.linalg.norm(dx, axis=1, keepdims=True)

    # Iterate until the maximum number of iterations is reached or the step size is small enough.
    for i in range(max_iters):
        
        # Compute the new point.
        x1 = x0 - t*g
        
        # Evaluate the function at the new point.
        f1 = f(x1)

        # Check if the function has decreased sufficiently.
#         ind = (f1 > f0 - tol*t*norm_g)
        
#         if np.any(ind):
#             return t
        
#         # Shrink the step size.
#         t[ind] *= beta
        if f1 < f0 - tol*t*norm_g:
            return t
        
        t *= beta

    # Return the maximum step size.
    # t[ind] = alpha
    return alpha



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
    norms = np.linalg.norm(X, axis=axis)
    mask = norms > radius
    # create a copy of x to avoid modifying the original input
    proj_X = X.copy()

    if np.any(mask):
        if axis==1:
            proj_X[mask,:] *= radius / norms[mask, np.newaxis]
        else:
            proj_X[:,mask] *= radius / norms[np.newaxis, mask]        

    return proj_X



def prox_gd(x, g, eta, C, proj_func=None, lam=0.):
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

    if proj_func is not None:
        x = proj_func(x)
    if lam>0.:
        x = np.sign(x) * np.maximum(np.abs(x) - lam, 0)
    
    return x



def alter_min(
    Y, r, X=None, P1=None, P2=None, C=None, lam=0.,
    A=None, B=None,
    kwargs_glm={},
    kwargs_ls={}, max_iters=100, eps=1e-6):
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
    
    if C is None:
        C = 5 * np.sqrt(r)
        
    if X is None:
        d = 0
    else:
        d = X.shape[1]
        Q, _ = sp.linalg.qr(X)
        
    if P1 is None:
        proj_func_1 = None
    else:
        proj_func_1 = lambda A:P1 @ A#np.c_[A[:,:d], P1 @ A[:,d:]]
    if P2 is None:
        proj_func_2 = None
    else:
        proj_func_2 = lambda B:np.c_[P2 @ B[:,:d], B[:,d:]]        

    # to do : check X has col norm <= C

    # to do: svd start
    # initialization
    # Theta = A @ B^T
    if A is None:
        A = np.random.rand(n, r)/np.sqrt(n)
        if d>0:
            A = np.c_[X, A]        
    if B is None:
        B = np.random.rand(p, d+r)/np.sqrt(p)
    

    func_val_pre = nll(Y, A, B, **kwargs_glm)
    hist = [func_val_pre]
    # to do: parallel
    for t in tqdm(np.arange(max_iters)):
        g = grad(Y.T, B, A, **kwargs_glm)
        g[:, :d] = 0.
        eta = line_search(
            lambda A:nll(Y.T, B, A, **kwargs_glm), A, g, **kwargs_ls)
        A[:,d:] = prox_gd(A[:,d:], g[:,d:], eta, C, proj_func_1, lam=0.)

        g = grad(Y, A, B, **kwargs_glm)
        eta = line_search(
            lambda B:nll(Y, A, B, **kwargs_glm), B, g, **kwargs_ls)
        B = prox_gd(B, g, eta, C, proj_func_2, lam=lam)
        
        func_val = nll(Y, A, B, **kwargs_glm)
        if np.abs(func_val_pre - func_val)<eps:
            break
        else:
            func_val_pre = func_val
            hist.append(func_val)
            print(func_val)
    info = {'n_iter':t, 'func_val':func_val, 'resid':func_val_pre - func_val,
           'hist':hist}
    return A, B, info



def _debias_opt(bpp, g, X, P_Gamma_j, j, i, lam_n, tau_n):
    n, d = X.shape
    v = 1/(bpp + 1e-8) * P_Gamma_j
    w = 1/np.sum(v**2, axis=-1)
    X_w = np.sqrt(w)[:,None] * X        
    
    ei = np.zeros((d,1))    
    ei[i] = 1.0
    
    hSigma = X_w.T @ X_w /n + 1e-2 * np.identity(d)
    # Construct the problem.
    u = cp.Variable((d,1))
    objective = cp.Minimize(cp.quad_form(u, hSigma)/n
                           + lam_n * cp.norm_inf(hSigma @ u - ei))
    constraints = [
#         cp.norm_inf(hSigma @ u - ei) <= lam_n,
        cp.norm_inf(X @ u) <= tau_n
    ]
    prob = cp.Problem(objective, constraints)
    obj_val = prob.solve(solver=cp.ECOS)
#     obj_val = u.value.T @ hSigma @ u.value / n

    return np.array([np.sqrt(obj_val / n), np.mean(w[:,None] * (v @ g @ u.value))])


def debias(Y, X, B, A1, A2, P_Gamma, i,
           c1=1., c2=1., kwargs_glm={}):
    n, p = Y.shape
    _, d = X.shape
    
    lam_n = c1 #* np.sqrt(np.log(p)/n)
    tau_n = c2 * np.sqrt(np.log(n))
    
    Theta_hat = A1 @ A2.T
    g = grad(Y, A1, A2, **kwargs_glm)[:,:d]
    bpp = hess(Y, Theta_hat, **kwargs_glm)
    
    se = np.zeros_like(B)
    B_de = np.zeros_like(B)
    
    with Parallel(n_jobs=8, verbose=0, timeout=99999) as parallel:
        res = parallel(
                delayed(_debias_opt)(bpp, g, X, P_Gamma[:, j], j, i, lam_n, tau_n) for j in tqdm(range(p))
            )
    res = np.r_[res]
    se[:, i] = res[:,0]
    B_de[:, i] = B[:,i] - res[:,1]

#         info = {'i':i, 'j':j, 'obj_val':obj_val, 'status':prob.status}
    return B_de, se