from gcate.opt import *



def _debias_opt(bpp, g, X, P_Gamma_j, j, i, lam_n, tau_n):
    n, d = X.shape
    v = 1/(bpp + 1e-8) * P_Gamma_j
    w = 1/np.sum(v**2, axis=-1)
    X_w = np.sqrt(w)[:,None] * X        
    
    ei = np.zeros((d,1))    
    ei[i] = 1.0
    
    hSigma = X_w.T @ X_w / n + 1e-2 * np.identity(d)

    u = cp.Variable((d,1))
    objective = cp.Minimize(
#         cp.sum_squares(X_w @ u)# /n
        cp.quad_form(u, hSigma) + lam_n* cp.norm_inf(hSigma @ u - n*ei)
                           ) 
    constraints = [
#         cp.norm_inf(hSigma @ u - n * ei) <= lam_n,
        cp.norm_inf(X @ u) <= tau_n
    ]
    prob = cp.Problem(objective, constraints)
    obj_val = prob.solve(solver=cp.ECOS)

    obj_val = np.mean((X_w @ u.value)**2)

    return np.array([np.sqrt(obj_val/n), 
                     np.mean(w[:,None] * (v @ g @ u.value))])


def debias(Y, X, B, A1, A2, P_Gamma, i,
           c1=1., c2=1., kwargs_glm={}):
    n, p = Y.shape
    _, d = X.shape
    
    lam_n = c1# * np.sqrt(np.log(p)/n)
    tau_n = c2 * np.sqrt(np.log(n))
    
    Theta_hat = A1 @ A2.T
    g = grad(Y, A1, A2, **kwargs_glm)[:,:d]
    bpp = hess(Y, Theta_hat, **kwargs_glm)
    
    se = np.zeros_like(B)
    B_de = np.zeros_like(B)
    
    with Parallel(n_jobs=16, verbose=0, timeout=99999) as parallel:
        res = parallel(
                delayed(_debias_opt)(bpp, g, X, P_Gamma[:, j], j, i, lam_n, tau_n) for j in tqdm(range(p))
            )
    res = np.r_[res]
    se[:, i] = res[:,0]
    B_de[:, i] = B[:,i] - res[:,1]

#         info = {'i':i, 'j':j, 'obj_val':obj_val, 'status':prob.status}
    return B_de, se