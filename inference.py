from gcate.opt import *
import pandas as pd
import warnings



def _debias_opt(bpp, g, X, P_Gamma_j, j, i, lam_n, tau_n):
    n, d = X.shape
    w = 1/np.sum((np.sqrt(1/(bpp+1e-8)) * P_Gamma_j)**2, axis=-1)
    X_w = np.sqrt(w)[:,None] * X
    
    ei = np.zeros((d,1))
    ei[i] = 1.0
    
    hSigma = X_w.T @ X_w / n + 1e-4 * np.identity(d)
    hSigma_inv = np.linalg.pinv(hSigma)

    u = cp.Variable((d,1))
    uv = cp.Variable((1,1))
    
    lam = cp.Parameter(nonneg=True)
    lam.value = lam_n
    
    objective = cp.Minimize(
        #cp.quad_form(u, hSigma)# + lam_n* cp.norm_inf(hSigma @ u - ei)
        # dual problem
        cp.quad_form(u + uv * ei, hSigma_inv)/4 + 
        cp.sum(cp.multiply(u, ei)) + uv + 
        lam * (cp.norm1(uv)+ cp.norm1(u))
    )
    constraints = [
#         cp.norm_inf(hSigma @ u - ei) <= lam,
#         cp.norm_inf(X @ u) <= tau_n
    ]
    prob = cp.Problem(objective, constraints)
    while lam.value<10.:        
        try:
            obj_val = prob.solve(solver=cp.ECOS)
            u = - (u.value + uv.value * ei) / 2.
            obj_val = np.mean((X_w @ u)**2)
            se = np.sqrt(obj_val/n)
            bias = u.T @ (w[:,None] * X).T @ g @ P_Gamma_j / n
            break
        except:
            lam.value = lam.value * 2
    if lam.value>1.:        
        warnings.warn("Optimization fail for {}. Use median estimates.".format(j))
        se = np.nan
        bias = np.nan

    return np.array([se, bias])


def debias(Y, X, B, A1, A2, P_Gamma, i,
           c1=1., c2=1., kwargs_glm={}):
    n, p = Y.shape
    _, d = X.shape
    
    lam_n = c1 * np.sqrt(np.log(p)/n)
    tau_n = c2 * np.sqrt(np.log(n))
    
    Theta_hat = A1 @ A2.T
    g = grad(Y, A1, A2, **kwargs_glm, direct=True)
    bpp = hess(Y, Theta_hat, **kwargs_glm)
    g = g/bpp
    
    se = np.zeros_like(B)
    B_de = np.zeros_like(B)
    
    with Parallel(n_jobs=16, verbose=0, timeout=99999) as parallel:
        res = parallel(
                delayed(_debias_opt)(bpp, g, X, P_Gamma[:, j], j, i, lam_n, tau_n) for j in tqdm(range(p))
            )
    res = np.r_[res]
    med = np.nanmedian(res, axis=0)
    for j in range(2):
        res[:,j] = np.nan_to_num(res[:,j], med[j])
    se[:, i] = res[:,0]
    B_de[:, i] = B[:,i] - res[:,1]

#         info = {'i':i, 'j':j, 'obj_val':obj_val, 'status':prob.status}
    return B_de, se



def comp_stat(df, alpha=0.05, q_alpha=0.2):
    return pd.DataFrame.from_dict({
        'num_rej': np.sum(df['p_values']<alpha),
        'num_sig': np.sum(df['signals']==1),
        "type1_err": np.mean(df[df['signals']==0]['p_values']<alpha),
        "power": np.mean(df[df['signals']==1]['p_values']<alpha),
        "num_rej_q": np.sum(df['q_values']<q_alpha),
        "fdp": np.mean(df[df['q_values']<q_alpha]['signals']==0)
    }, orient='index').T