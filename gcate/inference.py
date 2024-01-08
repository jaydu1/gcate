from gcate.opt import *
import pandas as pd
import warnings



def _debias_opt(bpp, g, X, P_Gamma_j, j, i, lam_n):
    n, d = X.shape
    w = 1/np.sum((np.sqrt(1/(bpp+1e-8)) * P_Gamma_j)**2, axis=-1)
    X_w = np.sqrt(w)[:,None] * X
    
    ei = np.zeros((d,1))
    ei[i] = 1.0
    
    hSigma = X_w.T @ X_w + 1e-6 * np.identity(d)
    scale = 1#np.trace(hSigma)/d
    
    u = cp.Variable((d,1))
    uv = cp.Variable((1,1))
    
    lam = cp.Parameter(nonneg=True)
    lam.value = lam_n
    
    objective = cp.Minimize(
        #cp.quad_form(u, hSigma)# + lam_n* cp.norm_inf(hSigma @ u - ei)
        # dual problem
        cp.quad_form(u + uv * ei, hSigma)/(4* scale * n) + 
        1 / scale * (
            cp.sum(cp.multiply(u, ei)) + uv + 
            lam * (cp.norm1(uv)+ cp.norm1(u))
        )
    )

    prob = cp.Problem(objective)
    while lam.value>1e-5:
        try:
            obj_val = prob.solve()
            u_val = - (u.value + uv.value * ei) / 2.
            obj_val = np.mean((X_w @ u_val)**2)
            se = np.sqrt(obj_val/n)
            bias = u_val.T @ (w[:,None] * X).T @ g @ P_Gamma_j / n
            bias = bias[0]
            if se>1e-6:
                break
        except:
            pass
        lam.value = 0.9 * lam.value
    if lam.value<1e-5:        
        warnings.warn("Optimization fail for {}. Use median estimates.".format(j))
        se = np.nan
        bias = np.nan

    return np.array([se, bias])



def _debias_opt_path(bpp_tilde, bpp, g, X, P_Gamma_j, j, i, lams, w_type=0):
    n, d = X.shape
    
    if w_type==0:
        w = bpp[:,j]
    elif w_type==1:
        w = 1. / (np.sqrt(1/bpp) @ P_Gamma_j)**2
        
    X_wt = np.sqrt(w)[:,None] * X
    X_w = X_wt
    
    ei = np.zeros((d,1))
    ei[i] = 1.0
    
    hSigma = X_wt.T @ X_wt
    scale = np.trace(hSigma)/d
    
    u = cp.Variable((d,1))
    uv = cp.Variable((1,1))
    
    lam = cp.Parameter(nonneg=True)

    # primal formulation
#     objective = cp.Minimize(
#         cp.quad_form(u, hSigma)/ n
#     )    
#     constraint = [cp.norm(hSigma/n @ u - ei, 'inf')<=lam]
#     prob = cp.Problem(objective, constraint)

    # dual formulation
    objective = cp.Minimize(
        cp.quad_form(u, hSigma)/(4 * scale * n) + 
        1. / scale * (
            - cp.sum(cp.multiply(u, ei)) + 
            lam * cp.norm1(u)
        )
    )
    prob = cp.Problem(objective)
    
    arr_se, arr_bias = np.zeros(len(lams)), np.zeros(len(lams))
    for i_lam, lam_n in enumerate(lams):
        lam.value = lam_n
        try:
            obj_val = prob.solve()
            u_val = u.value / 2.
            obj_val = np.mean((X_w @ u_val)**2)
            arr_se[i_lam] = np.sqrt(obj_val/n)
            bias = u_val.T @ (w[:,None] * X).T @ g @ P_Gamma_j / n
            arr_bias[i_lam] = bias[0]
        except:
            arr_se[i_lam] = np.nan
            arr_bias[i_lam] = np.nan
        
    return np.c_[arr_se, arr_bias].T




def debias(Y, A1, A2, P_Gamma, d, i,
           lam, kwargs_glm={}, intercept=0, offset=0, 
           x_type=0, w_type=0, n_jobs=64, num_d=None):
    n, p = Y.shape
    
    Y = Y.astype(type_f)
    A1 = A1.astype(type_f)
    A2 = A2.astype(type_f)

    if num_d is None:
        num_d = d - offset
    
    Theta_hat = A1 @ A2.T
    bpp = hess(Y, Theta_hat, **kwargs_glm)
    g = grad(Y, A1, A2, **kwargs_glm, direct=True)
    g = g/bpp
    
    if np.isscalar(lam):
        lam = np.array([lam])
        
    with Parallel(n_jobs=n_jobs, verbose=0, timeout=99999) as parallel:
        res = parallel(delayed(_debias_opt_path)(
            bpp, bpp, g, 
            A1[:,offset:] if x_type==0 else A1[:,offset:d], 
            P_Gamma[:, j], j, i-offset, lam, w_type) 
            for j in tqdm(range(p))
            )
    res = np.stack(res, axis=0)

    
    B_de = A2[:,i][:,None] - res[:,1]
    
    B_de = pd.DataFrame(B_de.T).fillna(method='bfill').values.T    
    se = pd.DataFrame(res[:,0].T).fillna(method='bfill').values.T
    
    return B_de, se



def comp_stat(df, alpha=0.05, q_alpha=0.2, n_top=250):
    return pd.DataFrame.from_dict({
        'num_rej': np.sum(df['p_values']<alpha),
        'num_sig': np.sum(df['signals']==1),
        "type1_err": np.mean(df[df['signals']==0]['p_values']<alpha),
        "power": np.mean(df[df['signals']==1]['p_values']<alpha),
        "num_rej_q": np.sum(df['q_values']<q_alpha),
        "fdp": np.mean(df[df['q_values']<q_alpha]['signals']==0),
        'precision':np.mean(df['signals'][np.argsort(np.abs(df['p_values']))[:n_top]])
    }, orient='index').T
