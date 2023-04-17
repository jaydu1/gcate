import numpy as np
from scipy.special import expit, xlogy, xlog1py, logsumexp, factorial
from scipy.stats import binom, poisson, norm, nbinom
import scipy as sp
import numba as nb
from numba import njit, prange


@njit
def nll(Y, A, B, family='gaussian', nuisance=1.):
    """
    Compute the negative log likelihood for generalized linear models with optional nuisance parameters.
    
    Parameters:
    Y : array-like of shape (n_samples, n_features)
        The response variable.
    A : array-like of shape (n_samples, n_factors)
        The input data matrix.
    B : array-like of shape (n_features, n_factors)
        The input data matrix.
    family : str, optional (default='gaussian')
        The family of the generalized linear model. Options include 'gaussian', 'binomial', 'poisson',
        and 'negative_binomial'.
    nuisance : float or array-like of shape (n_samples,), optional (default=None)
        The nuisance parameter for the family. For the Gaussian family, this is the variance; for the Poisson
        family, this is the scaling factor; and for the negative binomial family, this is the overdispersion
        parameter. If None, the default value of the nuisance parameter is used.
    
    Returns:
    nll : float
        The negative log likelihood.
    """
    
    Theta = A @ B.T
    Ty = Y.copy()
    n = Y.shape[0]
    
    if family == 'binomial':
        b = nuisance * np.log(1 + np.exp(Theta))
    elif family == 'poisson':
        Theta = np.clip(Theta, -np.inf, 1e2)
        b = np.exp(Theta)
    elif family == 'gaussian':
        b = Theta**2/2.
        Ty /= np.sqrt(nuisance)
    elif family == 'negative_binomial': # link = 'log'  
        Theta = np.clip(Theta, -np.inf, -1e-2)
        b = - nuisance * np.log(1 - np.exp(Theta))
    else:
        raise ValueError('Family not recognized')
    nll = - np.sum(Ty * Theta - b) / np.float64(n)
    return nll



@njit
def grad(Y, A, B, family='gaussian', nuisance=1.):
    """
    Compute the gradient of log likelihood with respect to B
    for generalized linear models with optional nuisance parameters.
    
    The natural parameter of Y is A @ B^T.
    
    Parameters:
    Y : array-like of shape (n_samples, n_features)
        The response variable.
    A : array-like of shape (n_samples, n_factors)
        The input data matrix.
    B : array-like of shape (n_features, n_factors)
        The input data matrix.
    family : str, optional (default='gaussian')
        The family of the generalized linear model. Options include 'gaussian', 'binomial', 'poisson',
        and 'negative_binomial'.
    nuisance : float or array-like of shape (n_samples,), optional (default=None)
        The nuisance parameter for the family. For the Gaussian family, this is the variance; for the Poisson
        family, this is the scaling factor; and for the negative binomial family, this is the overdispersion
        parameter. If None, the default value of the nuisance parameter is used.
    
    Returns:
    grad : array-like of shape (n_features, n_factors)
        The gradient of log likelihood.
    """
    Theta = A @ B.T
    Ty = Y.copy()
    n = Y.shape[0]
    
    if family == 'binomial':
        b_p = nuisance / (1 + np.exp(-Theta))
    elif family == 'poisson':
        Theta = np.clip(Theta, -np.inf, 1e2)
        b_p = np.exp(Theta)
    elif family == 'gaussian':
        b_p = Theta
        Ty /= np.sqrt(nuisance)
    elif family == 'negative_binomial': # link = 'log'  
        Theta = np.clip(Theta, -np.inf, -1e-2)
        b_p = nuisance / (np.exp(-Theta) - 1)
    else:
        raise ValueError('Family not recognized')
    grad = - (Ty - b_p).T @ A / np.float64(n)#(np.float64(n) * np.float64(p))
    return grad



@njit
def hess(Y, Theta, family='gaussian', nuisance=1.):
    """
    Compute the gradient of log likelihood with respect to B
    for generalized linear models with optional nuisance parameters.
    
    The natural parameter of Y is A @ B^T.
    
    Parameters:
    Y : array-like of shape (n_samples, n_features)
        The response variable.
    Theta : array-like of shape (n_samples, n_features)
        The natural parameter matrix.
    family : str, optional (default='gaussian')
        The family of the generalized linear model. Options include 'gaussian', 'binomial', 'poisson',
        and 'negative_binomial'.
    nuisance : float or array-like of shape (n_samples,), optional (default=None)
        The nuisance parameter for the family. For the Gaussian family, this is the variance; for the Poisson
        family, this is the scaling factor; and for the negative binomial family, this is the overdispersion
        parameter. If None, the default value of the nuisance parameter is used.
    
    Returns:
    hess : float
        The hessian of the log likelihood with respect to Theta.
    """
    Ty = Y.copy()
    n = Y.shape[0]
    
    if family == 'binomial':
        b_pp = nuisance * np.exp(-Theta) / (1 + np.exp(-Theta))**2
    elif family == 'poisson':
        Theta = np.clip(Theta, -np.inf, 1e2)
        b_pp = np.exp(Theta)
    elif family == 'gaussian':
        b_pp = np.ones_like(Theta)
    elif family == 'negative_binomial': # link = 'log'  
        Theta = np.clip(Theta, -np.inf, -1e-2)
        b_pp = nuisance * np.exp(Theta) / (1 - np.exp(Theta))**2
    else:
        raise ValueError('Family not recognized')
    hess = b_pp #(np.float64(n) * np.float64(p))
    return hess