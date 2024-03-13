import numpy as np
import pandas as pd
import scipy as sp
import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot



class Early_Stopping():
    '''
    The early-stopping monitor.
    '''
    def __init__(self, warmup=25, patience=25, tolerance=0., is_minimize=True, **kwargs):
        self.warmup = warmup
        self.patience = patience
        self.tolerance = tolerance
        self.is_minimize = is_minimize

        self.step = -1
        self.best_step = -1
        self.best_metric = np.inf

        if not self.is_minimize:
            self.factor = -1.0
        else:
            self.factor = 1.0

    def __call__(self, metric):
        self.step += 1
        
        if self.step < self.warmup:
            return False
        elif self.factor*metric<self.factor*self.best_metric-self.tolerance:
            self.best_metric = metric
            self.best_step = self.step
            return False
        elif self.step - self.best_step>self.patience:
            print('Best Epoch: %d. Best Metric: %f.'%(self.best_step, self.best_metric))
            return True
        else:
            return False




def plot_r(df_r, c=1):
    '''
    Plot the results of the estimation of the number of latent factors.

    Parameters
    ----------
    df_r : DataFrame
        Results of the number of latent factors.
    c : float
        The constant factor for the complexity term.

    Returns
    -------
    fig : Figure
        The figure of the plot.
    '''
    
    
    fig = plt.figure(figsize=[18,6])
    host = host_subplot(121)
    par = host.twinx()

    host.set_xlabel("Number of factors $r$")
    host.set_ylabel("Deviance")
    # par.set_ylabel("$\nu$")


    p1, = host.plot(df_r['r'], df_r['deviance'], '-o', label="Deviance")
    p2, = par.plot(df_r['r'], df_r['nu']*c, '-o', label=r"$\nu$")


    host.set_xticks(df_r['r'])
    host.yaxis.get_label().set_color(p1.get_color())
    par.tick_params(axis='y', colors=p2.get_color(), labelsize=14)
    host.tick_params(axis='y', colors=p1.get_color(), labelsize=14)

    p1, = host.plot(df_r['r'], df_r['deviance']+df_r['nu']*c, '-o', label="JIC")
    host.legend(labelcolor="linecolor")


    host = host_subplot(122)
    par = host.twinx()
    host.set_xlabel("Number of factors $r$")
    par.set_ylabel(r"$\nu$")

    p1, = host.plot(df_r['r'].iloc[1:], -np.diff(df_r['deviance']), '-o', label='diff dev')
    p2, = par.plot(df_r['r'].iloc[1:], np.diff(df_r['nu'])*c,  '-o', label=r'diff $\nu$')

    host.legend(labelcolor="linecolor")
    host.set_xticks(df_r['r'].iloc[1:])
    par.set_ylim(*host.get_ylim())
    
    par.yaxis.get_label().set_color(p2.get_color())
    par.tick_params(axis='y', colors=p2.get_color(), labelsize=14)
    host.tick_params(axis='y', colors=p1.get_color(), labelsize=14)

    return fig


def plot_lam(df_res, med_max=np.inf, mad_max=np.inf,
    lam_min=0., lam_max=None, alpha=0.1):
    '''
    Plot the results of the estimation of the regularization parameter.

    Parameters
    ----------
    df_res : DataFrame
        Results of the regularization parameter.
    med_max : float
        The maximum value of the median.
    mad_max : float
        The maximum value of the MAD.
    lam_min : float
        The minimum value of the regularization parameter.
    lam_max : float
        The maximum value of the regularization parameter.
    alpha : float
        The transparency of the shaded area.

    Returns
    -------
    fig : Figure
        The figure of the plot.
    lams, medians, mads : numpy array
        The median and mad statistics.
    '''

    lams, medians, mads = estimate_lam(df_res)
    
    fig = plt.figure(figsize=[8,6])
    host = host_subplot(111)
    par = host.twinx()

    host.set_xlabel("Penalty $\lambda_n$")
    host.set_ylabel("Median")
    par.set_ylabel("MAD")

    p1, = host.plot(lams, medians, '-o', label="Median")
    p2, = par.plot(lams, mads, '-o', label="MAD")

    host.legend(labelcolor="linecolor")

    host.yaxis.get_label().set_color(p1.get_color())
    par.yaxis.get_label().set_color(p2.get_color())

    par.tick_params(axis='y', colors=p2.get_color(), labelsize=14)
    host.tick_params(axis='y', colors=p1.get_color(), labelsize=14)
    
    if lam_max is None:
        idx = (np.abs(medians) < med_max)&(mads < mad_max)
        if np.any(idx):
            lam_max = np.max(lams[idx])
        else:
            lam_max = np.inf
            
    if lam_max < np.max(lams):
        fig.axes[0].axvspan(lam_min, lam_max, color='green', alpha=alpha)

    return fig, (lams, medians, mads)


def estimate_lam(df_res):
    lams = df_res['lam'].unique()
    medians = []
    mads = []
    for lam in lams:
        _df = df_res[df_res['lam']==lam]
        median = np.nanmedian(_df['z_scores'])
        mad = sp.stats.median_abs_deviation(_df['z_scores'], scale="normal", nan_policy='omit')
        medians.append(median)
        mads.append(mad)
    medians = np.array(medians)
    mads = np.array(mads)
    return lams, medians, mads