from scipy.optimize import linprog
import numpy as np
from numpy.linalg import inv
from sklearn import linear_model
import scipy
import math
import matplotlib.pyplot as plt
from utils import max_z_width, max_Xz
from lasso_utils import *

"""
    Conditional inference
"""
def conditional_inference(y, Sigma, A, b, eta, alpha = 0.1, grid_radius = 10, num_gridpoints = 10000):
    
    m = len(y)

    c = Sigma @ eta / (eta.T @ Sigma @ eta)
    z = (np.eye(m) - np.outer(c,eta)) @ y
    Az = A @ z
    Ac = A @ c    
    V_frac = np.divide(b - Az, Ac)
    if (Ac < 0).any():
        V_minus = np.max(V_frac[Ac < 0])
    else:
        V_minus = - np.inf
    if (Ac > 0).any():
        V_plus = np.min(V_frac[Ac > 0])
    else:
        V_plus = np.inf
        
    
    eta_dot_y = eta.dot(y)
    sigma = np.sqrt(eta.T @ Sigma @ eta)
    
    grid = np.linspace(eta_dot_y - grid_radius, eta_dot_y + grid_radius, num_gridpoints)
    
    ci_l = grid[0]
    ci_u = grid[-1]
    found_l = False
    found_u = False
    
    for i in range(len(grid)):
        mu = grid[i]
        num = scipy.stats.norm.cdf((eta_dot_y - mu)/sigma) - scipy.stats.norm.cdf((V_minus - mu)/sigma)
        denom = scipy.stats.norm.cdf((V_plus - mu)/sigma) - scipy.stats.norm.cdf((V_minus - mu)/sigma)
        
        if not found_l:
            if num/denom < 1-alpha/2:
                ci_l = mu
                found_l = True
            
        if not found_u and mu >= eta_dot_y:
            if num/denom < alpha/2:
                ci_u = mu
                found_u = True
        
        if found_u and found_l:
            break
            

    return [ci_l, ci_u]

"""
    Hybrid inference
"""
def hybrid_inference(y, Sigma, A, b, eta, alpha = 0.1, beta=0.01, num_gridpoints = 10000, SI_halfwidth = None):
    
    m = len(y)
    
    c = Sigma @ eta / (eta.T @ Sigma @ eta)
    z = (np.eye(m) - np.outer(c,eta)) @ y
    Az = A @ z
    Ac = A @ c
    V_frac = np.divide(b - Az, Ac)
    if (Ac < 0).any():
        V_minus = np.max(V_frac[Ac < 0])
    else:
        V_minus = - np.inf
    if (Ac > 0).any():
        V_plus = np.min(V_frac[Ac > 0])
    else:
        V_plus = np.inf
    
    eta_dot_y = eta.dot(y)
    sigma = np.sqrt(eta.T @ Sigma @ eta)

    if SI_halfwidth == None:
        SI_halfwidth = eta.dot(max_z_width(Sigma, beta)*np.sqrt(np.diag(Sigma)))
    
    grid = np.linspace(eta_dot_y - SI_halfwidth, eta_dot_y + SI_halfwidth, num_gridpoints)
    
    ci_l = grid[0]
    ci_u = grid[-1]
    found_l = False
    found_u = False
    
    for i in range(len(grid)):
        mu = grid[i]
        V_minus_hybrid = np.maximum(V_minus, mu - SI_halfwidth)
        V_plus_hybrid = np.minimum(V_plus, mu + SI_halfwidth) 
        
        num = scipy.stats.norm.cdf((eta_dot_y - mu)/sigma) - scipy.stats.norm.cdf((V_minus_hybrid - mu)/sigma)
        denom = scipy.stats.norm.cdf((V_plus_hybrid - mu)/sigma) - scipy.stats.norm.cdf((V_minus_hybrid - mu)/sigma)
        if not found_l:
            if num/denom < 1-(alpha-beta)/(2*(1-beta)):
                ci_l = mu
                found_l = True
            
        if not found_u and mu >= eta_dot_y:
            if num/denom < (alpha-beta)/(2*(1-beta)):
                ci_u = mu
                found_u = True
        
        if found_u and found_l:
            break
            

    return [ci_l, ci_u]

"""
    Max-z simultaneous inference
"""
def max_z_inference(point_estimate, Sigma, alpha = 0.1):
    halfwidth = max_z_width(Sigma, alpha)*np.sqrt(np.diag(Sigma))
    return [point_estimate - halfwidth, point_estimate + halfwidth]

"""
    Locally simultaneous inference
"""
def locally_simultaneous_inference(point_estimate, Sigma, plausible_inds, selected_inds, alpha = 0.1, nu = 0.01):
    Sigma_plausible = Sigma[np.ix_(plausible_inds, plausible_inds)]
    Sigma_selected = Sigma[np.ix_(selected_inds, selected_inds)]
    local_halfwidth = max_z_width(Sigma_plausible, alpha-nu)
    SI_halfwidth = max_z_width(Sigma, alpha)
    halfwidth = np.minimum(local_halfwidth, SI_halfwidth)*np.sqrt(np.diag(Sigma_selected))
    return [point_estimate - halfwidth, point_estimate + halfwidth]

def plausible_winners(y, plausible_gap):
    return np.where(y >= np.max(y) - plausible_gap)[0]

def plausible_filedrawer(y, plausible_gap, T):
    return np.where(y + plausible_gap > T)[0]

"""
    Fully simultaneous PoSI
"""
def simultaneous_PoSI(X, y, model_space, M_y, alpha=0.1, num_draws=10000):  
    curr_max_t = np.zeros(num_draws)
    n,d = X.shape
    bstrap_samples = np.random.multivariate_normal(np.zeros(n), np.eye(n), num_draws)
    for i in range(len(model_space)):
        Mhat = model_space[i]
        X_Mhat = X[:,Mhat]
        Sigma_inv = np.linalg.inv(X_Mhat.T @ X_Mhat)
        normalizing_vec = np.reshape(np.sqrt(np.diag(Sigma_inv)),(-1,1))
        t_stats = np.divide(np.abs(Sigma_inv @ X_Mhat.T @ bstrap_samples.T), normalizing_vec)
        max_t_stat_in_M = np.amax(t_stats,axis=0)
        curr_max_t = np.maximum(max_t_stat_in_M, curr_max_t)
    PoSI_constant = np.quantile(curr_max_t, 1-alpha)
    X_Mhat = X[:,M_y]
    pointest = np.linalg.pinv(X_Mhat) @ y
    Sigma_inv = np.linalg.inv(X_Mhat.T @ X_Mhat)
    return [pointest - PoSI_constant*np.sqrt(np.diag(Sigma_inv)), pointest + PoSI_constant*np.sqrt(np.diag(Sigma_inv))]

"""
    Locally simultaneous inference for the LASSO
"""
def plausible_LASSO_models_and_signs(n, d, lam, X, y, nu, M_y, s_y):
    # compute box width
    s_nu = 2*max_Xz(X, err_level = nu)
    
    # find all models and signs in box
    M_s_done = all_LASSO_models_and_signs_in_box(X, y, lam, s_nu, M_y, s_y)
    
    return M_s_done

def locally_simultaneous_LASSO(X, y, M_s_done, M_y, alpha=0.1, nu=0.01):
    
    # set of models (no signs)
    models = [M_s_done[i][0] for i in range(len(M_s_done)) if M_s_done[i][0]]
    
    # constructing intervals with local correction
    LSI_int = simultaneous_PoSI(X, y, models, M_y, alpha=(alpha-nu))
    return LSI_int