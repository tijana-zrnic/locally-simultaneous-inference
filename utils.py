import numpy as np
from numpy.linalg import inv
from sklearn import linear_model
import scipy
import math
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

"""
    Max-z quantile
"""
def max_z_width(Sigma, err_level = 0.1, num_draws = 100000):
    m = Sigma.shape[0]
    bstrap_noise = np.random.multivariate_normal(np.zeros(m), Sigma, num_draws)
    bstrap_noise_normed = np.divide(bstrap_noise, np.sqrt(np.diag(Sigma)))
    max_noise = np.amax(np.abs(bstrap_noise_normed), axis = 1)
    return np.quantile(max_noise, 1-err_level)

"""
    Plot and save results
"""
def plot_and_save_results(xaxis, simultaneous_widths, local_widths, cond_widths, hybrid_widths, plot_title, ylabel, xlabel, filename, alpha=0.1, log_scale=False, plot_conditional=True, plot_hybrid=True, plot_simultaneous=True, fill_between=True, plot_baseline=False, baseline_val = None, ylim = None):
    if plot_simultaneous:
        simultaneous_median = np.quantile(simultaneous_widths, 0.5, axis=1)
        simultaneous_95 = np.quantile(simultaneous_widths, 0.95, axis=1)
        simultaneous_5 = np.quantile(simultaneous_widths, 0.05, axis=1)

    local_median = np.quantile(local_widths, 0.5, axis=1)
    local_95 = np.quantile(local_widths, 0.95, axis=1)
    local_5 = np.quantile(local_widths, 0.05, axis=1)

    if plot_conditional:
        cond_median = np.quantile(cond_widths, 0.5, axis=1)
        cond_95 = np.quantile(cond_widths, 0.95, axis=1)
        cond_5 = np.quantile(cond_widths, 0.05, axis=1)
    if plot_hybrid:
        hybrid_median = np.quantile(hybrid_widths, 0.5, axis=1)
        hybrid_95 = np.quantile(hybrid_widths, 0.95, axis=1)
        hybrid_5 = np.quantile(hybrid_widths, 0.05, axis=1)


    plt.clf()
    plt.plot(xaxis, local_median, 'lightcoral', label='locally simultaneous', linewidth=3)
    if fill_between:
        plt.fill_between(xaxis, local_5, local_95, color='lightcoral', alpha = 0.2)

    if plot_simultaneous:
        plt.plot(xaxis, simultaneous_median, 'dodgerblue', label='simultaneous', linewidth=3)
        if fill_between:
            plt.fill_between(xaxis, simultaneous_5, simultaneous_95, color='dodgerblue', alpha = 0.2)

    if plot_conditional:
        plt.plot(xaxis, cond_median, 'goldenrod', label='conditional', linewidth=3)
        if fill_between:
            plt.fill_between(xaxis, cond_5, cond_95, color='goldenrod', alpha = 0.2)
        
    if plot_hybrid:
        plt.plot(xaxis, hybrid_median, 'mediumturquoise', label='hybrid', linewidth=3)
        if fill_between:
            plt.fill_between(xaxis, hybrid_5, hybrid_95, color='mediumturquoise', alpha = 0.2)

    if plot_baseline:
        if baseline_val != None:
            plt.plot(xaxis, [baseline_val] * len(xaxis), 'gray', linewidth=3, linestyle = 'dashed')
        else:
            plt.plot(xaxis, [2*scipy.stats.norm.isf(alpha/2)] * len(xaxis), 'gray', linewidth=3, linestyle = 'dashed')

    plt.legend(fontsize=14)
    plt.title(plot_title, fontsize = 22)
    pivot_ax = plt.gca()
    pivot_ax.set_ylabel(ylabel, fontsize = 22)
    pivot_ax.set_xlabel(xlabel, fontsize = 22)
    if log_scale:
        plt.xscale('log')
    if ylim != None:
        plt.ylim(ylim)
        
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    path = filename
    plt.savefig(path, bbox_inches='tight')

"""
    Power set
"""
def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]

"""
    Polyhedron describing selecting the winner
"""
def inference_on_winner_polyhedron(m, selected_ind):
    b = np.zeros(m-1)
    A = np.eye(m)
    A = np.delete(A, selected_ind, 0)
    A[:, selected_ind] = -1
    return A, b

"""
    Polyhedron describing filedrawer problem
"""
def filedrawer_polyhedron(m, selected_inds, T):
    A = np.eye(m)
    A[selected_inds, selected_inds] = -1
    b = np.ones(m)*T
    b[selected_inds] *= -1
    return A, b

"""
    Kernel for filedrawer experiments
"""
def exponentiated_quadratic(xa, xb, scale):
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm/(scale**2))

"""
    Simulate maximum over i of |X_i^T Z|, where Z is standard normal
"""
def max_Xz(X, err_level = 0.1, num_draws = 100000):
    n,d = X.shape
    bstrap_noise = np.random.multivariate_normal(np.zeros(n), np.eye(n), num_draws)
    Xz_abs = np.abs(X.T.dot(bstrap_noise.T))
    max_noise = np.amax(Xz_abs, axis=0)
    return np.quantile(max_noise, 1-err_level)

"""
    Generate (X,y) data used in the LASSO experiments
"""
def generate_data(n, d, beta):
    X = np.random.randn(n,d)
    X = X / np.linalg.norm(X,axis=0)
    mu = X @ beta
    y = mu + np.random.randn(n)
    return X, y


def X_in_selected_model(X, M_y):
    d = X.shape[1]
    E_M = np.zeros((d,len(M_y)))
    for l in range(len(M_y)):
        E_M[M_y[l], l] = 1

        E_Mc = np.zeros((d, d - len(M_y)))
        Mc_y = sorted(list(set(range(d)) - set(M_y)))
        for lc in range(len(Mc_y)):
            E_Mc[Mc_y[lc], lc] = 1

        X_M = X @ E_M
        X_Mc = X @ E_Mc
    return X_M, X_Mc, E_M, E_Mc

"""
    Bentkus' concentration bound used for plausible set in nonparametric IOW
"""
def bentkus_gap(n, alpha):
    grid = np.linspace(0,1,10000)[1:-1]
    p_lb = scipy.stats.binom.ppf(alpha/(2*np.exp(1)), n, grid)
    p_ub = scipy.stats.binom.ppf(1-alpha/(2*np.exp(1)), n, grid)
    return np.max(np.maximum(grid*n - p_lb, p_ub - grid*n))/n

"""
    Nonparametric confidence intervals by Waudby-Smith and Ramdas
"""
def wsr(x_n, alpha, grid = None, intersection: bool = True,
            theta: float = 0.5, c: float = 0.75):
    if grid == None:
        grid = np.linspace(0,1,10000)[1:-1]
    n = x_n.shape[0]
    t_n = np.arange(1, n + 1)
    muhat_n = (0.5 + np.cumsum(x_n)) / (1 + t_n)
    sigma2hat_n = (0.25 + np.cumsum(np.power(x_n - muhat_n, 2))) / (1 + t_n)
    sigma2hat_tminus1_n = np.append(0.25, sigma2hat_n[: -1])
    assert(np.all(sigma2hat_tminus1_n > 0))
    lambda_n = np.sqrt(2 * np.log(2 / alpha) / (n * sigma2hat_tminus1_n))

    def M(m):
        lambdaplus_n = np.minimum(lambda_n, c / m)
        lambdaminus_n = np.minimum(lambda_n, c / (1 - m))
        return np.maximum(
            theta * np.exp(np.cumsum(np.log(1 + lambdaplus_n * (x_n - m)))),
            (1 - theta) * np.exp(np.cumsum(np.log(1 - lambdaminus_n * (x_n - m))))
        )

    
    indicators_gxn = np.zeros([grid.size, n])
    found_lb = False
    for m_idx, m in enumerate(grid):
        m_n = M(m)
        indicators_gxn[m_idx] = m_n < 1 / alpha
        if not found_lb and np.prod(indicators_gxn[m_idx]):
            found_lb = True
        if found_lb and not np.prod(indicators_gxn[m_idx]):
            break  # since interval, once find a value that fails, stop searching
    if intersection:
        ci_full = grid[np.where(np.prod(indicators_gxn, axis=1))[0]]
    else:
        ci_full =  grid[np.where(indicators_gxn[:, -1])[0]]
    if ci_full.size == 0:  # grid maybe too coarse
        idx = np.argmax(np.sum(indicators_gxn, axis=1))
        if idx == 0:
            return np.array([grid[0], grid[1]])
        return [grid[idx - 1], grid[idx]]
    return [ci_full.min(), ci_full.max()]


"""
    For plotting
"""
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def make_interval(ci_lsi, ci_fsi, ci_cond, ci_hybrid, plot_title, to_celsius = True, legend=True):
    if to_celsius:
        offset = 273
    else:
        offset = 0
        
    ci_cond_temp = ci_cond.copy()
        
    noclip_l = 1
    noclip_u = 1
        
    lsi_width = ci_lsi[1] - ci_lsi[0]
    if ci_cond[0] < ci_lsi[0] - lsi_width:
        ci_cond_temp[0] = ci_lsi[0] - lsi_width
        noclip_l = 0
    if ci_cond[1] > ci_lsi[1] + lsi_width:
        ci_cond_temp[1] = ci_lsi[1] + lsi_width
        noclip_u = 0
        
    plt.plot([ci_lsi[0] - offset, ci_lsi[1] - offset],[0.7,0.7], linewidth=20, color='lightsalmon', path_effects=[pe.Stroke(linewidth=22, offset=(-1,0), foreground='lightcoral'), pe.Stroke(linewidth=22, offset=(1,0), foreground='lightcoral'), pe.Normal()], label='locally simultaneous', solid_capstyle='butt')
    plt.plot([ci_fsi[0] - offset, ci_fsi[1] - offset],[0.55, 0.55], linewidth=20, color='lightskyblue', path_effects=[pe.Stroke(linewidth=22, offset=(-1,0), foreground='dodgerblue'), pe.Stroke(linewidth=22, offset=(1,0), foreground='dodgerblue'), pe.Normal()], label='simultaneous', solid_capstyle='butt')
    plt.plot([ci_cond_temp[0] - offset, ci_cond_temp[1] - offset],[0.4,0.4], linewidth=20, color='wheat', path_effects=[pe.Stroke(linewidth=22, offset=(-noclip_l,0), foreground='goldenrod'), pe.Stroke(linewidth=22, offset=(noclip_u,0), foreground='goldenrod'), pe.Normal()], label='conditional', solid_capstyle='butt')
    plt.plot([ci_hybrid[0] - offset, ci_hybrid[1] - offset],[0.25, 0.25], linewidth=20, color='paleturquoise', path_effects=[pe.Stroke(linewidth=22, offset=(-1,0), foreground='mediumturquoise'), pe.Stroke(linewidth=22, offset=(1,0), foreground='mediumturquoise'), pe.Normal()], label='hybrid', solid_capstyle='butt')
#     plt.plot([ci_uncorected[0], ci_uncorected[1]],[0.1, 0.1], linewidth=20, color="#FFEACC", path_effects=[pe.Stroke(linewidth=22, offset=(-1,0), foreground="#FFCD82"), pe.Stroke(linewidth=22, offset=(1,0), foreground="#FFCD82"), pe.Normal()], label='uncorrected', solid_capstyle='butt')
    plt.ylabel("")
    plt.title(plot_title, fontsize=14)
    plt.xlabel('temperature (°C)', fontsize=14)
    plt.yticks([])
    plt.ylim([0.15,0.8])
    plt.xlim([None, None])
    if legend:
        plt.legend(bbox_to_anchor = (1.7,1), borderpad=1, labelspacing = 1, fontsize=14)


def make_multiple_intervals(ci_lsi, ci_fsi, ci_cond, ci_hybrid, plot_title, to_celsius = True, legend=True):
    if to_celsius:
        offset = 273
    else:
        offset = 0

    k = 3 # number of intervals to plot
    
    for i in reversed(range(k)):
        
        ci_cond_temp = [ci_cond[0][i].copy(), ci_cond[1][i].copy()]
        
        noclip_l = 1
        noclip_u = 1
        
        lsi_width = ci_lsi[1][i] - ci_lsi[0][i]
        if ci_cond[0][i] < ci_lsi[0][i] - lsi_width:
            ci_cond_temp[0] = ci_lsi[0][i] - lsi_width
            noclip_l = 0
        if ci_cond[1][i] > ci_lsi[1][i] + lsi_width:
            ci_cond_temp[1] = ci_lsi[1][i] + lsi_width
            noclip_u = 0
    
        if i == 0:
            plt.plot([ci_lsi[0][i] - offset, ci_lsi[1][i] - offset],[0.7+i*0.02,0.7+i*0.02], linewidth=20, color='lightsalmon', path_effects=[pe.Stroke(linewidth=22, offset=(-1,0), foreground='lightcoral'), pe.Stroke(linewidth=22, offset=(1,0), foreground='lightcoral'), pe.Normal()], label='locally simultaneous', solid_capstyle='butt')
            plt.plot([ci_fsi[0][i] - offset, ci_fsi[1][i] - offset],[0.55+i*0.02, 0.55+i*0.02], linewidth=20, color='lightskyblue', path_effects=[pe.Stroke(linewidth=22, offset=(-1,0), foreground='dodgerblue'), pe.Stroke(linewidth=22, offset=(1,0), foreground='dodgerblue'), pe.Normal()], label='simultaneous', solid_capstyle='butt')
            plt.plot([ci_cond_temp[0] - offset, ci_cond_temp[1] - offset],[0.4+i*0.02,0.4+i*0.02], linewidth=20, color='wheat', path_effects=[pe.Stroke(linewidth=22, offset=(-noclip_l,0), foreground='goldenrod'), pe.Stroke(linewidth=22, offset=(noclip_u,0), foreground='goldenrod'), pe.Normal()], label='conditional', solid_capstyle='butt')
            plt.plot([ci_hybrid[0][i] - offset, ci_hybrid[1][i] - offset],[0.25+i*0.02, 0.25+i*0.02], linewidth=20, color='paleturquoise', path_effects=[pe.Stroke(linewidth=22, offset=(-1,0), foreground='mediumturquoise'), pe.Stroke(linewidth=22, offset=(1,0), foreground='mediumturquoise'), pe.Normal()], label='hybrid', solid_capstyle='butt')
        if i > 0:
            plt.plot([ci_lsi[0][i] - offset, ci_lsi[1][i] - offset],[0.7+i*0.02,0.7+i*0.02], linewidth=20, color= lighten_color('lightsalmon',0.8/i), path_effects=[pe.Stroke(linewidth=22, offset=(-1,0), foreground='lightcoral'), pe.Stroke(linewidth=22, offset=(1,0), foreground='lightcoral'), pe.Normal()], solid_capstyle='butt')
            plt.plot([ci_fsi[0][i] - offset, ci_fsi[1][i] - offset],[0.55+i*0.02, 0.55+i*0.02], linewidth=20, color=lighten_color('lightskyblue',0.8/i), path_effects=[pe.Stroke(linewidth=22, offset=(-1,0), foreground='dodgerblue'), pe.Stroke(linewidth=22, offset=(1,0), foreground='dodgerblue'), pe.Normal()], solid_capstyle='butt')
            plt.plot([ci_cond_temp[0] - offset, ci_cond_temp[1] - offset],[0.4+i*0.02,0.4+i*0.02], linewidth=20, color=lighten_color('wheat',0.8/i), path_effects=[pe.Stroke(linewidth=22, offset=(-noclip_l,0), foreground='goldenrod'), pe.Stroke(linewidth=22, offset=(noclip_u,0), foreground='goldenrod'), pe.Normal()], solid_capstyle='butt')
            plt.plot([ci_hybrid[0][i] - offset, ci_hybrid[1][i] - offset],[0.25+i*0.02, 0.25+i*0.02], linewidth=20, color=lighten_color('paleturquoise',0.8/i), path_effects=[pe.Stroke(linewidth=22, offset=(-1,0), foreground='mediumturquoise'), pe.Stroke(linewidth=22, offset=(1,0), foreground='mediumturquoise'), pe.Normal()], solid_capstyle='butt')

            
    plt.ylabel("")
    plt.title(plot_title, fontsize=14)
    plt.xlabel('temperature (°C)', fontsize=14)
    plt.yticks([])
    plt.ylim([0.15,0.8])
    plt.xlim([None, None])
    if legend:
        plt.legend(bbox_to_anchor = (1.7,1), borderpad=1, labelspacing = 1, fontsize=14)