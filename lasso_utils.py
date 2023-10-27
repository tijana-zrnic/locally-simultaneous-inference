from scipy.optimize import linprog
import numpy as np
from numpy.linalg import inv
from sklearn import linear_model
import scipy
import math

from utils import powerset

"""
    LASSO selection event
"""
def lasso_constraints(lam, X_Mc, X_M, E_M, E_Mc, s):   
    Sigma_inv = np.linalg.inv(X_M.T @ X_M)
    A0_plus = 1/lam*(E_Mc.T - X_Mc.T @ X_M @ Sigma_inv @ E_M.T)
    A0_minus = - A0_plus  
    b0 = X_Mc.T @ np.linalg.pinv(X_M.T) @ s
    b0_plus = np.ones(X_Mc.shape[1]) - b0
    b0_minus = np.ones(X_Mc.shape[1]) + b0
    A1 = - np.diag(s) @ Sigma_inv @ E_M.T
    b1 = - lam*np.diag(s) @ Sigma_inv @ s
    return A0_plus, A0_minus, A1, b0_plus, b0_minus, b1

def lasso_constraints_Xy_space(lam, X_Mc, X_M, E_M, E_Mc, s):
    Sigma_inv = np.linalg.inv(X_M.T @ X_M)
    A0_plus = 1/lam*(X_Mc.T - X_Mc.T @ X_M @ Sigma_inv @ X_M.T)
    A0_minus = - A0_plus
    b0 = X_Mc.T @ np.linalg.pinv(X_M.T) @ s
    b0_plus = np.ones(X_Mc.shape[1]) - b0
    b0_minus = np.ones(X_Mc.shape[1]) + b0
    A1 = - np.diag(s) @ Sigma_inv @ X_M.T
    b1 = - lam*np.diag(s) @ Sigma_inv @ s
    return A0_plus, A0_minus, A1, b0_plus, b0_minus, b1

"""
    Redundancy check
"""
def redundancy_check(lam, X_Mc, X_M, suff_stat, s, M, Mc, s_nu):
    Sigma_inv = np.linalg.inv(X_M.T @ X_M)
    beta_M_s = Sigma_inv @ (suff_stat[M] - lam * s)
    res_vec = suff_stat[Mc] - X_Mc.T @ X_M @ beta_M_s
    thresh = lam - s_nu*(1 + np.linalg.norm(X_Mc.T @ X_M @ Sigma_inv,1,axis=1))
    return [(np.abs(res_vec) < thresh), (np.linalg.norm(Sigma_inv, 1, axis=1)*s_nu < np.abs(beta_M_s))]

"""
    Remove redundant constraints from LP
"""
def redundant_constr_LP(A_ineq, b_ineq, removed_indices, d):
    num_constr = len(b_ineq) - 2*d # don't care about box constraints
    constraints_to_check = sorted(list(set(range(num_constr)) - set(removed_indices)))
    for i in constraints_to_check:

        a_removed = A_ineq[i, :]
        A_ineq_no_i = np.delete(A_ineq, removed_indices + [i], axis=0)

        b_removed = b_ineq[i]
        b_ineq_no_i = np.delete(b_ineq, removed_indices + [i])

        LP = linprog(-a_removed, A_ub=A_ineq_no_i, b_ub=b_ineq_no_i, bounds = (-np.Inf,np.Inf), method='simplex')
        LP_val = -LP.fun

        if LP_val <= b_removed + 0.000001: # fudge factor to get around numerical issues
            removed_indices.append(i)
    return sorted(list(set(range(num_constr)) - set(removed_indices)))

"""
    Finding minimal polyhedral representation of LASSO selection event
"""
def lasso_minimal_polyhedron(X, suff_stat, lam, M, s, s_nu, indices_to_skip):
    
    [n, d] = X.shape
    E_M = np.zeros((d,len(M)))
    for i in range(len(M)):
        E_M[M[i], i] = 1
    E_Mc = np.zeros((d, d - len(M)))
    Mc = sorted(list(set(range(d)) - set(M)))
    for i in range(len(Mc)):
        E_Mc[Mc[i], i] = 1

    X_M = X @ E_M
    X_Mc = X @ E_Mc
    
    # LASSO constraints
    A0_plus, A0_minus, A1, b0_plus, b0_minus, b1 = lasso_constraints(lam, X_Mc, X_M, E_M, E_Mc, s)

    # box constraints
    b_box_plus = s_nu*np.ones(d) + suff_stat
    b_box_minus = s_nu*np.ones(d) - suff_stat
    
    A_box_plus = np.identity(d)
    A_box_minus = -np.identity(d)
    
    A_ineq = np.concatenate((A0_plus, A0_minus, A1, A_box_plus, A_box_minus))
    b_ineq = np.concatenate((b0_plus, b0_minus, b1, b_box_plus, b_box_minus))
    
    removed_indices = indices_to_skip

    # check which variables cannot enter or exist in neighboring polytopes
    cant_enter, cant_exit = redundancy_check(lam, X_Mc, X_M, suff_stat, s, M, Mc, s_nu)
    
    [removed_indices.append(i) for i in range(len(Mc)) if cant_enter[i]==1 and i not in removed_indices]
    [removed_indices.append(len(Mc) + i) for i in range(len(Mc)) if cant_enter[i]==1 and (len(Mc) + i) not in removed_indices]
    [removed_indices.append(2*len(Mc) + i) for i in range(len(M)) if cant_exit[i]==1 and (2*len(Mc) + i) not in removed_indices]
    
    relevant_indices = redundant_constr_LP(A_ineq, b_ineq, removed_indices, d)
            
    return relevant_indices

"""
    Finding neighboring polyhedra to skip
"""
def neighboring_polytopes_to_skip(M, s, Mc, M_s_discovered):
    redundant_constraints = []
    for j in range(len(M)):
        pair = [sorted(list(set(M) - set([M[j]]))), np.delete(s, j)]
        if not all([pair[0] != M_s_discovered[k][0] or any(pair[1] != M_s_discovered[k][1]) for k in range(len(M_s_discovered))]):
            redundant_constraints.append(2*len(Mc) + j)
    for j in range(len(Mc)):
        new_M = M + [Mc[j]]
        new_s = np.concatenate((s,[1]))
        new_s_sorted = np.array([new_s[k] for k in np.argsort(new_M)])
        pair = [sorted(new_M), new_s_sorted]
        if not all([pair[0] != M_s_discovered[k][0] or any(pair[1] != M_s_discovered[k][1]) for k in range(len(M_s_discovered))]):
            redundant_constraints.append(j)
        new_s = np.concatenate((s,[-1]))
        new_s_sorted = np.array([new_s[k] for k in np.argsort(new_M)])
        pair = [sorted(new_M), new_s_sorted]
        if not all([pair[0] != M_s_discovered[k][0] or any(pair[1] != M_s_discovered[k][1]) for k in range(len(M_s_discovered))]):
            redundant_constraints.append(len(Mc) + j)
    return redundant_constraints

"""
    All LASSO models and signs in ell-infinity box
"""
def all_LASSO_models_and_signs_in_box(X, y, lam, s_nu, M_y, s_y, max_models = 2000):
    
    n, d = X.shape

    M_s_todo = [[M_y, s_y]]
    M_s_done = []

    while M_s_todo:

        # take some model-sign pair that's not done
        [M, s] = M_s_todo.pop()

        if not M:
            M_s_done.append([M, s])
            continue

        Mc = sorted(list(set(range(d)) - set(M)))
        
        # find models that have already been added to queue and can thus be skipped
        indices_to_skip = neighboring_polytopes_to_skip(M, s, Mc, M_s_done + M_s_todo)
        
        # identify neighboring faces
        relevant_indices = lasso_minimal_polyhedron(X, X.T @ y, lam, M, s, s_nu, indices_to_skip)

        neighboring_M_and_s = []

        # identify neighboring models and signs
        for ind in relevant_indices:
            if ind < len(Mc):
                new_M = M + [Mc[ind]]
                new_s = np.concatenate((s,[1]))
                new_M_sorted = sorted(new_M)
                new_s_sorted = np.array([new_s[k] for k in np.argsort(new_M)])
                neighboring_M_and_s.append([new_M_sorted, new_s_sorted])
            elif ind < 2*len(Mc):
                new_M = M + [Mc[ind-len(Mc)]]
                new_s = np.concatenate((s,[-1]))
                new_M_sorted = sorted(new_M)
                new_s_sorted = np.array([new_s[k] for k in np.argsort(new_M)])
                neighboring_M_and_s.append([new_M_sorted, new_s_sorted])
            elif ind < d + len(Mc):
                M_copy = M.copy()
                M_copy.pop(ind-2*len(Mc))
                neighboring_M_and_s.append([M_copy, np.delete(s, ind-2*len(Mc))])
                
        M_s_done.append([M, s])
        new_M_s = [pair for pair in neighboring_M_and_s if all([pair[0] != M_s_done[k][0] or any(pair[1] != M_s_done[k][1]) for k in range(len(M_s_done))]) and all([pair[0] != M_s_todo[k][0] or any(pair[1] != M_s_todo[k][1]) for k in range(len(M_s_todo))])]
        M_s_todo.extend(new_M_s)
        
        
        if len(M_s_todo) > max_models:
            powset = list(powerset(range(d)))[1:]
            return [[powset[i], []] for i in (range(len(powset)))]
        
    return M_s_done

