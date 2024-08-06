"""
wocca.py
Core WOCCA algorithm

Version 231219.01
Yichao Li
"""

import numpy as np
import scipy.linalg as lin
from scipy import optimize

torch_loaded = False

# %% utilities

def start_torch():
    global torch, torch_loaded, gpu_available, device
    
    if not torch_loaded:
        import torch
        gpu_available = torch.cuda.is_available()
        device = torch.device("cuda:0" if gpu_available else "cpu")
        torch_loaded = True
    print("Using GPU:", gpu_available)
    return device

def norm(x):
    return x.dot(np.conj(x)).real

def sqnorm(x):
    return np.sqrt(x.dot(np.conj(x)).real)

def complex_wrapper(fun):
    def _fun(x_real):
        return fun(x_real[ : : 2] + 1j * x_real[1 : : 2])
    
    return _fun

def complex_decoder(x):
    y = np.zeros([len(x) * 2])
    y[ : : 2] = np.real(x)
    y[1 : : 2] = np.imag(x)
    
    return y

def w(v1, v2):
    x = np.real(v1)
    y = np.imag(v1)
    z = np.real(v2)
    w = np.imag(v2)
    
    return 4 * (x.dot(z) * y.dot(w) - y.dot(z) * x.dot(w))

def purify(v):
    v = v.copy()
    v /= sqnorm(v)
    if w(v, v) >= 1 - 1e-6:
        return v
    
    c = np.conj(v).dot(np.conj(v))
    t = (-1 + np.sqrt(max(0, 1 - np.real(c * np.conj(c))))) / c
    
    r = v + t * np.conj(v)
    r_norm = r.dot(np.conj(r))
    
    if r_norm > 1e-6:
         return r / np.sqrt(r_norm)
    return []

# %% CPCA with weights

def cpca(data, weights = None):
    if weights is None:
        cov = data.T @ np.conj(data)
        evl, evc = lin.eigh(cov)
        return evc.T[ : : -1]
    
    cov = (data.T * weights) @ np.conj(data)
    evl, evc = lin.eigh(cov)
    return evc.T[np.argsort(-np.abs(evl))]

# %% Tools for rank-2 skew symmetric matrices and their covariances

sqrt2 = np.sqrt(2)
def skew_matrix(alpha):
    tmp = alpha.imag.reshape([-1, 1]) @ alpha.real.reshape([1, -1])
    return sqrt2 * (tmp - tmp.T)

def upper_triangle(n_v):
    mask = ~np.tri(n_v, dtype = bool).flatten()
    def _fun(sm):
        return sm.flatten()[mask]
    
    return _fun

def get_s_matrix(data, weights = None, use_torch = False):
    if use_torch:
        device = start_torch()
        data = torch.tensor(data).to(device)
        if not (weights is None):
            weights = torch.tensor(weights).to(device)
    
    if weights is None:
        tmp = data.imag.T @ data.real
    else:
        tmp = data.imag.T @ (data.real.T * weights).T
    mat_s = sqrt2 * (tmp - tmp.T)
    
    if use_torch:
        mat_s = mat_s.cpu().detach().numpy()
    
    return mat_s

def get_b_matrix(data, tri_mask, weights = None, use_torch = False):
    n_v = data.shape[1]
    b = np.zeros([n_v * (n_v - 1) // 2] * 2)
    
    if use_torch:
        device = start_torch()
        data = torch.tensor(data).to(device)
        b = torch.tensor(b).to(device)
    
    if weights is None:
        for v in data:
            tmp_beta = tri_mask(skew_matrix(v))
            b += tmp_beta.reshape([-1, 1]) @ tmp_beta.reshape([1, -1])
    else:
        for i in range(len(data)):
            tmp_beta = tri_mask(skew_matrix(data[i]))
            b += tmp_beta.reshape([-1, 1]) @ (tmp_beta.reshape([1, -1]) * weights[i])
    
    if use_torch:
        b = b.cpu().detach().numpy()
    
    return b

def reduce_s_matrix(s, vi):
    tmp_s = skew_matrix(vi)
    
    return s - tmp_s * (tmp_s * s).sum() / (tmp_s ** 2).sum()

def reduce_b_matrix(b, vi, tri_mask):
    tmp_b = tri_mask(skew_matrix(vi))
    tmp_b = tmp_b.reshape([-1, 1]) @ tmp_b.reshape([1, -1])
    
    return b - tmp_b * (tmp_b * b).sum() / (tmp_b ** 2).sum()

# %% Base optimization algorithm of weakly orthogonal decomposition problems

def opt_wo_comps(solver, existing_comps = [], n_components = 3, max_iter = 1000, \
                 backtrack = True, aggressive_backtrack = 0, backtrack_max_iter = 10, \
                 unconstrained = False, verbose = False):
    def norm_constraint(x):
        return (x ** 2).sum() - 1
    
    def norm_constraint_jac(x):
        return 2 * x
    
    @complex_wrapper
    def wo_constraint(alpha):
        return np.abs(comps @ np.conj(alpha)) ** 2 - np.abs(comps @ alpha) ** 2
    
    @complex_wrapper
    def wo_constraint_jac(alpha):
        jac = np.zeros([len(comps), solver.n_v * 2])
        jac[ : , : : 2] = ((comps.imag @ alpha.imag) * comps.real.T - (comps.real @ alpha.imag) * comps.imag.T).T
        jac[ : , 1 : : 2] = ((comps.real @ alpha.real) * comps.imag.T - (comps.imag @ alpha.real) * comps.real.T).T
        
        return jac * 4
    
    def contrast(x):
        return solver.objective(x) * np.exp(-(((x ** 2).sum() - 1) ** 2) - (wo_constraint(x) ** 2).sum())
    
    if backtrack:
        backtrack_count = 0
    
    x0_flag = False
    scores = np.zeros([n_components])
    n_exist = len(existing_comps)
    i = n_exist
    if n_exist == 0:
        comps = np.zeros([0, solver.n_v], dtype = complex)
    else:
        comps = np.array(existing_comps).reshape([-1, solver.n_v])
        scores[ : i] = [-solver.objective(complex_decoder(comp_i)) for comp_i in comps]
    while i < n_components:
        if i != 0:
            solver.update(comps[i - 1])
        
        if not x0_flag:
            x0 = solver.initial()
        
        if (i == 0) or unconstrained:
            constraints = optimize.NonlinearConstraint(norm_constraint, 0, 0, norm_constraint_jac)
        else:
            constraints = [optimize.NonlinearConstraint(norm_constraint, 0, 0, norm_constraint_jac), \
                           optimize.NonlinearConstraint(wo_constraint, np.zeros([i]), np.zeros([i]), wo_constraint_jac)]
        res = optimize.minimize(contrast, x0, method = "trust-constr", constraints = constraints, \
                                options = {"maxiter": max_iter, "disp": verbose})
        scores[i] = -solver.objective(res.x)
        
        print("Iter", i + 1, "/", n_components)
        print("No. of iters", res.nit)
        print("Score", scores[i], "\n")
        
        if scores[i] > scores[max(n_exist, i - 1)] + 1e-6:
            print("Score not decreasing")
            if backtrack:
                if backtrack_count < backtrack_max_iter:
                    backtrack_count += 1
                    i0 = i
                    while scores[i0] > scores[i - 1] + 1e-6:
                        i -= 1
                        if i == n_exist:
                            break
                    i = max(n_exist, i - aggressive_backtrack)
                    comps = comps[ : i]
                    
                    solver.update(comps, clean = True)
                    x0_flag = True
                    x0 = res.x.copy()
                    print("Backtrack to Iter", i + 1, "\n")
                    continue
                else:
                    print("Limit of backtracks exceeded, continue without more backtracks")
            print()
        
        comps = np.concatenate([comps, (res.x[ : : 2] + 1j * res.x[1 : : 2]).reshape([1, -1])])
        comps[i] /= sqnorm(comps[i])
        i += 1
        x0_flag = False
            
    return comps

# %% Vanilla WOCCA

class wocca_solver:
    reg_thres = 1e-6
    
    def __init__(self, data, b = None, existing_comps = [], use_torch = False):
        if b is None:
            self.b1 = get_b_matrix(data, upper_triangle(data.shape[1]), use_torch = use_torch)
        else:
            self.b1 = b.copy()
            
        self.n_v = int(round((1 + np.sqrt(1 + 8 * len(self.b1))) // 2))
        self.tri_mask = upper_triangle(self.n_v)
        
        for vi in existing_comps:
            self.b1 = reduce_b_matrix(self.b1, vi, self.tri_mask)
        self.b0 = self.b1.copy()
    
    def objective(self, x):
        alpha = x[ : : 2] + 1j * x[1 : : 2]
        tmp_alpha = self.tri_mask(skew_matrix(alpha))
        sqc = (tmp_alpha.reshape([1, -1]) @ self.b1 @ tmp_alpha)[0]
        
        norm_x = (x ** 2).sum()
        
        return -4 * sqc / max(norm_x ** 2, wocca_solver.reg_thres)
    
    def update(self, comps, clean = False):
        if len(comps.shape) == 1:
            c1 = comps.reshape([1, -1])
        else:
            c1 = comps
        
        if clean:
            self.b1 = self.b0.copy()
        for vi in c1:
            self.b1 = reduce_b_matrix(self.b1, vi, self.tri_mask)
    
    def initial(self):
        evl, evc = lin.eigh(self.b1)
        sk0 = np.zeros([self.n_v, self.n_v])
        sk0[~np.tri(self.n_v, dtype = bool)] = evc[ : , np.argmax(np.abs(evl))]
        sk0 -= sk0.T
        evl, evc = lin.eig(sk0)
        x0_c = evc[ : , np.argmax(np.abs(evl))]
        
        x0_c /= np.sqrt(norm(x0_c))
        return complex_decoder(x0_c)

def wocca(data, b = None, existing_comps = [], use_torch = False, **kwargs):
    """
    The fast WOCCA algorithm based on theory of rank-2 skew symmetric matrices.

    Parameters
    ----------
    data : m*n array of complex
        Phasemap data; can be set to None if valid b matrix is provided.
    b : (n^2)*(n^2) array of float, optional
        Covariance of skew matrices, could be obtained using the "get_b_matrix" function.
        Will be generated automatically if data is provided.
        The default is None.
    n_components : int, optional
        Number of next components of WOCCA decomposition. The default is 3.
    existing_comps : n_ec*n array of complex, optional
        If provided, will continue WOCCA algorithm with constraint that the next components are all weakly orthogonal to existing components.
        The default is [].
    max_iter : int, optional
        Maximum number of iteration for each component. The default is 1000.
    use_torch : bool, optional
        Whether to use torch and GPU (if available) to calculate b matrix. The default is False.
    backtrack : bool, optional
        Whether to backtrack decomposition results in case there is non-decreasing explained traveling energy.
        If set to True, optimization would be slower, but results would likely be more accurate for n_components > 5.
        The default is True.
    aggressive_backtrack : int, optional
        The number of aggressive backtrack steps, larger means slower but likely more accurate results. The default is 0.
    backtrack_max_iter : int, optional
        Maximum allowed number of backtracts. If reached, possibly incomplete results would be returned. The default is 10.
    unconstrained : bool, optional
        If set to True, remove the weak orthogonality constraint. The default is False.
    verbose : bool, optional
        Whether to print texts in scipy.optimize.minimize function. The default is False.

    Returns
    -------
    comps : n_components*n array of complex
        Results of WOCCA decompostion. In case of unfinished backtrack, the number of components could be less than n_components.

    """
    solver = wocca_solver(data, b, existing_comps, use_torch)
    return opt_wo_comps(solver, existing_comps, **kwargs)

# %% Energy-weighted WOCCA

class ew_wocca_solver:
    reg_thres = 1e-6

    def __init__(self, data, weights, bw = None, existing_comps = [], use_torch = False):
        if bw is None:
            self.bw1 = get_b_matrix(data, upper_triangle(data.shape[1]), weights = weights, use_torch = use_torch)
        else:
            self.bw1 = bw.copy()
        
        self.n_v = int(round((1 + np.sqrt(1 + 8 * len(self.bw1))) // 2))
        self.tri_mask = upper_triangle(self.n_v)
        
        for vi in existing_comps:
            self.bw1 = reduce_b_matrix(self.bw1, vi, self.tri_mask)
        self.bw0 = self.bw1.copy()
        
    def objective(self, x):
        alpha = x[ : : 2] + 1j * x[1 : : 2]
        tmp_alpha = self.tri_mask(skew_matrix(alpha))
        sqc_w = tmp_alpha.reshape([1, -1]) @ self.bw1 @ tmp_alpha
        
        norm_x = (x ** 2).sum()
        
        return -np.abs(4 * sqc_w / max(norm_x ** 2, wocca_solver.reg_thres))
    
    def update(self, comps, clean = False):
        if len(comps.shape) == 1:
            c1 = comps.reshape([1, -1])
        else:
            c1 = comps
        
        if clean:
            self.bw1 = self.bw0.copy()
        for vi in c1:
            self.bw1 = reduce_b_matrix(self.bw1, vi, self.tri_mask)
    
    def initial(self):
        evl, evc = lin.eigh(self.bw1)
        sk0 = np.zeros([self.n_v, self.n_v])
        sk0[~np.tri(self.n_v, dtype = bool)] = evc[ : , np.argmax(np.abs(evl))]
        sk0 -= sk0.T
        evl, evc = lin.eig(sk0)
        x0_c = evc[ : , np.argmax(np.abs(evl))]
        
        x0_c /= np.sqrt(norm(x0_c))
        x0 = complex_decoder(x0_c)
        
        if self.objective(x0) < 0:
            x0 = -x0
        return x0

def ew_wocca(data, weights, bw = None, existing_comps = [], use_torch = False, **kwargs):
    """
    Supervised WOCCA weighted on traveling energy (EW-WOCCA).
    
    Parameters
    ----------
    data : m*n array of complex
        Phasemap data; can be set to None if valid bw matrix is provided.
    weights : n array of float
        The weight (contrast or label values) that are to be maximally explained by variance of traveling energy.
        Can be set to None if valid bw matrix is provided.
    bw : (n^2)*(n^2) array of float, optional
        Weighted covariance of skew matrices, could be obtained using the "get_b_matrix" function.
        Will be generated automatically if data and weights are provided.
        The default is None.
    n_components : int, optional
        Number of next components of WOCCA decomposition. The default is 3.
    existing_comps : n_ec*n array of complex, optional
        If provided, will continue WOCCA algorithm with constraint that the next components are all weakly orthogonal to existing components.
        The default is [].
    max_iter : int, optional
        Maximum number of iteration for each component. The default is 1000.
    use_torch : bool, optional
        Whether to use torch and GPU (if available) to calculate bw matrix. The default is False.
    backtrack : bool, optional
        Whether to backtrack decomposition results in case there is non-decreasing explained traveling energy.
        If set to True, optimization would be slower, but results would likely be more accurate for n_components > 5.
        The default is False.
    aggressive_backtrack : int, optional
        The number of aggressive backtrack steps, larger means slower but likely more accurate results. The default is 0.
    backtrack_max_iter : int, optional
        Maximum allowed number of backtracts. If reached, possibly incomplete results would be returned. The default is 10.
    unconstrained : bool, optional
        If set to True, remove the weak orthogonality constraint. The default is False.
    verbose : bool, optional
        Whether to print texts in scipy.optimize.minimize function. The default is False.

    Returns
    -------
    comps : n_components*n array of complex
        Results of eWOCCA decompostion. In case of unfinished backtrack, the number of components could be less than n_components.

    """
    solver = ew_wocca_solver(data, weights, bw, existing_comps, use_torch)
    return opt_wo_comps(solver, existing_comps, **kwargs)

# %% Contrast-weighted WOCCA

def cw_wocca(data, weights, sw = None, n_components = 3, existing_comps = [], use_torch = False):
    """
    Supervised WOCCA weighted on conjugate contrast (CW-WOCCA).
    
    Parameters
    ----------
    data : m*n array of complex
        Phasemap data; can be set to None if valid sw matrix is provided.
    weights : n array of float
        The weight (contrast or label values) that are to be maximally explained by variance of traveling orientations.
        Can be set to None if valid bw matrix is provided.
    sw : n*n array of float, optional
        Weighted sum of skew matrices, could be obtained using the "get_s_matrix" function.
        Will be generated automatically if data and weights are provided.
        The default is None.
    n_components : int, optional
        Number of next components of WOCCA decomposition. The default is 3.
    existing_comps : n_ec*n array of complex, optional
        If provided, will continue WOCCA algorithm with constraint that the next components are all weakly orthogonal to existing components.
        The default is [].
    use_torch : bool, optional
        Whether to use torch and GPU (if available) to calculate sw matrix. The default is False.

    Returns
    -------
    comps : n_components*n array of complex
        Results of oWOCCA decompostion. In case of unfinished backtrack, the number of components could be less than n_components.

    """
    if sw is None:
        sw1 = get_s_matrix(data, weights = weights, use_torch = use_torch)
    else:
        sw1 = sw.copy()
    
    for vi in existing_comps:
        sw1 = reduce_s_matrix(sw1, vi)
    
    n_v = len(sw1)
    
    evl, evc = lin.eig(1j * sw1)
    order = np.argsort(-evl.real)[ : n_v // 2]
    
    return evc.T[order[ : n_components]]

# %% PCA preprocess

def pca_decomp(u, n_dim = 20, basis = None):
    """
    Dimensionality reduction of real and imaginary parts for phasemap data, intended for accelerating WOCCA algorithm.

    Parameters
    ----------
    u : m*n data of complex
        Original phasemap data.
    n_dim : int, optional
        Number of dimensions, preferably at least twice the number of components in WOCCA. The default is 20.
    basis : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    v : m*n_dim data of complex
        The "phasemap" data with dimensionality reduced.
    basis : n_dim*n data of real
        The PCA basis matrix, should be saved for data recovery.

    """
    if basis is None:
        basis = lin.svd(np.concatenate([u.real, u.imag]), full_matrices = False)[2][ : n_dim]
    
    return u @ basis.T, basis

def pca_recover(v, basis):
    """
    Recover dimensionality reduced data.

    Parameters
    ----------
    v : m*n_dim data of complex
        The "phasemap" data with dimensionality reduced.
    basis : n_dim*n data of real
        The PCA basis matrix, should be saved for data recovery.

    Returns
    -------
    u : m*n data of complex
        The recovered phasemap data, close but not identical to the original data.

    """
    return v @ basis

# Intended usage:
#   v, basis = pca_decomp(u, 20)
#   v_comps = wocca(v, n_components = 5)
#   comps = pca_recover(v_comps, basis)
# Which approximates the result of:
#   comps = wocca(u, n_components = 5)

# %% WOCCA decomposition

def projection(v, ws):
    """
    Projecting phasemap data on WOCCA space.

    Parameters
    ----------
    v : m*n array of complex
        Phasemap data.
    ws : n_components*n array of complex
        WOCCA basis (preferably from WOCCA algorithm).

    Returns
    -------
    coords : m*n_components array of float
        Coordination of each phasemap in WOCCA space.

    """
    v1 = np.abs(v.dot(np.conj(ws.T))) ** 2
    v2 = np.abs(v.dot(ws.T)) ** 2
    
    return v1 - v2

def score(v, ws):
    """
    Calculate explained traveling energy (i.e. WOCCA score) of phasemap data and WOCCA basis.

    Parameters
    ----------
    v : m*n array of complex
        Phasemap data.
    ws : n_components*n array of complex
        WOCCA basis (preferably from WOCCA algorithm).

    Returns
    -------
    scores : n_components array of float
        Explained traveling energy of each component.

    """
    return (projection(v, ws) ** 2).sum(axis = 0)

def total_energy(v):
    """
    Calculate total traveling energy of phasemap data.

    Parameters
    ----------
    v : m*n array of complex
        Phasemap data.

    Returns
    -------
    total_score: float
        Total traveling energy of phasemap data.

    """
    v1 = (np.abs(v) ** 2).sum(axis = -1) ** 2
    v2 = np.abs((v ** 2).sum(axis = -1)) ** 2
    
    return (v1 - v2).sum()