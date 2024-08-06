"""
gradient.py
Computation of phase gradient vectors

Version 240410.01
Yichao Li
"""

import numpy as np
import mne
import scipy.linalg as lin
from scipy import optimize

def gradient_product(g1, g2):
    return (g1 * g2).sum()

# Circular linear regression, using gradient descent
# v ~ ax + b
def circular_lr(v, x, intercept = True, weights = None, init_mode = "zero", debug = False):
    def _loss(p):
        dv = (v - x @ p + np.pi) % (np.pi * 2) - np.pi
        if no_weights:
            return (dv ** 2).sum()
        return ((dv ** 2) * weights).sum()

    if intercept:
        x = np.concatenate([x.T, np.ones([len(x)])]).T
    n = x.shape[1]
    no_weights = weights is None
    
    if type(init_mode) == str:
        if init_mode == "ols":
            p0 = lin.inv(x.T @ x) @ x.T @ v
        else:
            p0 = np.zeros([n])
    else:
        p0 = init_mode.copy()

    opt_res = optimize.minimize(_loss, p0)

    if debug:
        return opt_res, _loss
    
    if intercept:
        return opt_res.x[ : -1], opt_res.x[-1]
    return opt_res.x

def fit_local_gradient(pos0, pos, v0, v, calc_r2 = False):
    x = pos - pos0
    y = (np.angle(v) - np.angle(v0) + np.pi) % (np.pi * 2) - np.pi
    w = -np.cos(y)
    w = 1 - (np.abs(w) + w) / 2
    
    p = circular_lr(y, x, intercept = False, weights = w, debug = False)

    if calc_r2:
        dv = (y - x @ p + np.pi) % (np.pi * 2) - np.pi
        return -p, 1 - (dv ** 2).sum() / (y ** 2).sum()
    return -p

def get_gradient(raw_info, v, calc_r2 = False, pos = None, adj = None):
    """
    Get raw gradient vectors from a complex-valued phase map
    
    Parameters:
        raw_info : instance of mne.Info
        v : 1D array of complex numbers
            A phase map in the form of a complex array, e.g. can be the outcome of Hilbert transformation
        calc_r2 : boolean
            If true, returns "r2" after "grad"
        pos : 2D n*2 array
            If not empty, overrides "raw_info" in terms of channel layout
        adj : 2D n*n array
            If not empty, overrides "raw_info" in terms of channel adjacency
    
    Returns:
        grad : 2D n*2 array
            Phase gradient vectors
        r2 : 1D array
            The determination coefficient of local gradient fit for each channel
    """
    n = len(v)
    if pos is None:
        pos = mne.channels.make_eeg_layout(raw_info, width = 0, height = 0).pos[ : , : 2].copy() - 0.5
    grad = np.zeros([n, 2])
    if calc_r2:
        r2 = np.zeros([n])
    
    if adj is None:
        adj = mne.channels.find_ch_adjacency(raw_info, "eeg")[0].toarray()
    for i in range(n):
        local = np.array(adj[i, : ], dtype = bool)
        local[i] = False
        t = fit_local_gradient(pos[i, : ], pos[local, : ], v[i], v[local], calc_r2)
        if calc_r2:
            grad[i, : ], r2[i] = t
        else:
            grad[i, : ] = t
    
    if calc_r2:
        return grad, r2
    return grad

def gradient_transform(raw_info, pos = None, adj = None):
    """
    Generate a function to efficiently calculate phase gradient of a given channel layout

    Parameters
    ----------
    raw_info : instance of mne.Info
        The EEG layout object, can be set to None if given appropriate pos and adj
    pos : 2D n*2 array, optional
        If not empty, overrides "raw_info" in terms of channel layout
    adj : 2D n*n array, optional
        If not empty, overrides "raw_info" in terms of channel adjacency

    Returns
    -------
    _fun : function(v, calc_r2 = False) -> grad, (r2)
        function mapping a phasemap vector to its phase gradient, also calcuating r^2 optionally
    """
    if pos is None:
        pos = mne.channels.make_eeg_layout(raw_info, width = 0, height = 0).pos[ : , : 2].copy() - 0.5
    if adj is None:
        adj = mne.channels.find_ch_adjacency(raw_info, "eeg")[0].toarray()
    n = len(pos)
    
    local = []
    for i in range(n):
        tmp_l = np.array(adj[i, : ], dtype = bool)
        tmp_l[i] = False
        local.append(np.argwhere(tmp_l).reshape([-1]))
    
    def _fun(v, calc_r2 = False):
        grad = np.zeros([n, 2])
        if calc_r2:
            r2 = np.zeros([n])
        
        for i in range(n):
            t = fit_local_gradient(pos[i, : ], pos[local[i], : ], v[i], v[local[i]], calc_r2)
            if calc_r2:
                grad[i, : ], r2[i] = t
            else:
                grad[i, : ] = t
        
        if calc_r2:
            return grad, r2
        return grad
    
    return _fun

def raw_winding_number(info, pos = None, adj = None):
    def _fun(v):
        phase = np.angle(v)
        wn = np.zeros([n])
        for i in range(n):
            for j in range(len(local[i])):
                k = (j - 1) if j > 0 else (len(local[i]) - 1)
                wn[i] += (phase[local[i][j]] - phase[local[i][k]] + np.pi) % (np.pi * 2) - np.pi
        
        return wn / (np.pi * 2)
    
    if pos is None:
        pos = mne.channels.make_eeg_layout(info, width = 0, height = 0).pos[ : , : 2].copy() - 0.5
    if adj is None:
        adj = mne.channels.find_ch_adjacency(info, "eeg")[0].toarray()
    n = len(pos)
    
    local = []
    for i in range(n):
        tmp_l = np.array(adj[i, : ], dtype = bool)
        tmp_l[i] = False
        local.append(np.argwhere(tmp_l).reshape([-1]))
        tmp_pos_diff = pos[local[-1]] - pos[i]
        tmp_angle = -np.angle(tmp_pos_diff[ : , 0] + 1j * tmp_pos_diff[ : , 1])
        local[-1] = local[-1][np.argsort(tmp_angle)]

    return _fun

def detect_phase_discontinuity(info, pos = None, adj = None):
    def _fun(v, thres = np.pi / 2, dist_thres = 0.2):
        res = []
        res_pos = []
        res_pd = []
        phase = np.angle(v)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if adj[i, j]:
                    tmp = np.pi - np.abs((phase[i] - phase[j]) % (np.pi * 2) - np.pi)
                    if tmp >= thres:
                        if ((pos[i] - pos[j]) ** 2).sum() <= dist_thres:
                            res.append([i, j])
                            res_pos.append([pos[i], pos[j]])
                            res_pd.append(tmp)

        return np.array(res), np.array(res_pos), np.array(res_pd)

    if pos is None:
        pos = mne.channels.make_eeg_layout(info, width = 0, height = 0).pos[ : , : 2].copy() - 0.5
    if adj is None:
        adj = mne.channels.find_ch_adjacency(info, "eeg")[0].toarray()
    n = len(pos)

    return _fun

def fit_rotation_center(info, v, target, init = [0, 0], weight_scale = [0.1, 0.5], pos = None, debug = False):
    """
    Find rotation centers of given winding number

    Parameters
    ----------
    info : instance of mne.Info
        Info object of EEG.
    v : array [n_ch] of complex
        Phasemap.
    target : int or float
        Target winding number; positive stands for counterclockwise.
    init : array [2] of float, optional
        Initial value of rotation center for gradient descent. The default is [0, 0].
    weight_scale : array [2] of float, optional
        Parameters for double Gaussian weighting, to reduce influence of channels that are too close or too far away.
        The default is [0.1, 0.5].
    pos : array [n_ch, 2] of float, optional
        If provided, overrides channel locations in info. The default is None.
    debug : bool, optional
        Whether to output more data for debug. The default is False.

    Returns
    -------
    p : array [2] of float
        The rotation center.
    loss : float
        The loss function, indicating deviation from target winding number.
    """
    def _loss(x, full_mode = False):
        weights = np.zeros([n])
        vels = np.zeros([n])
        for i in range(n):
            tmp_vec = pos[i] - x
            tmp_dir = np.array([-tmp_vec[1], tmp_vec[0]])
            tmp_d = (tmp_vec ** 2).sum()
            weights[i] = max(1e-6, (np.exp(-tmp_d / (weight_scale[1] ** 2)) - np.exp(-tmp_d / (weight_scale[0] ** 2))) * max(0, g_r2[i] - 0.5))
            vels[i] = tmp_dir.dot(g[i])

        dev = np.sqrt((((vels - target) ** 2) * weights).sum() / weights.sum())

        if full_mode:
            mean = (vels * weights).sum() / weights.sum()
            return weights, vels, mean, dev
        return dev

    init = np.array(init)
    if pos is None:
        pos = mne.channels.make_eeg_layout(info, width = 0, height = 0).pos[ : , : 2].copy() - 0.5
    n = len(pos)
    g, g_r2 = get_gradient(info, v, True, pos = pos)
    
    opt_res = optimize.minimize(_loss, init)

    if debug:
        return opt_res, _loss(opt_res.x, True)
    return opt_res.x, opt_res.fun

def translational_index(chid, grad):
    """
    Calculate translational index (gradient vector homogeneity) given a group of channels
    
    Parameters:
        chid : 1D array of booleans, or 1D array of integers
            Indices or mask of channels of interest, e.g. all channels adjacent to a central channel (including itself)
        grad : 2D n*2 array
            Phase gradient vectors
    
    Returns:
        vec : 2-tuple
            Average translational vector
        r : double
            Homogeneity of gradient vectors
    """
    if chid.dtype == bool:
        n_chid = chid.sum()
    else:
        n_chid = len(chid)
    s = grad[chid, : ]
    v_sum = s.sum(axis = 0)
    l_sum = (s ** 2).sum() * n_chid
    
    return v_sum / n_chid, np.sqrt((v_sum ** 2).sum() / l_sum)

def rotational_index(raw_info, chid, center_id, grad, pos = []):
    """
    Calculate rotational index for a single central channel given its surroundings
    
    Parameters:
        raw_info : instance of mne.Info
        chid : 1D array of booleans, or 1D array of integers
            Indices or mask of surrounding channels, e.g. all channels adjacent to a mediating channel that is ajdacent to the central channel
        center_id : integer
            Index of the central channel
        grad : 2D n*2 array
            Phase gradient vectors
        pos : 2D n*2 array
            If not empty, overrides "raw_info" in terms of channel layout
        
    Returns:
        ri_tan : double
            Tangential rotational index
        ri_rad : double
            Radial rotational index (not particularly useful)
    """
    if len(pos) == 0:
        pos = mne.channels.make_eeg_layout(raw_info, width = 0, height = 0).pos[ : , : 2].copy() - 0.5
    vec0 = pos[center_id] - pos[chid, : ]
    vec = (vec0.T / (vec0 ** 2).sum(axis = 1)).T
    
    s = grad[chid, : ].T.dot(vec)
    u = np.sqrt((vec ** 2).sum() * (grad[chid, : ] ** 2).sum())
    
    return (s[0, 1] - s[1, 0]) / u, (s[0, 0] + s[1, 1]) / u

def fit_global_indices(raw_info, pos = [], adj = []):
    """
    Generate a function to efficiently calculate translational and rotational indices

    Parameters
    ----------
    raw_info : instance of mne.Info
        The EEG layout object, can be set to None if given appropriate pos and adj
    pos : 2D n*2 array, optional
        If not empty, overrides "raw_info" in terms of channel layout
    adj : 2D n*n array, optional
        If not empty, overrides "raw_info" in terms of channel adjacency

    Returns
    -------
    _fun : function(v) -> trans_vec, trans_ind, rot_tan, rot_rad
        function mapping a phasemap vector to its translation vectors, translational indices, tangential rotational indices, and radial rotational indices
    """
    if len(pos) == 0:
        pos = mne.channels.make_eeg_layout(raw_info, width = 0, height = 0).pos[ : , : 2].copy() - 0.5
    if len(adj) == 0:
        adj = mne.channels.find_ch_adjacency(raw_info, "eeg")[0].toarray()
    n = len(pos)
    
    f_grad = gradient_transform(None, pos, adj)
    
    adj1 = np.array(adj, dtype = bool)
    adj2 = (adj1 @ adj1) & (~adj1)
    locs = []
    s_locs = []
    for i in range(n):
        local = np.argwhere(adj[i]).reshape([-1])
        locs.append(local)
        
        s_local = np.argwhere(adj2[i]).reshape([-1])
        s_locs.append(s_local)
    
    def _fun(v):
        g = f_grad(v, calc_r2 = False)
        
        trans_vec = np.zeros([n, 2])
        trans_ind = np.zeros([n])
        rot_tan = np.zeros([n])
        rot_rad = np.zeros([n])
        
        for i in range(n):
            trans_vec[i], trans_ind[i] = translational_index(locs[i], g)
            rot_tan[i], rot_rad[i] = rotational_index(None, s_locs[i], i, g, pos = pos)
            
        return trans_vec, trans_ind, rot_tan, rot_rad
    
    return _fun

if __name__ == "__main__":
    pass