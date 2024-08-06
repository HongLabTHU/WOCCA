"""
ring.py
Phasemap manifold and traveling wave ring

Version 231010.01
Yichao Li
"""

import numpy as np
from scipy import optimize, interpolate

import core
import templates as tmpl

# Calculate density of all phasemap samples, based on number of other samples in neighbour
# Assume aff_thres is sorted in ascending order
def batch_density(v, batch_size = 100, aff_thres = 0.93):
    if type(aff_thres) == float:
        aff_thres = np.array([aff_thres])
    
    n_v = len(v)
    n_batch = int(np.ceil(len(v) / batch_size))
    res = np.zeros([n_v, len(aff_thres)], dtype = int)
    for i in range(n_batch):
        j0 = i * batch_size
        j1 = min(n_v, (i + 1) * batch_size)
        aff = core.topomap_affinity(v[j0 : j1], v, False, False)
        for j in range(j1 - j0):
            tmp = aff[j]
            for k in range(len(aff_thres)):
                tmp = tmp[tmp >= aff_thres[k]]
                res[j + j0, k] = len(tmp) - 1
    
    return res

def get_density_fit(v, batch_size = 100, neighbour_aff = 0.8, intercept_aff = 0.9, verbose = False):
    log_intercept_dist = np.log(np.arccos(intercept_aff))
    
    n_v = len(v)
    n_batch = int(np.ceil(len(v) / batch_size))
    res = np.zeros([n_v, 3])
    for i in range(n_batch):
        if type(verbose) == int:
            if i % verbose == 0:
                print(i, "/", n_batch)
        j0 = i * batch_size
        j1 = min(n_v, (i + 1) * batch_size)
        aff = core.topomap_affinity(v[j0 : j1], v, False, False)
        for j in range(j1 - j0):
            tmp = aff[j]
            tmp = tmp[tmp >= neighbour_aff]
            tmp = tmp[tmp <= 1 - 1e-4]
            tmp = np.sort(np.arccos(tmp))
            
            if len(tmp) < 2:
                res[j + j0] = -1, -1, len(tmp)
            else:
                tmp_fit = np.polyfit(np.log(tmp), np.log(np.arange(len(tmp)) + 1), deg = 1)
                res[j + j0] = tmp_fit[0], tmp_fit[1] + log_intercept_dist * tmp_fit[0], len(tmp)    
    
    return res

# Transforming geodesic length to linear interpolation fraction
def geodesic_l2k(r0, l):
    sl = np.sin(l)
    c2l = np.cos(2 * l)
    delta = max(0, sl ** 2 + (c2l + r0) / (1 - r0))
    
    return (-sl ** 2 + sl * np.sqrt(delta)) / (c2l + r0)

# Transforming linear interpolation fraction to geodesic length
def geodesic_k2l(r0, k):
    return np.arccos((1 - k + k * r0) / np.sqrt(1 - 2 * k * (1 - k) * (1 - r0)))

# Interpolate between two phasemaps based on fraction of geodesic length in between
def geodesic_interpolate(map0, map1, lk):
    t_aff = map0.dot(np.conj(map1))
    r0 = min(1, np.abs(t_aff))
    if r0 < 1e-4:
        return []
    map1_r = map1 * np.exp(1j * np.angle(t_aff))
    
    lmax = np.arccos(r0)
    k = geodesic_l2k(r0, lk * lmax)
    t_res = (1 - k) * map0 + k * map1_r
    t_res /= np.sqrt(np.real(t_res.dot(np.conj(t_res))))
    
    return t_res

# Distance from one phasemap (g) to geodesic between two other phasemaps (p and q)
# If affinity == True, returns squared affinity
def geodist(cpq, cpg, cqg, affinity = True):
    t1f2 = (cqg - cpg) ** 2
    t1f1 = 2 * cpg * (cqg - cpg)
    t1f0 = cpg ** 2
    t2f2 = 2 * (1 - cpq)
    t2f1 = -2 * (1 - cpq)
    
    if affinity:
        def _fun(k):
            return (t1f2 * (k ** 2) + t1f1 * k + t1f0) / (t2f2 * (k ** 2) + t2f1 * k + 1)
    else:
        def _fun(k):
            tmp = (t1f2 * (k ** 2) + t1f1 * k + t1f0) / (t2f2 * (k ** 2) + t2f1 * k + 1)
            return np.arccos(np.sqrt(max(0, tmp)))
    
    return _fun

def get_geodesic_projs(data, nodes, verbose = False):
    n_data = len(data)
    n_nodes = len(nodes)
    tl = np.zeros([n_nodes, 2], dtype = int)
    for i in range(n_nodes - 1):
        tl[i] = i, i + 1
    tl[-1] = n_nodes - 1, 0
    
    opt_kl = np.zeros([n_data, n_nodes])
    opt_aff = np.zeros_like(opt_kl)
    l_nodes = np.zeros([n_nodes])
    td_aff = core.topomap_affinity(data, nodes, False, False)
    tn_aff = core.topomap_affinity(nodes, [], False, False)
    
    for j in range(n_nodes):
        tr0 = tn_aff[tl[j, 0], tl[j, 1]]
        l_nodes[j] = np.arccos(tr0)
        for i in range(n_data):
            if verbose and (i % 10000 == 0):
                print(i, "/", n_data, ",", j, "/", n_nodes)
            fun0 = geodist(tr0, td_aff[i, tl[j, 0]], td_aff[i, tl[j, 1]])
            fun = lambda k: -fun0(k)
            tmp_k = min(1, max(0, optimize.minimize_scalar(fun).x))
            opt_kl[i, j] = geodesic_k2l(tr0, tmp_k) / l_nodes[j]
            opt_aff[i, j] = np.sqrt(fun0(tmp_k))
    
    seg_label = np.argmax(opt_aff, axis = 1)
    seg_pos = np.array([opt_kl[i, seg_label[i]] for i in range(n_data)])
    seg_aff = np.array([opt_aff[i, seg_label[i]] for i in range(n_data)])
    
    return seg_label, seg_pos, seg_aff, l_nodes

def refine_geodesic_nodes(data, nodes, verbose = False):
    n_data, n_map = data.shape
    n_nodes = len(nodes)
    seg_label, seg_pos = get_geodesic_projs(data, nodes, verbose)[ : 2]
    
    seg_centers = np.zeros([n_nodes, n_map], dtype = complex)
    
    for j in range(n_nodes):
        seg_inds = np.argwhere(seg_label == j).reshape([-1])
        weights = (0.5 - np.abs(seg_pos[seg_inds] - 0.5)) ** 2
        seg_centers[j] = tmpl.get_centroid((data[seg_inds].T * weights).T)
    
    return seg_centers

def iterate_geodesic_nodes(data, nodes, n_iter = 2, n_sub_iter = 1, verbose = False, record_steps = False):
    n_map = data.shape[1]
    nodes_cur = nodes.copy()
    
    if record_steps:
        rec = []

    for i in range(n_iter):
        n_nodes = len(nodes_cur)
        for j in range(n_sub_iter):
            nodes_add = refine_geodesic_nodes(data, nodes_cur, verbose)
            if record_steps:
                rec.append(nodes_add.copy())
            nodes_rep = refine_geodesic_nodes(data, nodes_add, verbose)
            if record_steps:
                rec.append(nodes_rep.copy())
        nodes_cur = np.zeros([n_nodes * 2, n_map], dtype = complex)
        nodes_cur[0] = nodes_rep[-1]
        nodes_cur[-1] = nodes_add[-1]
        for j in range(1, n_nodes):
            nodes_cur[j * 2 - 1] = nodes_add[j - 1]
            nodes_cur[j * 2] = nodes_rep[j - 1]
    
    if record_steps:
        return nodes_cur, rec
    return nodes_cur

def get_ring_locations(seg_label, seg_pos, seg_aff, l_nodes):
    l_tot = l_nodes.sum()
    
    return np.array([(l_nodes[ : seg_label[i]].sum() + seg_pos[i]) / l_tot for i in range(len(seg_label))])

def get_spline_ring_locations(data, nodes, verbose = False):
    n_data = len(data)
    n_nodes = len(nodes)
    tr = np.zeros([n_nodes + 1], dtype = int)
    tr[ : -1] = np.arange(n_nodes)
    
    td_aff = core.topomap_affinity(data, nodes, False, False)
    tn_aff = core.topomap_affinity(nodes, [], False, False)
    l_nodes = np.array([np.arccos(tn_aff[tr[j], tr[j + 1]]) for j in range(n_nodes)])
    xs = np.zeros([n_nodes + 1])
    for i in range(n_nodes):
        xs[i + 1] = xs[i] + l_nodes[i]
    
    ring_pos = np.zeros([n_data])
    ring_aff = np.zeros([n_data])
    
    for i in range(n_data):
        if verbose and (i % 10000 == 0):
            print(i, "/", n_data)
        fun = interpolate.CubicSpline(xs, -td_aff[i, tr], bc_type = "periodic", extrapolate = "periodic")
        ring_pos[i] = optimize.minimize_scalar(fun, bounds = [-xs[-1], 2 * xs[-1]], method = "bounded").x
        ring_aff[i] = -fun(ring_pos[i])
    
    return ring_pos, ring_aff, l_nodes

def spline_ring_interpolator(nodes):
    n_nodes = len(nodes)
    tr = np.zeros([n_nodes + 1], dtype = int)
    tr[ : -1] = np.arange(n_nodes)
    
    tn_aff = core.topomap_affinity(nodes, [], False, False)
    l_nodes = np.array([np.arccos(tn_aff[tr[j], tr[j + 1]]) for j in range(n_nodes)])
    
    l_sum = l_nodes.sum()
    l_intervals = np.concatenate([[0], np.cumsum(l_nodes)])
    
    def _fun(k):
        l = (k % 1) * l_sum
        i = max(0, np.searchsorted(l_intervals, l) - 1)
        
        return geodesic_interpolate(nodes[tr[i]], nodes[tr[i + 1]], (l - l_intervals[i]) / l_nodes[i])
    
    return _fun

def interpolate_continuous_movement(rp, ra, ts, dt = 0.02):
    tmin = int(np.ceil(ts[0] / dt))
    tmax = int(np.floor(ts[-1] / dt)) + 1
    
    rp_interp = np.zeros([tmax - tmin])
    ra_interp = np.zeros_like(rp_interp)
    interp = interpolate.CubicSpline(ts, np.unwrap(rp, period = 1))
    rp_interp = interp(np.arange(tmin, tmax) * dt)
    interp = interpolate.CubicSpline(ts, ra)
    ra_interp = interp(np.arange(tmin, tmax) * dt)
    
    return rp_interp, ra_interp, tmin, tmax