"""
core.py
Core functionalities of microstate-like data structure

Version 240201.01
Yichao Li
"""

import numpy as np
import scipy.linalg as lin
import scipy.signal as sig
import scipy.fftpack as fft
import scipy.optimize as opt
from scipy import stats

# gfp curve and peaks
gfp_peak_hib_default = 0.005

def get_gfp_peaks(data, srate, gfp_peak_hib = gfp_peak_hib_default):
    gfp = data.std(axis = 1)
    
    hib = max(1, int(round(gfp_peak_hib * srate)))
    n = len(gfp)
    
    res = np.zeros_like(gfp, dtype = int)
    k = 0
    for i in range(hib, n - hib):
        if gfp[i] == gfp[i - hib : i + hib + 1].max():
            res[k] = i
            k += 1
    
    return gfp, res[ : k]

# calculate affinity for real or complex maps
def batch_l2_normalize(x):
    l2 = np.sqrt((x * np.conj(x)).sum(axis = -1))
    
    return (x.T / l2).T

qaff = lambda x, y : x.dot(np.conj(y))

def topomap_affinity(_x, _y = [], normalize = True, with_phase = True):
    x = _x.copy()
    y = _y.copy()
    if normalize:
        x = batch_l2_normalize(x)
        if len(_y) != 0:
            y = batch_l2_normalize(y)
    if len(_y) == 0:
        y = x.copy()
    
    res = x.dot(np.conj(y.T))
    
    if with_phase:
        return res
    return np.abs(res)

# get inverse of covariance matrix for a microstate bases
def get_invcov(centroids):
    # return lin.inv(np.conj(centroids).dot(centroids.T))
    return lin.inv(centroids.dot(np.conj(centroids.T)))

# convert affinity to decomposition coefficients
def get_decomp(invcov, aff):
    return aff.dot(invcov)

# calculate normalized phase maps as principal eigenvectors
def get_phasemap(h, starting_phase = False):
    c = topomap_affinity(h.T, [], True, True)
    w, v = lin.eigh(c)
    
    if not starting_phase:
        return v[ : , np.argmax(np.abs(w))]
    
    vt = v[ : , np.argmax(np.abs(w))]
    diff = topomap_affinity(h[0, : ], vt, True, True)
    
    return vt * diff / np.abs(diff)

# matching sets of centroids
def affinity_matching(aff, by_dist = False, check_ambiguity = False):
    if by_dist:
        aff1 = aff.copy()
        aff1[aff1 > 1] = 1
        m = np.sqrt(2 - 2 * aff1)
    else:
        m = aff
    t0, t1 = opt.linear_sum_assignment(m, not by_dist)
    
    r0 = -np.ones([aff.shape[0]], dtype = int)
    r1 = -np.ones([aff.shape[1]], dtype = int)
    s = 0
    for i in range(len(t0)):
        r0[t0[i]] = t1[i]
        r1[t1[i]] = t0[i]
        s += m[t0[i], t1[i]]
    
    if check_ambiguity:
        flag = True
        for i in range(len(t0)):
            flag &= (aff[t0[i], t1[i]] == aff[t0[i]].max()) & (aff[t0[i], t1[i]] == aff[ : , t1[i]].max())
            if not flag:
                break
        return r0, r1, s, flag
    return r0, r1, s

# definition of data structures
class base_segment:
    def __init__(self, data, info = []):
        self.data = data.copy()
        self.info = info
    
    def __del__(self):
        del(self.data)
        del(self.info)
    
    def copy(self):
        return base_segment(self.data.copy(), self.info)
    
    def affinity(self, centroids, normalize = True, with_phase = True, peak_only = False):
        return topomap_affinity(self.data, centroids, normalize, with_phase)

class cms_segment(base_segment):
    def __init__(self, data, info, gfp, peaks):
        base_segment.__init__(self, data, info)
        self.gfp = gfp.copy()
        self.peaks = peaks.copy()
    
    def __del__(self):
        base_segment.__del__(self)
        del(self.gfp)
        del(self.peaks)
    
    def copy(self):
        tmp = base_segment.copy()
        tmp.gfp = self.gfp.copy()
        tmp.peaks = self.peaks.copy()
        
        return tmp
    
    def affinity(self, centroids, normalize = True, with_phase = True, peak_only = False):
        if peak_only:
            return topomap_affinity(self.data[self.peaks], centroids, normalize, with_phase)
        return topomap_affinity(self.data, centroids, normalize, with_phase)
    
    def params(self, centroids, param_funcs, peak_only = False, decomp_mode = False):
        aff = self.affinity(centroids, True, True, peak_only)
        if decomp_mode:
            invcov = get_invcov(centroids)
            aff = get_decomp(invcov, aff)
        if peak_only:
            weight = np.zeros([len(aff)])
            half_interval = np.diff(self.peaks / self.info["sfreq"])
            weight[ : -1] = half_interval
            weight[1 : ] += half_interval
            weight[0] *= 2
            weight[-1] *= 2
        else:
            weight = np.ones([len(aff)]) / self.info["sfreq"]
        res = []
        for func in param_funcs:
            res.append(func(aff, weight))
        
        return res

class twms_segment(base_segment):
    def __init__(self, data, info, pgfp, peaks):
        base_segment.__init__(self, data, info)
        self.pgfp = pgfp.copy()
        self.peaks = peaks.copy()
    
    def __del__(self):
        base_segment.__del__(self)
        del(self.pgfp)
        del(self.peaks)
    
    def copy(self):
        tmp = base_segment.copy()
        tmp.pgfp = self.pgfp.copy()
        tmp.peaks = self.peaks.copy()
        
        return tmp
    
    def params(self, centroids, param_funcs, peak_only = False, decomp_mode = False):
        aff = self.affinity(centroids, True, True)
        if decomp_mode:
            invcov = get_invcov(centroids)
            aff = get_decomp(invcov, aff)
        weight = np.diff(self.peaks / self.info["sfreq"])
        res = []
        for func in param_funcs:
            res.append(func(aff, weight))
        
        return res

class raw_segment(base_segment):
    def __init__(self, data, info, gfp = [], peaks = []):
        base_segment.__init__(self, data, info)
        self.gfp = gfp.copy()
        self.peaks = peaks.copy()
    
    def to_cms(self):
        t_data = np.real(self.data)
        if len(self.gfp) == 0:
            self.gfp, self.peaks = get_gfp_peaks(t_data, self.info["sfreq"])
        
        return cms_segment(t_data, self.info, self.gfp, self.peaks)
    
    def to_twms(self, with_phase = True):
        if len(self.gfp) == 0:
            self.gfp, self.peaks = get_gfp_peaks(np.real(self.data), self.info["sfreq"])
        
        pms = np.zeros([len(self.peaks) - 1, self.data.shape[1]], dtype = complex)
        for i in range(len(self.peaks) - 1):
            pms[i, : ] = get_phasemap(self.data[self.peaks[i] : self.peaks[i + 1]], with_phase)
        
        return twms_segment(pms, self.info, self.gfp[self.peaks], self.peaks)

# quick hilbert transformation
hilbert = lambda x : sig.hilbert(x, fft.next_fast_len(len(x)), axis = 0)[ : len(x)]

def batch_hilbert(maps):
    hil = np.zeros_like(maps, dtype = complex)
    for i in range(maps.shape[1]):
        hil[ : , i] = hilbert(maps[ : , i])
    
    return hil

# effect sizes in the form of Cohen's d
def effect_size_ind(x, y):
    mx = x.mean(axis = 0)
    my = y.mean(axis = 0)
    sqx = ((x - mx) ** 2).sum(axis = 0)
    sqy = ((y - my) ** 2).sum(axis = 0)
    s = np.sqrt((sqx + sqy) / (x.shape[0] + y.shape[0] - 2))
    
    return (mx - my) / s

def effect_size_rel(x, y):
    d = x - y
    
    return d.mean(axis = 0) / d.std(axis = 0, ddof = 1)

# test for linear correlations
def pearson_t_test(x, y):
    n = len(x)
    r = np.corrcoef(x, y)[0, 1]
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    if t < 0:
        p = stats.t(n - 2).cdf(t) * 2
    else:
        p = (1 - stats.t(n - 2).cdf(t)) * 2
    
    return r, t, p

# half-length of confidence interval estimation from t distribution
def t_confidence_interval(x, alpha = 0.05):
    n = len(x)
    sx = np.std(x, ddof = 1) / np.sqrt(n)
    return stats.t(n - 1).ppf(1 - alpha / 2) * sx

# k-l divergence for two distributions
def kldiv_bin(x, y, bins = 10, v_range = [], x_weights = None, y_weights = None):
    if len(v_range) == 0:
        v_range = [min(x.min(), y.min()), max(x.max(), y.max())]
    
    xh = np.histogram(x, bins = bins, range = v_range, weights = x_weights)[0]
    yh = np.histogram(y, bins = bins, range = v_range, weights = y_weights)[0]
    xh = xh / xh.sum()
    yh = yh / yh.sum()
    
    vx, vy = 0, 0
    for i in range(bins):
        if xh[i] != 0:
            if yh[i] == 0:
                vx = -np.log(2)
                break
            vx -= xh[i] * np.log(yh[i] / xh[i])
    for i in range(bins):
        if yh[i] != 0:
            if xh[i] == 0:
                vy = -np.log(2)
                break
            vy -= yh[i] * np.log(xh[i] / yh[i])
    
    return vx / np.log(2), vy / np.log(2)
    
# circular angle differences
def angle_abs_difference(x, y):
    t = np.abs(x - y)
    
    return np.array([min(i, np.pi * 2 - i) for i in t])

def angle_difference(x, y):
    t = x - y
    t[t <= -np.pi] += np.pi * 2
    t[t > np.pi] -= np.pi * 2
    
    return t

# conversion between channel names and indices
def chname2ind(info, st):
    for i in range(len(info["ch_names"])):
        if st == info["ch_names"][i]:
            return i
    
    return -1

def chind2name(info, ind):
    if len(info["ch_names"]) <= ind:
        return -1
    
    return info["ch_names"][ind]