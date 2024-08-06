"""
templates.py
Clustering and validation algorithm

Version 240104.01
Yichao Li
"""

import numpy as np
import sklearn.cluster as clu
import scipy.linalg as lin

import core

# topomap or phase map clustering
def spectral_clustering(aff, k):
    return clu.SpectralClustering(n_clusters = k, affinity = "precomputed").fit_predict(aff)

# get centroid from a group of maps
def get_centroid(x):
    return lin.svd(x, full_matrices = False)[2][0]

# randomly choose a subset of maps
def sampling_maps(pool, n, random_state):
    perm = random_state.permutation(len(pool))
    
    return pool[perm[ : n]].copy()

# optimized batch clustering repeating with random sampling and different number of clusters
def batch_clustering(load_iter, n_file, n_rep, n_map, k_clu, peak_only = False, random_state = None, verbose = True):
    if random_state is None:
        random_state = np.random.RandomState()
    elif type(random_state) == int:
        random_state = np.random.RandomState(seed = random_state)
    
    # get basic info for clustering
    cl = load_iter(0)
    n_ch = cl[0].data.shape[1]
    dt = cl[0].data.dtype
    maps = np.zeros([n_rep, n_map, n_ch], dtype = dt)
    
    # choose and load candidate maps
    k_map = 0
    for i in range(n_file):
        if i != 0:
            cl = load_iter(i)
        
        n_pool = 0
        for j in range(len(cl)):
            if peak_only:
                n_pool += len(cl[j].peaks)
            else:
                n_pool += len(cl[j].data)
        
        pool = np.zeros([n_pool, n_ch], dtype = dt)
        k_pool = 0
        for j in range(len(cl)):
            if peak_only:
                k_pool_cur = k_pool + len(cl[j].peaks)
                pool[k_pool : k_pool_cur] = cl[j].data[cl[j].peaks]
                k_pool = k_pool_cur
            else:
                k_pool_cur = k_pool + len(cl[j].data)
                pool[k_pool : k_pool_cur] = cl[j].data
                k_pool = k_pool_cur
        
        k_map_cur = round(n_map / n_file * (i + 1))
        for j in range(n_rep):
            maps[j, k_map : k_map_cur] = sampling_maps(pool, k_map_cur - k_map, random_state)
        k_map = k_map_cur
    
    # clustering and saving results
    res = [np.zeros([n_rep, ki, n_ch], dtype = dt) for ki in k_clu]
    for i in range(n_rep):
        aff = core.topomap_affinity(maps[i], [], True, False)
        for j in range(len(k_clu)):
            if verbose:
                print(i, "/", n_rep, " - ", k_clu[j], "of", k_clu)
            labels = spectral_clustering(aff, k_clu[j])
            for k in range(k_clu[j]):
                res[j][i, k] = get_centroid(maps[i, labels == k])
    
    return res

# evaluate cluster centroids
def batch_evaluation(load_iter, n_file, centroids, peak_only = False):
    n_rep = centroids[0].shape[0]
    k_clu = [ci.shape[1] for ci in centroids]
    dist = [np.zeros([n_rep, ki]) for ki in k_clu]
    dn = [np.zeros([n_rep, ki]) for ki in k_clu]
    
    for i in range(n_file):
        cl = load_iter(i)
        for j in range(len(cl)):
            for k1 in range(len(k_clu)):
                for k2 in range(n_rep):
                    aff = cl[j].affinity(centroids[k1][k2], True, False, peak_only)
                    label = np.argmax(aff, axis = 1)
                    for k3 in range(k_clu[k1]):
                        ind = np.argwhere(label == k3).reshape([-1])
                        dist[k1][k2, k3] += np.sqrt(2 - 2 * aff[ind, k3]).sum()
                        dn[k1][k2, k3] += len(ind)
    
    return dist, dn

# choosing optimal centroids and analyzing landscape
def analyze_landscape(ctr, edist, edn, convex = False):
    ad = edist.sum(axis = 1) / edn.sum(axis = 1)
    opt = np.argmin(ad)
    
    n = len(ctr)
    d = np.zeros([n])
    for i in range(n):
        if i != opt:
            t = core.topomap_affinity(ctr[opt], ctr[i], False, False)
            d[i] = core.affinity_matching(t, True, False)[2]
    
    if convex:
        v = np.array([(d[i], ad[i], i) for i in range(n)], dtype = [("d", float), ("v", float), ("i", int)])
        v = np.sort(v, order = "d")
        
        cnv = [0]
        while cnv[-1] < n - 1:
            last = cnv[-1]
            slope = (v["v"][last + 1 : ] - v["v"][last]) / (v["d"][last + 1 : ] - v["d"][last])
            cnv.append(np.argmin(slope) + last + 1)
        
        return d, ad, ctr[opt].copy(), v["i"][cnv]
    return d, ad, ctr[opt].copy()