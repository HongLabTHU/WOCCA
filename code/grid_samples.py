"""
grid_samples.py
Generation and analysis examples of regular grid toy data

Version 240122.01
Yichao Li
"""

import numpy as np
import scipy.linalg as lin
from scipy import stats
import matplotlib.pyplot as plt
import time

import core
import wocca
import gradient
import visualize as vis

# %% Grid generation function

def set_grids(n_grid):
    # Define grid positions
    gshape = [n_grid, n_grid]
    n_vec = n_grid ** 2
    grids = np.zeros([n_vec, 2])
    for i in range(n_grid):
        for j in range(n_grid):
            grids[i * n_grid + j] = i / n_grid, j / n_grid
    grids += 0.5 / n_grid - 0.5

    # Define grid adjacency (8-way connections), for gradient computation
    grid_adj = np.zeros([n_vec, n_vec], dtype = bool)
    dirs = [[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
    for i in range(n_grid):
        for j in range(n_grid):
            for kd in dirs:
                if (0 <= i + kd[0] < n_grid) and (0 <= j + kd[1] < n_grid):
                    grid_adj[i * n_grid + j, (i + kd[0]) * n_grid + j + kd[1]] = True

    # Define gradient function
    g_fun = gradient.gradient_transform(None, grids, grid_adj)
    
    return n_vec, grids, gshape, grid_adj, g_fun

# %% Visualization functions

def draw_grid_frame(xl = -0.55, xr = 0.55, yl = -0.55, yr = 0.55, lw = 2):
    plt.plot([xr, xr, xl, xl, xr], [yl, yr, yr, yl, yl], lw = lw, c = "#000000", \
             solid_capstyle = "round", solid_joinstyle = "round")

def axes_rel_pos(axp, fs):
    return [axp[0] / fs[0], axp[1] / fs[1], axp[2] / fs[0], axp[3] / fs[1]]

def disp_grid_topomap(grids, gshape, v, colorbar = False, vmin = None, vmax = None, cmap = "RdBu_r"):
    if (vmin == None) and (vmax == None):
        vrng = np.max(np.abs(v))
        vmin = -vrng
        vmax = vrng
    if vmin == None:
        vmin = np.min(v)
    if vmax == None:
        vmax = np.max(v)
    
    plt.pcolormesh(grids[ : , 0].reshape(gshape), grids[ : , 1].reshape(gshape), v.reshape(gshape), \
                   cmap = cmap, vmin = vmin, vmax = vmax, shading = "nearest")
    plt.axis("equal")
    ax = plt.gca()
    ax.set_frame_on(False)
    ax.set_axis_off()
    if colorbar:
        plt.colorbar()

def disp_grid_phasemap(grids, gshape, v, colorbar = False):
    plt.pcolormesh(grids[ : , 0].reshape(gshape), grids[ : , 1].reshape(gshape), np.angle(v).reshape(gshape) / np.pi * 180, \
                   cmap = "hsv", vmin = -180, vmax = 180, shading = "nearest")
    plt.axis("equal")
    ax = plt.gca()
    ax.set_frame_on(False)
    ax.set_axis_off()
    if colorbar:
        plt.colorbar()

small_grid_phasemap = lambda grids, gshape: (lambda v: disp_grid_phasemap(grids, gshape, v))

def disp_grid_quiver_map(grids, grid_adj, g0, **kwargs):
    g = g0.reshape([-1, 2])
    draw_grid_frame()
    plt.quiver(grids[ : , 0], grids[ : , 1], g[ : , 0], g[ : , 1], color ="#000000", pivot = "mid", **kwargs)
    plt.axis("equal")
    ax = plt.gca()
    ax.set_frame_on(False)
    ax.set_axis_off()

def disp_grid_gradient(grids, grid_adj, v, colorbar = False, show_r2 = False, vmin = 0.5, vmax = 1, **kwargs):
    g, r2 = gradient.get_gradient([], v, True, grids, grid_adj)
    
    colors = np.zeros([len(g), 3])
    for i in range(len(g)):
        colors[i] = max(min(1 - (r2[i] - vmin) / (vmax - vmin), 1), 0)
    draw_grid_frame()
    plt.quiver(grids[ : , 0], grids[ : , 1], g[ : , 0], g[ : , 1], color = colors, pivot = "mid", **kwargs)
    plt.axis("equal")
    ax = plt.gca()
    ax.set_frame_on(False)
    ax.set_axis_off()
    if colorbar:
        plt.colorbar()

def disp_grid_pattern(grids, gshape, grid_adj, vi, fy = 3, title = "Component", gradient_only = False):
    phasemap_flag = int(not gradient_only)
    
    s = 0.2
    u = 0.15
    v = 0.15
    x = fy - 2 * v - 2 * s
    fx = 2 * v + x + phasemap_flag * (x + u)
    fig = plt.figure(figsize = [fx, fy])
    
    if not gradient_only:
        plt.axes([v / fx, v / fy, x / fx, x / fy])
        disp_grid_phasemap(grids, gshape, vi)
        draw_grid_frame()
        plt.title("Phasemap", size = "large")
    
    plt.axes([(v + (x + u) * phasemap_flag) / fx, v / fy, x / fx, x / fy])
    if gradient_only:
        disp_grid_quiver_map(grids, grid_adj, vi)
    else:
        disp_grid_gradient(grids, grid_adj, vi, show_r2 = True)
    plt.title("Gradient", size = "large")
    
    fig.text(0.5, (v + x + s * 3 / 2) / fy, title, size = "large", ha = "center", va = "bottom")

def save_grid_gif(grids, gshape, v, filename, dpi = 50, fps = 24, n_frames = 48):
    tmp = np.array([(v * np.exp(1j * np.pi * 2 * i / n_frames)).real for i in range(n_frames)])
    vmax = np.abs(tmp).max()
    vmin = -vmax
    
    def _fun(i):
        plt.axes([0, 0, 1, 1])
        disp_grid_topomap(grids, gshape, tmp[i], vmin = vmin, vmax = vmax)
        draw_grid_frame()
    
    fig = plt.figure(figsize = [2, 2])
    vis.save_gif(filename, n_frames, _fun, fps, dpi, fig)

# %% Phasemap sample generation functions

def translational(grids, dirs, wl = 2):
    n_dirs = len(dirs)
    n_vec = len(grids)
    v = np.zeros([n_dirs, n_vec], dtype = complex)
    for i in range(n_dirs):
        proj_vec = np.array([np.cos(dirs[i]), np.sin(dirs[i])])
        v[i] = np.exp(-1j * grids.dot(proj_vec) * np.pi * 2 / wl)
    
    return v

def rotational(grids, centers, polarity = 1, wn = 1):
    n_centers = len(centers)
    n_vec = len(grids)
    v = np.zeros([n_centers, n_vec], dtype = complex)
    for i in range(n_centers):
        dir_vec = grids - centers[i]
        v[i] = np.exp(-1j * np.angle(dir_vec[ : , 0] + 1j * dir_vec[ : , 1]) * wn * polarity)
    
    return v

def singular(grids, centers, polarity = 1, wl = 1):
    n_centers = len(centers)
    n_vec = len(grids)
    v = np.zeros([n_centers, n_vec], dtype = complex)
    for i in range(n_centers):
        dir_vec = grids - centers[i]
        v[i] = np.exp(-1j * np.sqrt((dir_vec ** 2).sum(axis = 1)) * np.pi * 2 / wl * polarity)
    
    return v

def saddle(grids, centers, orientations, wl = 1):
    n_centers = len(centers)
    n_vec = len(grids)
    v = np.zeros([n_centers, n_vec], dtype = complex)
    for i in range(n_centers):
        dir_vec = grids - centers[i]
        ang_vec = np.angle(dir_vec[ : , 0] + 1j * dir_vec[ : , 1])
        v[i] = np.exp(-1j * np.sqrt((dir_vec ** 2).sum(axis = 1)) * np.cos((ang_vec - orientations[i]) * 2) * np.pi * 2 / wl)
    
    return v

def randomize_batch_translational(n_samples, grids, dir_mean, dir_std, wl, random_state):
    dirs = stats.norm().rvs(size = [n_samples], random_state = random_state) * dir_std + dir_mean
    return translational(grids, dirs, wl)
    
def randomize_batch_rotational(n_samples, grids, c_mean, c_std, polarity, wn, random_state):
    centers = stats.norm().rvs(size = [n_samples, 2], random_state = random_state) * c_std + c_mean
    return rotational(grids, centers, polarity, wn)

def randomize_batch_singular(n_samples, grids, c_mean, c_std, polarity, wl, random_state):
    centers = stats.norm().rvs(size = [n_samples, 2], random_state = random_state) * c_std + c_mean
    return singular(grids, centers, polarity, wl)

def randomize_batch_saddle(n_samples, grids, c_mean, c_std, ori_mean, ori_std, wl, random_state):
    centers = stats.norm().rvs(size = [n_samples, 2], random_state = random_state) * c_std + c_mean
    oris = stats.norm().rvs(size = [n_samples], random_State = random_state) * ori_std + ori_mean
    return saddle(grids, centers, oris, wl)

# %% Run examples and benchmarks

if __name__ == "__main__":
    n_grid = 6
    n_vec, grids, gshape, grid_adj, g_fun = set_grids(n_grid)
    
    n_grid_l = 8
    n_vec_l, grids_l, gshape_l, grid_adj_l, g_fun_l = set_grids(n_grid_l)
    
    # %% Generate and visualize some examples
    
    vt, vh = translational(grids, np.array([np.pi / 2, 0]))
    vr = rotational(grids, np.array([[0, 0]]), 1).squeeze()
    vs = singular(grids, np.array([[0, 0]]), 1).squeeze()
    vst, vss = saddle(grids, np.array([[0, 0], [0, 0]]), np.array([np.pi / 2,  np.pi / 4]), 1)
    vt_l, vh_l = translational(grids, np.array([np.pi / 2, 0]), wl = 2 / 3).squeeze()
    vr_l = rotational(grids, np.array([[0, 0]]), 1, 2).squeeze()
    
    samples = np.array([vt, np.conj(vt), vh, np.conj(vh), vr, np.conj(vr), vs, np.conj(vs), vst, vt_l, vr_l])
    sample_names = ["FW", "BW", "LW", "RW", "CCw", "Cw", "Source", "Drain", "Saddle", "FW-s", "CCw-s"]
    sample_names_long = ["Forward", "Backward", "Leftward", "Rightward", "Counter-clockwise", \
                         "Clockwise", "Source singularity", "Drain singularity", "Saddle singularity", "Forward with 1/3 wavelength", \
                         "Counter-clockwise with double winding number"]
    for i in range(len(samples)):
        samples[i] /= np.sqrt(wocca.norm(samples[i]))
    
    for i in range(len(samples)):
        disp_grid_pattern(grids, gshape, grid_adj, samples[i], title = sample_names_long[i])
    
    vis.disp_group_affinity(core.topomap_affinity(samples, [], False, False), f_topomap = small_grid_phasemap(grids, gshape), vx = samples, \
                            names_x = sample_names, symmetric = True, cmap = "RdBu_r", colorbar = True, vmin = -1, vmax = 1, text = True, \
                            groups = [[0, 2], [2, 4], [4, 6], [6, 8]])
    
    # %% Generate the first benchmark dataset
    
    random_seed = 0
    n_batch = 1000
    
    # Randomly generate dataset
    random_state = np.random.RandomState(random_seed)
    bv1 = []
    bv1.append(randomize_batch_translational(n_batch, grids, random_state = random_state, \
                                             dir_mean = np.pi / 2, dir_std = np.pi / 6, wl = 2))
    bv1.append(randomize_batch_translational(n_batch, grids, random_state = random_state, \
                                             dir_mean = -np.pi / 2, dir_std = np.pi / 6, wl = 2))
    bv1.append(randomize_batch_rotational(n_batch, grids, random_state = random_state, \
                                          c_mean = [0, 0], c_std = 0.1, polarity = 1, wn = 1))
    bv1.append(randomize_batch_rotational(n_batch, grids, random_state = random_state, \
                                          c_mean = [0, 0], c_std = 0.1, polarity = -1, wn = 1))
    bv1 = np.concatenate(bv1)
    bv1 = np.array([vi / np.sqrt(wocca.norm(vi)) for vi in bv1])

    # Generate standard benchmark templates
    bt1 = np.zeros([4, n_vec], dtype = complex)
    bt1[ : 2] = translational(grids, np.array([np.pi / 2, -np.pi / 2]), wl = 2)
    bt1[2] = rotational(grids, np.array([[0, 0]]), polarity = 1, wn = 1)
    bt1[3] = rotational(grids, np.array([[0, 0]]), polarity = -1, wn = 1)
    bt1 = np.array([vi / np.sqrt(wocca.norm(vi)) for vi in bt1])
    
    # Visualize the composition of the first benchmark dataset
    for i in range(4):
        disp_grid_pattern(grids, gshape, grid_adj, bt1[i], title = "Bench 1 Template " + str(i + 1))

    # %% Decompose and analyze the first benchmark dataset
    # NOTICE: MAY TAKE SEVERAL MINUTES
    
    n_comps = 10
    n_visualize = 5
    
    # CPCA
    cpca_comps_b1 = wocca.cpca(bv1)[ : n_comps]
    cpca_scores_b1 = (np.abs(bv1 @ np.conj(cpca_comps_b1.T)) ** 2).sum(axis = 0)
    cpca_total_b1 = (np.abs(bv1) ** 2).sum()
    
    # WOCCA
    wocca_comps_b1 = wocca.wocca(bv1, n_components = n_comps, use_torch = True, backtrack = True)
    wocca_scores_b1 = wocca.score(bv1, wocca_comps_b1)
    wocca_total_b1 = wocca.total_energy(bv1)
    
    # Vector field SVD
    bg1 = np.array([gradient.get_gradient([], v, False, grids, grid_adj).reshape([-1]) for v in bv1])
    u, sv, vh = lin.svd(bg1, full_matrices = False)
    vsvd_comps_b1 = vh[ : n_comps].reshape([n_comps, -1, 2])
    vsvd_scores_b1 = sv[ : n_comps] ** 2
    vsvd_total_b1 = (bg1 ** 2).sum()
    
    # Visualize all components
    for i in range(n_visualize):
        disp_grid_pattern(grids, gshape, grid_adj, cpca_comps_b1[i], \
                          title = "Bench 1 CPCA " + str(i + 1) + r" $EV={:.2f}\%$".format(cpca_scores_b1[i] / cpca_total_b1 * 100))
    for i in range(n_visualize):
        disp_grid_pattern(grids, gshape, grid_adj, wocca_comps_b1[i], \
                          title = "Bench 1 WOCCA " + str(i + 1) + r" $EV={:.2f}\%$".format(wocca_scores_b1[i] / wocca_total_b1 * 100))
    for i in range(n_visualize):
        disp_grid_pattern(grids, gshape, grid_adj, vsvd_comps_b1[i], gradient_only = True, \
                          title = "Bench 1 VSVD " + str(i + 1) + r" $EV={:.2f}\%$".format(vsvd_scores_b1[i] / vsvd_total_b1 * 100))
    
    # Compare components with standard templates
    aff_t_cpca_b1 = core.topomap_affinity(bt1, cpca_comps_b1[ : 4], False, False)
    vis.disp_group_affinity(aff_t_cpca_b1, f_topomap = small_grid_phasemap(grids, gshape), vx = bt1, vy = cpca_comps_b1[ : 4], \
                            names_x = ["FW", "BW", "Rot+", "Rot-"], names_y = ["CPC " + str(i + 1) for i in range(4)], \
                            fy = 4.8, x_text = 0.8, colorbar = True, cmap = "Blues", vmin = 0, vmax = 1, text = True)
    
    tmp_wocca_b1_comps = np.concatenate([[vi, np.conj(vi)] for vi in wocca_comps_b1[ : 2]])
    tmp_wocca_b1_names = ["WOC 1", "WOC 1C", "WOC 2", "WOC 2C"]
    aff_t_wocca_b1 = core.topomap_affinity(bt1, tmp_wocca_b1_comps, False, False)
    vis.disp_group_affinity(aff_t_wocca_b1, f_topomap = small_grid_phasemap(grids, gshape), vx = bt1, vy = tmp_wocca_b1_comps, \
                            names_x = ["FW", "BW", "Rot+", "Rot-"], names_y = tmp_wocca_b1_names, \
                            fy = 4.8, x_text = 0.8, colorbar = True, cmap = "Blues", vmin = 0, vmax = 1, text = True)
    
    # %% Generate the second benchmark dataset
    
    random_seed = 0
    n_batch = 1000
    
    # Randomly generate dataset
    random_state = np.random.RandomState(random_seed)
    bv2 = []
    bv2.append(randomize_batch_translational(n_batch, grids, random_state = random_state, \
                                             dir_mean = np.pi / 2, dir_std = np.pi / 12, wl = 2))
    bv2.append(randomize_batch_translational(n_batch, grids, random_state = random_state, \
                                             dir_mean = np.pi / 2, dir_std = np.pi / 12, wl = 2 / 3))
    bv2.append(randomize_batch_rotational(n_batch, grids, random_state = random_state, \
                                          c_mean = [0, 0], c_std = 0.1, polarity = 1, wn = 1))
    bv2.append(randomize_batch_rotational(n_batch, grids, random_state = random_state, \
                                          c_mean = [0, 0], c_std = 0.1, polarity = 1, wn = 2))
    bv2 = np.concatenate(bv2)
    bv2 = np.array([vi / np.sqrt(wocca.norm(vi)) for vi in bv2])

    # Generate standard benchmark templates
    bt2 = np.zeros([4, n_vec], dtype = complex)
    bt2[0] = translational(grids, np.array([np.pi / 2]), wl = 2)
    bt2[1] = translational(grids, np.array([np.pi / 2]), wl = 2 / 3)
    bt2[2] = rotational(grids, np.array([[0, 0]]), polarity = 1, wn = 1)
    bt2[3] = rotational(grids, np.array([[0, 0]]), polarity = 1, wn = 2)
    bt2 = np.array([vi / np.sqrt(wocca.norm(vi)) for vi in bt2])
    
    # Visualize the composition of the second benchmark dataset
    for i in range(4):
        disp_grid_pattern(grids, gshape, grid_adj, bt2[i], title = "Bench 2 Template " + str(i + 1))
    
    # %% Decompose and analyze the second benchmark dataset
    # NOTICE: MAY TAKE SEVERAL MINUTES
    
    n_comps = 10
    n_visualize = 5
    
    # CPCA
    cpca_comps_b2 = wocca.cpca(bv2)[ : n_comps]
    cpca_scores_b2 = (np.abs(bv2 @ np.conj(cpca_comps_b2.T)) ** 2).sum(axis = 0)
    cpca_total_b2 = (np.abs(bv2) ** 2).sum()
    
    # WOCCA
    wocca_comps_b2 = wocca.wocca(bv2, n_components = n_comps, use_torch = True, backtrack = True)
    wocca_scores_b2 = wocca.score(bv2, wocca_comps_b2)
    wocca_total_b2 = wocca.total_energy(bv2)
    
    # Vector field SVD
    bg2 = np.array([gradient.get_gradient([], v, False, grids, grid_adj).reshape([-1]) for v in bv2])
    u, sv, vh = lin.svd(bg2, full_matrices = False)
    vsvd_comps_b2 = vh[ : n_comps].reshape([n_comps, -1, 2])
    vsvd_scores_b2 = sv[ : n_comps] ** 2
    vsvd_total_b2 = (bg2 ** 2).sum()
    
    # Visualize all components
    for i in range(n_visualize):
        disp_grid_pattern(grids, gshape, grid_adj, cpca_comps_b2[i], \
                          title = "Bench 2 CPCA " + str(i + 1) + r" $EV={:.2f}\%$".format(cpca_scores_b2[i] / cpca_total_b2 * 100))
    for i in range(n_visualize):
        disp_grid_pattern(grids, gshape, grid_adj, wocca_comps_b2[i], \
                          title = "Bench 2 WOCCA " + str(i + 1) + r" $EV={:.2f}\%$".format(wocca_scores_b2[i] / wocca_total_b2 * 100))
    for i in range(n_visualize):
        disp_grid_pattern(grids, gshape, grid_adj, vsvd_comps_b2[i], gradient_only = True, \
                          title = "Bench 2 VSVD " + str(i + 1) + r" $EV={:.2f}\%$".format(vsvd_scores_b2[i] / vsvd_total_b2 * 100))
   
    # %% Testing efficiency of PCA pre-reduction using a larger dataset (8*8 grid, 2000*4 maps)
    # NOTICE: MAY TAKE HOURS WITHOUT GPU, OR SEVERAL MINUTES WITH GPU
    
    random_seed = 0
    n_batch = 2000
    
    # Randomly generate dataset
    random_state = np.random.RandomState(random_seed)
    bv3 = []
    bv3.append(randomize_batch_translational(n_batch, grids_l, random_state = random_state, \
                                             dir_mean = np.pi / 2, dir_std = np.pi / 6, wl = 2))
    bv3.append(randomize_batch_translational(n_batch, grids_l, random_state = random_state, \
                                             dir_mean = -np.pi / 2, dir_std = np.pi / 6, wl = 2))
    bv3.append(randomize_batch_rotational(n_batch, grids_l, random_state = random_state, \
                                          c_mean = [0, 0], c_std = 0.1, polarity = 1, wn = 1))
    bv3.append(randomize_batch_rotational(n_batch, grids_l, random_state = random_state, \
                                          c_mean = [0, 0], c_std = 0.1, polarity = -1, wn = 1))
    bv3 = np.concatenate(bv3)
    bv3 = np.array([vi / np.sqrt(wocca.norm(vi)) for vi in bv3])
    
    n_dim = 20
    n_comps = 5
    
    # Calculate b matrix directly
    t0 = time.time()
    wc_b3 = wocca.wocca(bv3, n_components = n_comps, use_torch = True, backtrack = True)
    t_direct = time.time() - t0
    
    # Calculate b matrix with pre-reduction
    t0 = time.time()
    bv3_reduce, basis = wocca.pca_decomp(bv3, n_dim)
    wc_b3_reduce = wocca.wocca(bv3_reduce, n_components = n_comps, use_torch = True, backtrack = True)
    wc_b3_recover = wocca.pca_recover(wc_b3_reduce, basis)
    t_reduce = time.time()- t0
    
    print("Without pre-reduction: {:.3f}".format(t_direct))
    print("With pre-reduction: {:.3f}".format(t_reduce))
    
    # Compare accuracy of first 5 components
    for i in range(n_comps):
        print("Comp.", i + 1, ":", wocca.w(wc_b3[i], wc_b3_recover[i]))