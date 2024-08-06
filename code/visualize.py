"""
visualize.py
Basic visualization functionality of EEG phasemap data

Version 240125.01
Yichao Li
"""

import numpy as np
import mne
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import core
import gradient as grad

# %% Colors

colors = plt.rcParams['axes.prop_cycle'].by_key()["color"]

cdict = {'red':   [[0.0, 1.0, 1.0],
                   [1.0, 0.0, 0.0]],
         'green': [[0.0, 1.0, 1.0],
                   [1.0, 0.0, 0.0]],
         'blue':  [[0.0, 1.0, 1.0],
                   [1.0, 0.0, 0.0]]}
linear_greys_cmap = mpl.colors.LinearSegmentedColormap('L_Greys', segmentdata = cdict, N = 256)

cdict = {'red':   [[0.0, 1.0, 1.0],
                   [1.0, 1.0, 1.0]],
         'green': [[0.0, 1.0, 1.0],
                   [1.0, 1.0, 1.0]],
         'blue':  [[0.0, 1.0, 1.0],
                   [1.0, 1.0, 1.0]]}
whites_cmap = mpl.colors.LinearSegmentedColormap('Whites', segmentdata = cdict, N = 256)

default_head_size = 0.5

# Convert RGB format string into tuple
def rgb_str2arr(c):
    alpha = len(c) == 9
    res = []
    for i in range(3 + int(alpha)):
        res.append(int(c[i * 2 + 1 : i * 2 + 3], 16) / 255)
    
    return tuple(res)

# Convert RGB format tuple into string
def rgb_arr2str(c):
    alpha = len(c) == 4
    res = "#"
    for i in range(3 + int(alpha)):
        res += "{:02x}".format(int(c[i] * 255))
    
    return res

# Compute luminance from RGB format tuple
def luminance(c):
    if type(c) == str:
        c = rgb_str2arr(c)
    return np.sqrt(0.299 * (c[0] ** 2) + 0.587 * (c[1] ** 2) + 0.114 * (c[2] ** 2))

# Append alpha channel to RGB format string color
def append_alpha(c, alpha = 1):
    return c + "{:02x}".format(int(round(alpha * 255)))

# Compute saturation
def saturation(c):
    if type(c) == str:
        c = rgb_str2arr(c)
    cm = np.mean(c)
    g = [(ci - cm) / cm for ci in c]
    
    return np.max(g)

# Modify saturation
def sat_mod(c, delta):
    str_flag = type(c) == str
    if str_flag:
        c = rgb_str2arr(c)
    
    cm = np.mean(c)
    c1 = [min(max((ci - cm) * (1 + delta) + cm, 0), 1) for ci in c]
    
    if str_flag:
        return rgb_arr2str(c1)
    return tuple(c1)

# Modify saturation towards white
def sat_mod_white(c, delta):
    str_flag = type(c) == str
    if str_flag:
        c = rgb_str2arr(c)
    
    c1 = [max(1 - (1 - ci) * (1 + delta), 0) for ci in c]
    
    if str_flag:
        return rgb_arr2str(c1)
    return tuple(c1)

# %% Channel layout transfer

def calibrate_montage(m_ref, m0):
    mr_pos = m_ref.get_positions()
    m0_pos = m0.get_positions()
    
    scale = (np.abs(mr_pos["lpa"][0]) + np.abs(mr_pos["rpa"][0])) / (np.abs(m0_pos["lpa"][0]) + np.abs(m0_pos["rpa"][0]))
    y_offset = mr_pos["nasion"][1] - m0_pos["nasion"][1] * scale
    ch_pos = m0_pos["ch_pos"].copy()
    for i in ch_pos:
        ch_pos[i] *= scale
        ch_pos[i][1] += y_offset

    return mne.channels.make_dig_montage(ch_pos, mr_pos["nasion"], mr_pos["lpa"], mr_pos["rpa"])

def make_calibrated_info(info_ref, info):
    return info.copy().set_montage(calibrate_montage(info_ref.get_montage(), info.get_montage()))

def raw_montage_transfer(raw, info_target, info_ref = None):
    # Rename all channel names of the original channels
    r = raw.copy()
    rename_dict = dict()
    for st in r.info.ch_names:
        rename_dict[st] = "org_" + st
    r.rename_channels(rename_dict)

    # Calibrate and concatenate montages
    if info_ref is None:
        info_ref = info_target.copy()
    m_ref = info_ref.get_montage()
    montage = calibrate_montage(m_ref, info_ref.get_montage()) + calibrate_montage(m_ref, r.get_montage())
    
    # Make new combined RawArray instance
    n0 = len(r.info.ch_names)
    n1 = len(info_target.ch_names)
    data = np.zeros([n0 + n1, len(r)], dtype = r.get_data().dtype)
    data[n1 : ] = r.get_data()
    info_t = mne.create_info(montage.ch_names, r.info["sfreq"], ch_types = "eeg").set_montage(montage)
    raw_t = mne.io.RawArray(data, info_t)

    # Interpolate bads and drop the original channels
    raw_t.info["bads"] = info_target.ch_names.copy()
    raw_t.interpolate_bads()
    raw_t.drop_channels(r.info.ch_names)

    return raw_t

def phasemap_montage_transfer(v, info_org, info_target, info_ref = None, spc = 100):
    # Unroll phasemap
    ts = np.linspace(0, np.pi * 6, spc * 3, False)
    data = np.zeros([len(v), spc * 3])
    for i in range(spc * 3):
        data[ : , i] = np.real(v * np.exp(1j * ts[i]))

    # Create and transfer RawArray
    raw = mne.io.RawArray(data, info_org)
    raw_t = raw_montage_transfer(raw, info_target, info_ref)
    raw_t.set_eeg_reference("average")
    raw_t = raw_t.apply_hilbert()

    # Extract phasemap
    return core.get_phasemap(raw_t.get_data()[ : , spc : spc * 2].T, starting_phase = True)

# %% Icons for the most critical traveling waves

# icon_color = "#4b92c3"

# def set_icon_frame(fs = [2, 2], axes_range = [0.075, 0.075, 0.85, 0.85], with_frame = False, frame_lw = 7):
#     plt.figure(figsize = fs)
#     ax = plt.axes(axes_range)
#     ax.set_frame_on(False)
#     ax.set_axis_off()
#     plt.axis((0, 1, 0, 1))
#     if with_frame:
#         plt.plot([0, 1, 1, 0, 0], [1, 1, 0, 0, 1], lw = frame_lw, c = "#000000", clip_on = False, \
#                  solid_capstyle = "round", solid_joinstyle = "round")

# def forward(color = icon_color):
#     plt.arrow(0.5, 0.5, 0, 1e-3, length_includes_head = False, head_width = 0.3, head_length = 0.4, width = 0, \
#               ec = "#00000000", fc = color)
#     plt.plot([0.5, 0.5], [0.1, 0.61], lw = 15, c = color, solid_capstyle = "butt")

# def forward_side(color = icon_color):
#     thetas = np.linspace(np.pi * 160 / 180, np.pi * 220 / 180, 61, True)
#     plt.plot(np.cos(thetas) * 0.6 + 0.85, np.sin(thetas) * 0.6 + 0.5, lw = 12, c = color, solid_capstyle = "butt", solid_joinstyle = "round")
#     plt.arrow(np.cos(thetas[1]) * 0.6 + 0.85, np.sin(thetas[1]) * 0.6 + 0.5, 1e-3 * np.sin(thetas[1]), -1e-3 * np.cos(thetas[1]), \
#               length_includes_head = False, head_width = 0.2, head_length = 0.2, width = 0, ec = "#00000000", fc = color)
#     thetas = np.linspace(np.pi * (-40) / 180, np.pi * 20 / 180, 61, True)
#     plt.plot(np.cos(thetas) * 0.6 + 0.15, np.sin(thetas) * 0.6 + 0.5, lw = 12, c = color, solid_capstyle = "butt", solid_joinstyle = "round")
#     plt.arrow(np.cos(thetas[-2]) * 0.6 + 0.15, np.sin(thetas[-2]) * 0.6 + 0.5, -1e-3 * np.sin(thetas[-2]), 1e-3 * np.cos(thetas[-2]), \
#               length_includes_head = False, head_width = 0.2, head_length = 0.2, width = 0, ec = "#00000000", fc = color)

# def counter_clockwise(color = icon_color):
#     plt.arrow(0.5, 0.2, 1e-3, 0, length_includes_head = False, head_width = 0.24, head_length = 0.32, width = 0, \
#               ec = "#00000000", fc = color)
#     plt.plot([0.5, 0.51], [0.2, 0.2], lw = 15, c = color, solid_capstyle = "butt")
#     plt.plot([0.5], [0.5], ".", ms = 20, c = icon_color)
#     thetas = np.linspace(0, np.pi * 3 / 2, 271, True)
#     plt.plot(np.cos(thetas) * 0.3 + 0.5, np.sin(thetas) * 0.3 + 0.5, lw = 15, c = color, solid_capstyle = "butt", solid_joinstyle = "round")

# def rightward(color = icon_color):
#     plt.arrow(0.5, 0.5, 1e-3, 0, length_includes_head = False, head_width = 0.3, head_length = 0.4, width = 0, \
#               ec = "#00000000", fc = color)
#     plt.plot([0.1, 0.61], [0.5, 0.5], lw = 15, c = color, solid_capstyle = "butt")

# def ccw_front(color = icon_color, color2 = "#808080"):
#     plt.arrow(0.5, 0.35, 1e-3, 0, length_includes_head = False, head_width = 0.18, head_length = 0.24, width = 0, \
#               ec = "#00000000", fc = color)
#     plt.plot([0.5, 0.51], [0.35, 0.35], lw = 12, c = color, solid_capstyle = "butt")
#     plt.plot([0.5], [0.6], ".", ms = 18, c = icon_color)
#     thetas = np.linspace(0, np.pi * 3 / 2, 271, True)
#     plt.plot(np.cos(thetas) * 0.25 + 0.5, np.sin(thetas) * 0.25 + 0.6, lw = 12, c = color, solid_capstyle = "butt", solid_joinstyle = "round")
#     plt.plot([0.2, 0.8], [0.15, 0.15], lw = 12, c = color2, solid_capstyle = "butt")

# def ccw_post(color = icon_color, color2 = "#808080"):
#     plt.arrow(0.5, 0.15, 1e-3, 0, length_includes_head = False, head_width = 0.18, head_length = 0.24, width = 0, \
#               ec = "#00000000", fc = color)
#     plt.plot([0.5, 0.51], [0.15, 0.15], lw = 12, c = color, solid_capstyle = "butt")
#     plt.plot([0.5], [0.4], ".", ms = 18, c = icon_color)
#     thetas = np.linspace(0, np.pi * 3 / 2, 271, True)
#     plt.plot(np.cos(thetas) * 0.25 + 0.5, np.sin(thetas) * 0.25 + 0.4, lw = 12, c = color, solid_capstyle = "butt", solid_joinstyle = "round")
#     plt.plot([0.2, 0.8], [0.85, 0.85], lw = 12, c = color2, solid_capstyle = "butt")

# f_icons = [forward, forward_side, counter_clockwise, rightward, ccw_front, ccw_post]
# icon_names = ["fw", "fs", "ccw", "rw", "ccw_f", "ccw_p"]

# %% Topomaps and phasemaps

# get physical locations of electrodes via layout
def get_phys_pos(info):
    if type(info) == np.ndarray:
        return info.copy()
    if type(info) == mne.channels.layout.Layout:
        return info.pos.copy()
    return mne.channels.make_eeg_layout(info, width = 0, height = 0).pos[ : , : 2].copy() - 0.5

# get physical distances between electrodes
def get_phys_dist(info):
    pos = get_phys_pos(info)
    n = pos.shape[0]
    dist = np.zeros([n, n])
    for i in range(n - 1):
        for j in range(i + 1, n):
            dist[i, j] = np.sqrt(((pos[i, 0 : 2] - pos[j, 0 : 2]) ** 2).sum())
            dist[j, i] = dist[i, j]
    
    return dist

# display an ordinary topomap
def disp_topomap(info, val, vmin = None, vmax = None, cmap = "jet", colorbar = True, sensors = True, ret_im = False, start_figure = False):
    if start_figure:
        plt.figure()
    
    if type(info) != np.ndarray:
        pos = get_phys_pos(info)
    else:
        pos = info
    ax = plt.gca()
    im = mne.viz.plot_topomap(val, pos, vlim = (vmin, vmax), cmap = cmap, sphere = default_head_size, show = False, sensors = sensors, axes = ax)[0]
    if colorbar:
        plt.colorbar(im, ax = ax)
    if ret_im:
        return im

# display an array of electrodes
def disp_elecmap(info, val, vmin = None, vmax = None, cmap = "jet", markersize = 20, colorbar = False, sensors = True, ret_im = False, start_figure = False):
    if start_figure:
        plt.figure()
    
    if vmin is None:
        vmin = np.min(val)
    if vmax is None:
        vmax = np.max(val)
    n = val.shape[0]
    pos = get_phys_pos(info)
    disp_topomap(info, np.zeros([len(pos)]), cmap = whites_cmap, colorbar = False, sensors = sensors)
    for i in range(n):
        plt.plot([pos[i, 0]], [pos[i, 1]], ".", markersize = markersize, color = mpl.colormaps.get_cmap(cmap)((val[i] - vmin) / (vmax - vmin)))
    if colorbar:
        plt.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(vmin, vmax), cmap = cmap))
    if ret_im:
        return mpl.cm.ScalarMappable(mpl.colors.Normalize(vmin, vmax), cmap = cmap)

# display a phase map in the form of an electrode array
def disp_phasemap(info, val, markersize = 20, colorbar = False, ret_im = False):
    phase = np.angle(val) / np.pi * 180
    tmp_im = disp_elecmap(info, phase, vmin = -180, vmax = 180, cmap = "hsv", markersize = markersize, colorbar = colorbar, ret_im = ret_im)
    if ret_im:
        return tmp_im

# display an amplitude map in the form of a topomap
def disp_amp_topomap(info, val, colorbar = True, vmin = 0, vmax = 0.3):
    disp_topomap(info, np.abs(val), vmin = vmin, vmax = vmax, cmap = "jet", colorbar = colorbar)

# display an amplitude map in the form of an electrode array
def disp_amp_elecmap(info, val, markersize = 20, vmin = 0, vmax = 0.3, colorbar = False, sensors = True):
    disp_elecmap(info, np.abs(val), vmin = vmin, vmax = vmax, cmap = "jet", markersize = markersize, colorbar = colorbar, sensors = sensors)

# display an electrode array phase map with amplitudes represented by size of dots
def disp_phasemap_exact(info, val, markersize_norm = 20, vmin = -180, vmax = 180, colorbar = False, cmap = "hsv"):
    phase = np.angle(val) / np.pi * 180
    amp = np.abs(val)
    amp_rel = amp.max()
    n = phase.shape[0]
    pos = get_phys_pos(info)
    disp_topomap(info, np.zeros([len(pos)]), cmap = whites_cmap, colorbar = False, sensors = True)
    for i in range(n):
        plt.plot([pos[i, 0]], [pos[i, 1]], ".", markersize = markersize_norm * amp[i] / amp_rel, color = mpl.colormaps.get_cmap(cmap)((phase[i] - vmin) / (vmax - vmin)))
    if colorbar:
        plt.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(vmin, vmax), cmap = cmap))

# display an array of vectors with greyscale value indication
def disp_vectors(info, vec, val, vmin = 0.5, vmax = 1, colorbar = False, sensors = True, start_figure = False, **kwargs):
    pos = get_phys_pos(info)
    if start_figure:
        plt.figure()
    colors = np.zeros([len(vec), 4])
    for i in range(len(vec)):
        colors[i, 3] = max(min((val[i] - vmin) / (vmax - vmin), 1), 0)
    disp_topomap(info, np.zeros([len(pos)]), cmap = whites_cmap, colorbar = False, sensors = sensors)
    plt.quiver(pos[ : , 0], pos[ : , 1], vec[ : , 0], vec[ : , 1], color = colors, **kwargs)

# display a phase gradient quiver map with greyscale colored r2 values
def disp_quiver_map(info, val, show_r2 = False, vmin = 0.5, vmax = 1, colorbar = False, sensors = True, **kwargs):
    if show_r2:
        g, r2 = grad.get_gradient(info, val, True)
    else:
        g = grad.get_gradient(info, val)
        r2 = np.ones([len(g)])
    
    disp_vectors(info, g, r2, vmin, vmax, colorbar, sensors, **kwargs)

# %% Miscellaneous functions for data analysis and statistics

minus_sign = "−"

# Standalone colorbar
def colorbar(cmap, vmin = 0, vmax = 1, ticks = [0, 1], tick_labels = None, clip = None, caption = "", frame_on = True, \
             fs = [1, 1], bs = [0.15, 0.8], dpi = 300, n_step = 1000, tick_size = "medium", caption_size = "medium", norm = None):
    with_norm = not (norm is None)
    
    fig = plt.figure(figsize = fs, dpi = dpi)
    yl = (fs[1] - bs[1]) / 2
    ax = plt.axes([0.05 / fs[0], yl / fs[1], bs[0] / fs[0], bs[1] / fs[1]])
    
    if clip is None:
        tmp_lin = np.linspace(vmin, vmax, n_step + 1, True)
    else:
        tmp_lin = np.linspace(clip[0], clip[1], n_step + 1, True)
    if with_norm:
        vmin, vmax = None, None
    plt.pcolormesh([0, 1], tmp_lin, ((tmp_lin[ : -1] + tmp_lin[1 : ]) / 2).reshape([-1, 1]), \
                   cmap = cmap, vmin = vmin, vmax = vmax, norm = norm)
    ax.yaxis.tick_right()
    ax.tick_params(labelsize = tick_size)
    ax.set_frame_on(frame_on)
    ax.set_xticks([])
    ax.set_yticks(ticks)
    if not (tick_labels is None):
        ax.set_yticklabels(tick_labels)
    fig.text(1 - 0.05 / fs[0], 0.5, caption, size = caption_size, \
             ha = "right", va = "center", rotation = "vertical")
    
    return fig

# Generic function for saving a series of graphs as a gif
def save_gif(filename, n_frames, _fun, fps = 24, dpi = 50, fig = [], verbose = True, clear_fig = True):
    writer = anim.ImageMagickWriter(fps = fps)
    if type(fig) == list:
        fig = plt.figure()
    
    with writer.saving(fig, filename, dpi):
        for i in range(n_frames):
            if verbose:
                print(i, "/", n_frames)
            
            _fun(i)
            
            if i + 1 < n_frames:
                writer.grab_frame()
            if clear_fig:
                fig.clear()
    
    plt.close("all")

# Generate text labels for p-values
pv_thres = [0.001, 0.01, 0.05, 0.1]
pv_symbols = ["***", "**", "*", "†", "n.s."]
pv_symbols_no_ns = ["***", "**", "*", "†", ""]

def set_pv_string(t_thres, t_symbols):
    global pv_thres, pv_symbols, pv_symbols_no_ns
    pv_thres = t_thres.copy()
    pv_symbols = t_symbols.copy()
    pv_symbols_no_ns = t_symbols.copy()
    pv_symbols_no_ns[-1] = ""

def generate_pv_string(pv, show_ns = False):
    if pv < 0.01:
        power = int(np.floor(np.log(pv) / np.log(10)))
        frac = pv * (10 ** (-power))
        st_p = r"$p={:.1f}\times 10^".format(frac) + "{" + str(power) + "}$"
    elif pv < 0.1:
        st_p = "$p={:.3f}$".format(pv)
    else:
        st_p = "$p={:.2f}$".format(pv)
    st_s = (pv_symbols if show_ns else pv_symbols_no_ns)[np.searchsorted(pv_thres, pv)]
    
    return st_p, st_s

def generate_dv_string(dv):
    return "$d={:.3f}$".format(dv)

# Set no axes
def no_axes(ax):
    ax.set_frame_on(False)
    ax.set_axis_off()

# Set lower-left axes
def lower_left(ax, x_axis_at_zero = False, show_xaxis = True, show_yaxis = True):
    ax.set_frame_on(False)
    ax_pos = ax.axis()
    if show_xaxis:
        plt.axhline(y = 0 if x_axis_at_zero else ax_pos[2], lw = 1, c = "#000000", clip_on = False, zorder = -20)
    if show_yaxis:
        plt.axvline(x = ax_pos[0], lw = 1, c = "#000000", clip_on = False, zorder = -20)

# %% Long and reusable visualization modules

# Draw scree plot with component bars and accumulated lines
def draw_scree_plot(data, comp_labels = None, xlabel = None, text = False, percentage = True, \
                    fig = None, dpi = 200, xps = [0.7, 0.2, 0.28, 0.07, 0.7], yps = [0.3, 0.2, 2, 0.1], yrng = None, yrng2 = None, text_ylocs = None, \
                    classes = None, class_colors = [colors[0]], yaxis_color = colors[0], acc_color = colors[3]):
    n = len(data)
    
    with_xlabel = not (xlabel is None)
    if classes is None:
        classes = np.zeros([n], dtype = int)
    m = classes.max() + 1
    
    centers = xps[1] + np.arange(n) * (xps[2] + xps[3]) + 0.5 * xps[2]
    
    xmax = xps[1] * 2 + xps[2] * n + xps[3] * (n - 1)
    fx = xps[0] + xmax + xps[4]
    fy = yps[0] + yps[1] * int(with_xlabel) + yps[2] + yps[3]
    
    if fig is None:
        fig = plt.figure(figsize = [fx, fy], dpi = dpi)
    ax = plt.axes([xps[0] / fx, (yps[0] + yps[1] * int(with_xlabel)) / fy, xmax / fx, yps[2] / fy])
    
    for i in range(m):
        tmp_xs = np.argwhere(classes == i).reshape([-1])
        if len(tmp_xs) == 0:
            continue
        plt.bar(centers[tmp_xs], data[tmp_xs], width = xps[2], color = class_colors[i])
    
    plt.axis(xmin = 0, xmax = xmax)
    if not (yrng is None):
        plt.axis(ymin = yrng[0], ymax = yrng[1])
    ymin, ymax = plt.axis()[2 : ]
    
    if text:
        if text_ylocs is None:
            text_ylocs = data + 0.02 / yps[2] * ymax
        for i in range(n):
            if percentage:
                tmp_str = "${:.1f}\%$".format(data[i] * 100)
            else:
                tmp_str = "${:.3f}$".format(data[i])
            plt.text(centers[i], text_ylocs[i], tmp_str, ha = "center", va = "bottom", size = "small")
    
    if comp_labels is None:
        comp_labels = ["{:}".format(i + 1) for i in range(n)]
    ax.set_xticks(centers)
    ax.set_xticklabels(comp_labels)
    ax.tick_params(labelsize = "large")
    ax.tick_params(axis = "y", labelcolor = yaxis_color, color = yaxis_color)
    plt.ylabel("Explained TE", color = yaxis_color, size = "large")
    if with_xlabel:
        plt.xlabel(xlabel, size = "large")
    plt.axhline(y = ymin, lw = 1, c = "#000000", clip_on = False)
    plt.axvline(x = 0, lw = 1, c = yaxis_color, clip_on = False)
    plt.axvline(x = xmax, lw = 1, c = acc_color, clip_on = False)
    
    cumsum = np.cumsum(data)
    ax2 = ax.twinx()
    ax2.plot(centers, cumsum, lw = 2, c = acc_color)
    ax2.scatter(centers, cumsum, marker = "D", s = 20, c = acc_color)
    ax2.tick_params(labelsize = "large", labelcolor = acc_color, color = acc_color)
    if yrng2 is None:
        ax2.axis(ymin = 0)
    else:
        ax2.axis(ymin = yrng2[0], ymax = yrng2[1])
    ax2.set_ylabel("Accumulated", color = acc_color, size = "large")
    
    ax.set_frame_on(False)
    ax2.set_frame_on(False)
    
    return fig, ax, ax2

# Comparing multiple values between multiple groups (independent or paired)
# data: [d_1, d_2, ..., d_m], where d_i is [p_i, n] array
# If provided, column_lims should be [n, m, 2] array, and left side space intepreted as column_lims[0, 0, 0], ignoring xps[1 : 4]
def multi_compare(data, field_labels = None, title = None, ylabel = "", paired = True, \
                  show_scatters = True, show_lines = True, show_xaxis = True, show_yaxis = True, \
                  bar_colors = colors, scatter_colors = colors, line_color = "#808080", alpha_bar = 1, alpha_s = 0.3, alpha_l = 0.1, \
                  bar_width = 2, s_size = 2, s_span = 0.6, l_width = 0.5, yrng = None, bar_data = "mean", \
                  fig = None, column_lims = None, xps = [0.7, 0.1, 0.2, 0.08, 0.2, 0.1], yps = [0.3, 2, 0.2, 0.1], dpi = 200):
    def get_scatter_lims(i, j):
        sl = column_lims[i, j, 0] * (0.5 + s_span / 2) + column_lims[i, j, 1] * (0.5 - s_span / 2)
        sr = column_lims[i, j, 0] * (0.5 - s_span / 2) + column_lims[i, j, 1] * (0.5 + s_span / 2)
        
        return sl, sr
    
    m = len(data)
    n = data[0].shape[1]
    ps = np.array([len(di) for di in data])
    for i in range(1, m):
        paired &= ps[i] == ps[0]
    
    with_title = not (title is None)
    
    if column_lims is None:
        column_lims = np.zeros([n, m, 2])
        k = xps[1]
        for i in range(n):
            for j in range(m):
                column_lims[i, j] = k, k + xps[2]
                k += xps[2] + xps[3]
        k += xps[4]
    
    fx = xps[0] + column_lims[-1, -1, 1] + xps[1] + xps[5]
    fy = yps[0] + yps[1] + yps[2] * with_title + yps[3]
    
    if fig is None:
        fig = plt.figure(figsize = [fx, fy], dpi = dpi)
    ax = plt.axes([xps[0] / fx, yps[0] / fy, 1 - (xps[0] + xps[5]) / fx, yps[1] / fy])
    
    # Draw connect lines
    if paired and show_lines:
        for i in range(n):
            for k in range(ps[0]):
                t_xs = np.zeros([m])
                for j in range(m):
                    t_sl, t_sr = get_scatter_lims(i, j)
                    t_xs[j] = t_sl + (t_sr - t_sl) / (ps[0] - 1) * k
                plt.plot(t_xs, [di[k, i] for di in data], lw = l_width, c = line_color, alpha = alpha_l, zorder = -20)
    
    # Draw scatters
    if show_scatters:
        for i in range(n):
            for j in range(m):
                t_sl, t_sr = get_scatter_lims(i, j)
                s_xs = np.linspace(t_sl, t_sr, ps[j], True)
                plt.scatter(s_xs, data[j][ : , i], s = s_size, c = scatter_colors[j], alpha = alpha_s, zorder = -10)
    
    # Draw means
    for i in range(n):
        for j in range(m):
            if type(bar_data) == np.ndarray:
                bar_y = bar_data[i, j]
            elif bar_data == "mean":
                bar_y = data[j][ : , i].mean()
            elif bar_data == "median":
                bar_y = data[j][ : , i].median()
            plt.plot(column_lims[i, j], [bar_y, bar_y], lw = bar_width, c = bar_colors[j], alpha = alpha_bar, zorder = 0)
    
    plt.axis(xmin = 0, xmax = column_lims[-1, -1, 1] + xps[1])
    if not (yrng is None):
        plt.axis(ymin = yrng[0], ymax = yrng[1])
    
    lower_left(ax, show_xaxis = show_xaxis, show_yaxis = show_yaxis)
    if with_title:
        plt.title(title)
    if field_labels is None:
        ax.set_xticks([])
    else:
        ax.set_xticks([(column_lims[i, 0, 0] + column_lims[i, -1, 1]) / 2 for i in range(n)])
        ax.set_xticklabels(field_labels)
    plt.ylabel(ylabel, size = "large")
    
    ax.tick_params(labelsize = "large")
    
    return fig, ax, column_lims

# Draw affinity matrix with topomaps
def disp_group_affinity(aff, f_topomap = None, vx = None, vy = None, names_x = None, names_y = None, symmetric = False, \
                        fy = 8, x_text = 0.7, cmap = "jet", colorbar = False, vmin = None, vmax = None, \
                        text = False, topomap_frame = True, topomap_range = [-0.7, 0.7, -0.7, 0.7], groups = None):
    topomap = not (f_topomap is None)
    if topomap:
        if vy is None:
            vy = vx
    if names_y is None:
        names_y = names_x
    if (vy is None) and (names_y is None):
        symmetric = True
    
    if (vmin is None) and (vmax is None):
        vrng = np.max(np.abs(aff))
        vmin = -vrng
        vmax = vrng
    if vmin is None:
        vmin = np.min(aff)
    if vmax is None:
        vmax = np.max(aff)
    
    n_vx, n_vy = aff.shape
    if names_x is None:
        names_x = [""] * n_vx
    if names_y is None:
        names_y = [""] * n_vy
    u = 0.3
    v = 0.15
    x = (fy - u - 2 * v) / (n_vy + int(topomap))
    y = x_text
    w = 0.2
    z = 0.3
    t = 0.7
    fx = 2 * v + (n_vx + int(topomap)) * x + y
    if colorbar:
        fx += z + w + t - v
    
    fig = plt.figure(figsize = [fx, fy])
    
    if topomap:
        for i in range(n_vx):
            plt.axes([(v + (i + 1) * x + y) / fx, (v + n_vy * x + u) / fy, \
                      x / fx, x / fy])
            f_topomap(vx[i])
            if topomap_frame:
                plt.plot([0.6, 0.6, -0.6, -0.6, 0.6], [-0.6, 0.6, 0.6, -0.6, -0.6], lw = 2, c = "#000000", \
                         solid_capstyle = "round", solid_joinstyle = "round")
            plt.axis(topomap_range)
        
        for i in range(n_vy):
            plt.axes([v / fx, (v + (n_vy - i - 1) * x) / fy, \
                      x / fx, x / fy])
            f_topomap(vy[i])
            if topomap_frame:
                plt.plot([0.6, 0.6, -0.6, -0.6, 0.6], [-0.6, 0.6, 0.6, -0.6, -0.6], lw = 2, c = "#000000", \
                         solid_capstyle = "round", solid_joinstyle = "round")
            plt.axis(topomap_range)
    
    ax = plt.axes([(v + x * int(topomap) + y) / fx, v / fy, \
                   n_vx * x / fx, n_vy * x / fy])
    plt.imshow(aff.T, cmap = cmap, vmin = vmin, vmax = vmax)
    ax.set_xticks(np.arange(n_vx))
    ax.set_xticklabels(names_x, size = "large")
    ax.set_yticks(np.arange(n_vy))
    ax.set_yticklabels(names_y, size = "large")
    ax.xaxis.tick_top()
    
    if text:
        tmp_cm = mpl.colormaps.get_cmap(cmap)
        for i in range(n_vx):
            for j in range(i if symmetric else n_vy):
                tmp_c = tmp_cm((aff[i, j] - vmin) / (vmax - vmin))
                plt.text(i, j, "{:.3f}".format(aff[i, j]), color = "#000000" if luminance(tmp_c) > 0.5 else "#FFFFFF", \
                         size = "large", ha = "center", va = "center")
    
    if not (groups is None):
        for p, q in groups:
            plt.plot([p - 0.5, q - 0.5, q - 0.5, p - 0.5, p - 0.5], [p - 0.5, p - 0.5, q - 0.5, q - 0.5, p - 0.5], \
                     ls = "--", c = "#808080", lw = 5, clip_on = False, zorder = 10)
    
    if colorbar:
        cax = plt.axes([(v + (n_vx + int(topomap)) * x + y + z) / fx, v / fy, \
                        w / fx, n_vy * x / fy])
        cb = plt.colorbar(ax = ax, cax = cax)
        cb.ax.tick_params(labelsize = "large")
    
    return fig, ax