"""
icons.py
Drawing icons for traveling wave patterns

Version 240124.01
Yichao Li
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import visualize as vis

# Prepare standard icon canvas
def prepare_icon(figsize = [1, 1], dpi = 300):
    fig = plt.figure(figsize = figsize, dpi = dpi)
    ax = plt.axes([0, 0, 1, 1])
    vis.no_axes(ax)
    plt.axis([-0.5, 0.5, -0.5, 0.5])
    
    return fig, ax

def draw_arrow_head(x0, v0, color, head_width, overhang):
    vl = np.sqrt(v0[0] ** 2 + v0[1] ** 2)
    x = [x0[i] - v0[i] * overhang for i in range(2)]
    plt.arrow(x[0], x[1], v0[0], v0[1], length_includes_head = True, width = 0, head_length = vl, head_width = head_width, \
              head_starts_at_zero = False, overhang = overhang, fc = color, ec = "#00000000", lw = 0, zorder = 10, clip_on = False)

# Traveling energy
def traveling_energy(line_color, light_color, scale = 1):
    thetas = np.linspace(0, np.pi * 3, 541, True)
    plt.plot(np.linspace(-0.4, 0.2, 541, True), np.cos(thetas) * 0.15 + 0.25, lw = 4 * scale, \
             c = light_color, solid_capstyle = "butt", solid_joinstyle = "round")
    plt.plot(np.linspace(-0.2, 0.4, 541, True), np.cos(thetas) * 0.15 + 0.05, lw = 4 * scale, \
             c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    plt.plot([-0.24, 0.24], [-0.3, -0.3], lw = 6 * scale, c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    draw_arrow_head([0.24, -0.3], [0.3, 0], line_color, 0.25, 0.3)
    draw_arrow_head([-0.24, -0.3], [-0.3, 0], line_color, 0.25, 0.3)

# Asymmetry
def asymmetry(line_color, light_color, scale = 1):
    thetas = np.linspace(0, np.pi * 3, 541, True)
    plt.plot(np.linspace(-0.4, 0.05, 541, True), -np.cos(thetas) * 0.08 + 0.1, lw = 4 * scale, \
             c = light_color, solid_capstyle = "butt", solid_joinstyle = "round")
    plt.plot(np.linspace(-0.15, 0.4, 541, True), np.cos(thetas) * 0.15 - 0.05, lw = 4 * scale, \
             c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    plt.plot([-0.31, 0.05], [0.35, 0.35], lw = 5 * scale, c = light_color, solid_capstyle = "butt", solid_joinstyle = "round")
    draw_arrow_head([-0.31, 0.35], [-0.2, 0], light_color, 0.12, 0.3)
    plt.plot([-0.2, 0.24], [-0.35, -0.35], lw = 5 * scale, c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    draw_arrow_head([0.24, -0.35], [0.3, 0], line_color, 0.2, 0.3)

# Rotation
def rotation(line_color, center_color, scale = 1):
    thetas = np.linspace(0, np.pi * 2, 361, True)
    plt.plot(np.cos(thetas) * 0.35, np.sin(thetas) * 0.35, lw = 6 * scale, c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    plt.scatter([0], [0], s = 35 * (scale ** 2), c = center_color, linewidth = 0)

def rotation_ccw(line_color, center_color, scale = 1):
    thetas = np.linspace(-np.pi / 2, np.pi * 7 / 6, 301, True)
    plt.plot(np.cos(thetas) * 0.35, np.sin(thetas) * 0.35, lw = 6 * scale, c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    plt.scatter([0], [0], s = 35 * (scale ** 2), c = center_color, linewidth = 0)
    draw_arrow_head([-0.35 * np.cos(np.pi / 6), -0.35 * np.sin(np.pi / 6)], [0.24 * np.sin(np.pi / 6), -0.24 * np.cos(np.pi / 6)], \
                    line_color, 0.2, 0.3)

def rotation_cw(line_color, center_color, scale = 1):
    thetas = np.linspace(-np.pi / 6, np.pi * 3 / 2, 301, True)
    plt.plot(np.cos(thetas) * 0.35, np.sin(thetas) * 0.35, lw = 6 * scale, c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    plt.scatter([0], [0], s = 35 * (scale ** 2), c = center_color, linewidth = 0)
    draw_arrow_head([0.35 * np.cos(np.pi / 6), -0.35 * np.sin(np.pi / 6)], [-0.24 * np.sin(np.pi / 6), -0.24 * np.cos(np.pi / 6)], \
                    line_color, 0.2, 0.3)

# Rot.A and Rot.P
def rotation_anterior(line_color, center_color, scale = 1):
    thetas = np.linspace(0, np.pi * 2, 361, True)
    plt.plot(np.cos(thetas) * 0.35, np.sin(thetas) * 0.35, lw = 6 * scale, c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    plt.scatter([0], [0.15], marker = "s", s = 35 * (scale ** 2), c = center_color, linewidth = 0)

def rotation_posterior(line_color, center_color, scale = 1):
    thetas = np.linspace(0, np.pi * 2, 361, True)
    plt.plot(np.cos(thetas) * 0.35, np.sin(thetas) * 0.35, lw = 6 * scale, c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    plt.scatter([0], [-0.15], marker = "s", s = 35 * (scale ** 2), c = center_color, linewidth = 0)

# Longitudinal
def longitudinal(line_color, scale = 1):
    plt.plot([0, 0], [-0.24, 0.24], lw = 6 * scale, c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    draw_arrow_head([0, -0.24], [0, -0.3], line_color, 0.25, 0.3)
    draw_arrow_head([0, 0.24], [0, 0.3], line_color, 0.25, 0.3)

def longitudinal_fw(line_color, scale = 1):
    plt.plot([0, 0], [-0.4, 0.24], lw = 6 * scale, c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    draw_arrow_head([0, 0.24], [0, 0.3], line_color, 0.25, 0.3)

def longitudinal_bw(line_color, scale = 1):
    plt.plot([0, 0], [-0.24, 0.4], lw = 6 * scale, c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    draw_arrow_head([0, -0.24], [0, -0.3], line_color, 0.25, 0.3)

# Horizontal
def horizontal(line_color, scale = 1):
    plt.plot([-0.24, 0.24], [0, 0], lw = 6 * scale, c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    draw_arrow_head([-0.24, 0], [-0.3, 0], line_color, 0.25, 0.3)
    draw_arrow_head([0.24, 0], [0.3, 0], line_color, 0.25, 0.3)

def horizontal_rw(line_color, scale = 1):
    plt.plot([-0.4, 0.24], [0, 0], lw = 6 * scale, c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    draw_arrow_head([0.24, 0], [0.3, 0], line_color, 0.25, 0.3)

def horizontal_lw(line_color, scale = 1):
    plt.plot([-0.24, 0.4], [0, 0], lw = 6 * scale, c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    draw_arrow_head([-0.24, 0], [-0.3, 0], line_color, 0.25, 0.3)

# Lateral
def lateral(line_color, scale = 1):
    thetas = np.linspace(-np.pi / 6, np.pi / 6, 61, True)
    plt.plot(np.cos(thetas) * 0.5 - 0.25, np.sin(thetas) * 0.5, lw = 5 * scale, c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    plt.plot(-np.cos(thetas) * 0.5 + 0.25, np.sin(thetas) * 0.5, lw = 5 * scale, c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    
    draw_arrow_head([-0.25 + np.cos(np.pi / 6) * 0.5, np.sin(np.pi / 6) * 0.5], [-0.24 * np.sin(np.pi / 6), 0.24 * np.cos(np.pi / 6)], \
                    line_color, 0.2, 0.3)
    draw_arrow_head([-0.25 + np.cos(np.pi / 6) * 0.5, -np.sin(np.pi / 6) * 0.5], [-0.24 * np.sin(np.pi / 6), -0.24 * np.cos(np.pi / 6)], \
                    line_color, 0.2, 0.3)
    draw_arrow_head([0.25 - np.cos(np.pi / 6) * 0.5, np.sin(np.pi / 6) * 0.5], [0.24 * np.sin(np.pi / 6), 0.24 * np.cos(np.pi / 6)], \
                    line_color, 0.2, 0.3)
    draw_arrow_head([0.25 - np.cos(np.pi / 6) * 0.5, -np.sin(np.pi / 6) * 0.5], [0.24 * np.sin(np.pi / 6), -0.24 * np.cos(np.pi / 6)], \
                    line_color, 0.2, 0.3)

def lateral_fs(line_color, scale = 1):
    thetas = np.linspace(-np.pi / 180 * 45, np.pi / 6, 76, True)
    plt.plot(np.cos(thetas) * 0.5 - 0.25, np.sin(thetas) * 0.5, lw = 5 * scale, c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    plt.plot(-np.cos(thetas) * 0.5 + 0.25, np.sin(thetas) * 0.5, lw = 5 * scale, c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    
    draw_arrow_head([-0.25 + np.cos(np.pi / 6) * 0.5, np.sin(np.pi / 6) * 0.5], [-0.24 * np.sin(np.pi / 6), 0.24 * np.cos(np.pi / 6)], \
                    line_color, 0.2, 0.3)
    draw_arrow_head([0.25 - np.cos(np.pi / 6) * 0.5, np.sin(np.pi / 6) * 0.5], [0.24 * np.sin(np.pi / 6), 0.24 * np.cos(np.pi / 6)], \
                    line_color, 0.2, 0.3)

def lateral_bs(line_color, scale = 1):
    thetas = np.linspace(-np.pi / 6, np.pi / 180 * 45, 76, True)
    plt.plot(np.cos(thetas) * 0.5 - 0.25, np.sin(thetas) * 0.5, lw = 5 * scale, c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    plt.plot(-np.cos(thetas) * 0.5 + 0.25, np.sin(thetas) * 0.5, lw = 5 * scale, c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    
    draw_arrow_head([-0.25 + np.cos(np.pi / 6) * 0.5, -np.sin(np.pi / 6) * 0.5], [-0.24 * np.sin(np.pi / 6), -0.24 * np.cos(np.pi / 6)], \
                    line_color, 0.2, 0.3)
    draw_arrow_head([0.25 - np.cos(np.pi / 6) * 0.5, -np.sin(np.pi / 6) * 0.5], [0.24 * np.sin(np.pi / 6), -0.24 * np.cos(np.pi / 6)], \
                    line_color, 0.2, 0.3)

# Wavelengths
def longitudinal_wave(line_color, wave_color, cycles = 1, scale = 1):
    thetas = np.linspace(0, np.pi * 2 * cycles, 361, True)
    plt.plot(-np.sin(thetas) * 0.2 + 0.2, np.linspace(-0.4, 0.4, 361, True), lw = 5 * scale, c = wave_color, solid_capstyle = "butt", solid_joinstyle = "round")
    
    plt.plot([-0.25, -0.25], [-0.24, 0.24], lw = 6 * scale, c = line_color, solid_capstyle = "butt", solid_joinstyle = "round")
    draw_arrow_head([-0.25, -0.24], [0, -0.3], line_color, 0.25, 0.3)
    draw_arrow_head([-0.25, 0.24], [0, 0.3], line_color, 0.25, 0.3)

# %% Draw and save
if __name__ == "__main__":
    fig_path = "../wocca_project_figs/icons/"
    line_color = "#607280"
    light_color = "#AFB8BF"
    center_color = "#607280"
    unresp_color = "#FF6B5C"
    baseline_color = "#49ADFF"
    center_a_color = vis.sat_mod_white(unresp_color, 0.5)
    center_p_color = vis.sat_mod_white(baseline_color, 0.5)
    
    fig, ax = prepare_icon()
    traveling_energy(line_color, light_color)
    fig.savefig(fig_path + "te.svg", format = "svg")
    plt.close(fig)
    
    fig, ax = prepare_icon()
    asymmetry(line_color, light_color)
    fig.savefig(fig_path + "asym.svg", format = "svg")
    plt.close(fig)
    
    fig, ax = prepare_icon()
    rotation(line_color, center_color)
    fig.savefig(fig_path + "rot.svg", format = "svg")
    plt.close(fig)
    
    fig, ax = prepare_icon()
    rotation_ccw(line_color, center_color)
    fig.savefig(fig_path + "rot_ccw.svg", format = "svg")
    plt.close(fig)
    
    fig, ax = prepare_icon()
    rotation_cw(line_color, center_color)
    fig.savefig(fig_path + "rot_cw.svg", format = "svg")
    plt.close(fig)
    
    fig, ax = prepare_icon()
    rotation_anterior(line_color, center_a_color)
    fig.savefig(fig_path + "ra.svg", format = "svg")
    plt.close(fig)
    
    fig, ax = prepare_icon()
    rotation_posterior(line_color, center_p_color)
    fig.savefig(fig_path + "rp.svg", format = "svg")
    plt.close(fig)
    
    fig, ax = prepare_icon()
    longitudinal(line_color)
    fig.savefig(fig_path + "longi.svg", format = "svg")
    plt.close(fig)
    
    fig, ax = prepare_icon()
    longitudinal_fw(line_color)
    fig.savefig(fig_path + "longi_fw.svg", format = "svg")
    plt.close(fig)
    
    fig, ax = prepare_icon()
    longitudinal_bw(line_color)
    fig.savefig(fig_path + "longi_bw.svg", format = "svg")
    plt.close(fig)
    
    fig, ax = prepare_icon()
    horizontal(line_color)
    fig.savefig(fig_path + "horiz.svg", format = "svg")
    plt.close(fig)
    
    fig, ax = prepare_icon()
    horizontal_rw(line_color)
    fig.savefig(fig_path + "horiz_rw.svg", format = "svg")
    plt.close(fig)
    
    fig, ax = prepare_icon()
    horizontal_lw(line_color)
    fig.savefig(fig_path + "horiz_lw.svg", format = "svg")
    plt.close(fig)
    
    fig, ax = prepare_icon()
    lateral(line_color)
    fig.savefig(fig_path + "lat.svg", format = "svg")
    plt.close(fig)
    
    fig, ax = prepare_icon()
    lateral_fs(line_color)
    fig.savefig(fig_path + "lat_fs.svg", format = "svg")
    plt.close(fig)
    
    fig, ax = prepare_icon()
    lateral_bs(line_color)
    fig.savefig(fig_path + "lat_bs.svg", format = "svg")
    plt.close(fig)
    
    fig, ax = prepare_icon()
    longitudinal_wave(line_color, baseline_color, 1)
    fig.savefig(fig_path + "longi_long.svg", format = "svg")
    plt.close(fig)
    
    fig, ax = prepare_icon()
    longitudinal_wave(line_color, unresp_color, 2)
    fig.savefig(fig_path + "longi_short.svg", format = "svg")
    plt.close(fig)