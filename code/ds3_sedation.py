"""
ds3_sedation.py
Loading and processing the Sedation dataset

Version 231219.01
Yichao Li
"""

import numpy as np
import mne
from scipy import io

import core

path = "../data/sedation/"
set_info = io.loadmat(path + "datainfo.mat")["datainfo"]

stages = np.array([t[1][0, 0] for t in set_info])
ppfc = np.array([t[2][0, 0] for t in set_info])
rt = np.array([t[3][0, 0] for t in set_info])
rt[np.isnan(rt)] = 3000
qs = np.array([t[4][0, 0] for t in set_info])

full_fn = [i[0][0] for i in set_info]
filt_freq = [7, 13]

raw = mne.io.read_epochs_eeglab(path + full_fn[0] + ".set")
info = raw.info.copy()
sfreq = info["sfreq"]

n_subjs = 20
n_states = 4
n_fn_full = n_subjs * n_states

def reorder_channels(info0, info1):
    n = len(info0["ch_names"])
    inds = np.zeros([n], dtype = int)
    for i in range(n):
        inds[i] = core.chname2ind(info1, info0["ch_names"][i])
    
    return inds

def fetch_all_segments(fn):
    raw = mne.io.read_epochs_eeglab(path + fn + ".set").load_data()
    raw = raw.filter(l_freq = filt_freq[0], h_freq = filt_freq[1], method = "iir")
    raw.set_eeg_reference("average")
    raw = raw.apply_hilbert()
    
    res = []
    for i in range(len(raw)):
        dt = raw[i].get_data().squeeze().T
        res.append(core.raw_segment(dt[ : , reorder_channels(info, raw.info)], raw.info.copy()))
    
    return res

def iter_unfiltered(i):
    raw = mne.io.read_epochs_eeglab(path + full_fn[i] + ".set").load_data()
    raw.set_eeg_reference("average")
    
    res = []
    for i in range(len(raw)):
        res.append(raw[i].get_data().squeeze().T)
    
    return res

def iter_cms(i):
    tmp = fetch_all_segments(full_fn[i])
    res = []
    for s in tmp:
        res.append(s.to_cms())
    
    return res

def iter_twms(i):
    tmp = fetch_all_segments(full_fn[i])
    res = []
    for s in tmp:
        res.append(s.to_twms())
    
    return res
