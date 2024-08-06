"""
ds2_trt.py
Loading and processing the Test and Retest dataset

Version 240222.01
Yichao Li
"""

import numpy as np
import pandas as pd
import mne
import os

import core

def get_full_fn(subj_i, sess_i, state_i):
    return subj_ids[subj_i] + "_0" + str(sess_i + 1) + "_" + state_names[state_i]

state_names = ["EC", "EO", "Ma", "Me", "Mu"]

meta_path = "../data/test_retest/ds004148-download/"
path = "../data/test_retest/ds004148-download/derivatives/preprocessed_data/preprocessed_data/"
path_list = sorted(os.listdir(path))

subj_ids = []
for st in path_list:
    if st.endswith("01_EC.set"):
        subj_ids.append(st[ : 5])

n_subjs = len(subj_ids)
n_sess = 3
n_states = len(state_names)

filt_freq = [7, 13]
sfreq = 250

raw_ref = mne.io.read_raw_eeglab(path + get_full_fn(0, 0, 0) + ".set", preload = True)
raw_ref.set_montage(mne.channels.make_standard_montage("standard_1020"))
info = raw_ref.info.copy()
sfreq_0 = info["sfreq"]

def get_age(subj_ids, FN = "participants.tsv"):
    raw = pd.read_csv(meta_path + FN, sep = "\t")
    
    return [raw["age"][np.argwhere(np.array(raw["participant_id"] == st[ : 3] + "-" + st[3 : ]))[0][0]] for st in subj_ids]

def match_channels(raw, info):
    n_ch_raw = len(raw.info["ch_names"])
    n_ch_ref = len(info["ch_names"])
    matches = -np.ones([n_ch_ref], dtype = int)
    rename = dict()
    for i in range(n_ch_ref):
        tmp_ref_name = info["ch_names"][i].upper()
        for j in range(n_ch_raw):
            if tmp_ref_name == raw.info["ch_names"][j].upper():
                matches[i] = j
                break
        if matches[i] != -1:
            if info["ch_names"][i] != raw.info["ch_names"][matches[i]]:
                rename[raw.info["ch_names"][matches[i]]] = info["ch_names"][i]
        else:
            print("Unknown channel name")
            return None
    
    raw = raw.rename_channels(rename).reorder_channels(info["ch_names"])
    
    return raw

def fetch_all_segments(fn):
    if os.path.exists(path + fn + ".fif"):
        raw = mne.io.read_raw_fif(path + fn + ".fif", preload = True)
    else:
        raw = mne.io.read_raw_eeglab(path + fn + ".set", preload = True)
    raw = match_channels(raw, info)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"))
    raw.resample(sfreq)
    
    raw = raw.filter(*filt_freq, method = "iir")
    raw.set_eeg_reference("average")
    raw = raw.apply_hilbert()
    
    return core.raw_segment(raw.get_data().squeeze().T, raw.info.copy())

def iter_cms(subj_i, sess_i, state_i):
    return fetch_all_segments(get_full_fn(subj_i, sess_i, state_i)).to_cms()

def iter_twms(subj_i, sess_i, state_i):
    return fetch_all_segments(get_full_fn(subj_i, sess_i, state_i)).to_twms()