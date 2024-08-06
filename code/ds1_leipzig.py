"""
ds1_leipzig.py
Loading and processing the Leipzig dataset

Version 240221.01
Yichao Li
"""

import numpy as np
import pandas as pd
import mne
import os

import core

path = r"../data/MPI_Leipzig_MBB/"
path_list = sorted(os.listdir(path))
subj_ids = []
for st in path_list:
    if st.endswith(".set") and (st[12] == "C"):
        subj_ids.append(st[ : 10])
full_fn = []
for st in subj_ids:
    for suffix in ["_EC", "_EO"]:
        full_fn.append(st + suffix)

path_list_ext = sorted(os.listdir(path + "ext/"))
subj_ids_ext = []
for st in path_list_ext:
    if st.endswith(".set") and (st[12] == "C"):
        subj_ids_ext.append(st[ : 10])
for st in subj_ids_ext:
    for suffix in ["_EC", "_EO"]:
        full_fn.append("ext/" + st + suffix)

filt_freq = [7, 13]
t_segment = 0

raw_ref = mne.io.read_raw_eeglab(path + full_fn[0] + ".set").load_data()
info = raw_ref.info.copy()
sfreq = info["sfreq"]

n_subj_core = 83
n_subj_full = 203
n_fn_core = 166
n_fn_full = 406

behavioral_path = path + "behavioral/"

def set_t_segment(tseg):
    global t_segment
    
    t_segment = tseg

def get_gender(subj_ids, FN = "META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"):
    raw = pd.read_csv(behavioral_path + FN)
    
    return [raw["Gender_ 1=female_2=male"][np.argwhere(np.array(raw["ID"] == st))[0][0]] for st in subj_ids]

def get_age(subj_ids, FN = "META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"):
    raw = pd.read_csv(behavioral_path + FN)
    
    return [raw["Age"][np.argwhere(np.array(raw["ID"] == st))[0][0]] for st in subj_ids]

def get_handedness(subj_ids, FN = "META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"):
    raw = pd.read_csv(behavioral_path + FN)
    
    return [raw["Handedness"][np.argwhere(np.array(raw["ID"] == st))[0][0]] for st in subj_ids]

def get_tmta(subj_ids, FN = "Cognitive_Test_Battery_LEMON/TMT/TMT.csv"):
    raw = pd.read_csv(behavioral_path + FN)
    
    return [raw["TMT_1"][np.argwhere(np.array(raw["ID"] == st))[0][0]] for st in subj_ids]

def get_tmtb(subj_ids, FN = "Cognitive_Test_Battery_LEMON/TMT/TMT.csv"):
    raw = pd.read_csv(behavioral_path + FN)
    
    return [raw["TMT_5"][np.argwhere(np.array(raw["ID"] == st))[0][0]] for st in subj_ids]

def get_cvlt_delay(subj_ids, FN = "Cognitive_Test_Battery_LEMON/CVLT (1)/CVLT.csv"):
    raw = pd.read_csv(behavioral_path + FN)
    
    return [raw["CVLT_11"][np.argwhere(np.array(raw["ID"] == st))[0][0]] for st in subj_ids]

def get_cvlt_recog(subj_ids, FN = "Cognitive_Test_Battery_LEMON/CVLT (1)/CVLT.csv"):
    raw = pd.read_csv(behavioral_path + FN)
    
    return [raw["CVLT_13"][np.argwhere(np.array(raw["ID"] == st))[0][0]] for st in subj_ids]

def repair_with_interpolation(raw0, _raw_ref):
    raw_ref = _raw_ref.copy()
    raw_ch_list = raw0.info["ch_names"]
    ref_ch_list = raw_ref.info["ch_names"]
    missing_list = []
    for i in ref_ch_list:
        if not (i in raw_ch_list):
            missing_list.append(i)
    
    if len(missing_list) == 0:
        return raw0
    
    raw_ref = raw_ref.pick_channels(missing_list)
    if raw_ref.info["sfreq"] != raw0.info["sfreq"]:
        raw_ref.resample(raw0.info["sfreq"])
    while len(raw_ref) <= len(raw0):
        raw_ref.append(raw_ref)
    raw_ref = raw_ref.crop(tmax = len(raw0) / raw_ref.info["sfreq"], include_tmax = False)
    
    raw0.add_channels([raw_ref], force_update_info = True)
    raw0.info["bads"] = missing_list
    raw0.interpolate_bads()
    
    redundant_list = []
    for i in raw_ch_list:
        if not (i in ref_ch_list):
            redundant_list.append(i)
    raw0.drop_channels(redundant_list)
    raw0.reorder_channels(ref_ch_list)
    
    return raw0

def fetch_all_segments(fn):
    raw = mne.io.read_raw_eeglab(path + fn + ".set").load_data()
    if fn[ : 4] == "ext/":
        raw = repair_with_interpolation(raw, raw_ref)
    
    raw = raw.filter(l_freq = filt_freq[0], h_freq = filt_freq[1], method = "iir")
    raw.set_eeg_reference("average")
    raw = raw.apply_hilbert()
    
    if t_segment > 0:
        raw_data = raw.get_data().squeeze().T
        n_segment = round(t_segment * sfreq)
        res = []
        k = (len(raw_data) % n_segment) // 2
        while k + n_segment <= len(raw_data):
            res.append(core.raw_segment(raw_data[k : k + n_segment], raw.info.copy()))
            k += n_segment
        
        return res
    
    return [core.raw_segment(raw.get_data().squeeze().T, raw.info.copy())]

def iter_unfiltered(i):
    raw = mne.io.read_raw_eeglab(path + full_fn[i] + ".set").load_data()
    if full_fn[i][ : 4] == "ext/":
        raw = repair_with_interpolation(raw, raw_ref)
    
    raw.set_eeg_reference("average")
    
    if t_segment > 0:
        raw_data = raw.get_data().squeeze().T
        n_segment = round(t_segment * sfreq)
        res = []
        k = (len(raw_data) % n_segment) // 2
        while k + n_segment <= len(raw_data):
            res.append(raw_data[k : k + n_segment])
            k += n_segment
        
        return res
    
    return [raw.get_data().squeeze().T]

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
