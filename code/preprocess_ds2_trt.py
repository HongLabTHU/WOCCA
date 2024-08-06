"""
preprocess_ds2_trt.py
Manual preprocessing the Test and Retest dataset

Version 231124.01
Yichao Li
"""

import numpy as np
import pandas as pd
import mne
import pickle
import os

def fit_ica(raw_concat, random_state):
    ica_inst = mne.preprocessing.ICA(n_components = ica_n_comps, random_state = random_state)
    ica_inst.fit(raw_concat)
    
    return ica_inst, random_state

def view_ica(raw_concat, ica_inst, exclude_comps = None):
    ica_inst.plot_sources(raw_concat)
    ica_inst.plot_components()
    
    if not (exclude_comps is None):
        ica_inst.plot_properties(raw_concat, picks = exclude_comps)

def apply_ica(raw_concat, ica_inst, exclude_comps):
    raw_ica = raw_concat.copy()
    ica_inst.apply(raw_ica, exclude = exclude_comps)
    
    return raw_ica

if __name__ == "__main__":
    ica_random_seed = 1
    ica_n_comps = 20
    
    raw_path = "../data/test_retest/ds004148-download/sub-33/ses-session2/eeg/"
    raw_fn = "sub-33_ses-session2_task-eyesopen_eeg.vhdr"
    prep_path = "../data/test_retest/ds004148-download/derivatives/preprocessed_data/preprocessed_data/"
    prep_fn = "sub33_02_EO.fif"
    
    random_state = np.random.RandomState(seed = ica_random_seed)
    
    # %% Load data, average reference and filtering
    
    raw = mne.io.read_raw(raw_path + raw_fn, preload = True)
    raw.rename_channels({"Cpz" : "CPz"})
    
    raw.set_eeg_reference("average")
    raw.filter(l_freq = 0.3, h_freq = 45, method = "fir")
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"))
    
    # %% Fit ICA
    
    ica_inst, random_state = fit_ica(raw, random_state)
    
    # %% Check if ICA selection already exists
    
    ica_exist_flag = False
    ica_pickle_fn = prep_path + raw_fn + ".ica_exclude.pickle"
    if os.path.exists(ica_pickle_fn):
        with open(ica_pickle_fn, mode = "rb") as FIn:
            ica_random_seed_1, ica_n_comps_1, exclude_comps = pickle.load(FIn)
        if (ica_random_seed_1 == ica_random_seed) and (ica_n_comps_1 == ica_n_comps):
            ica_exist_flag = True
            print("Load ICA selection record")
    
    if not ica_exist_flag:
        print("Require manual ICA selection")
        
        # %% Visualize ICA
        
        raw.plot()
        view_ica(raw, ica_inst)
        
        # %% Choose components
        
        exclude_comps = [0, 1, 3, 17]
        view_ica(raw, ica_inst, exclude_comps)
    
        with open(ica_pickle_fn, mode = "wb") as FOut:
            pickle.dump((ica_random_seed, ica_n_comps, exclude_comps), FOut)
    
    # %% Apply ICA but does not re-reference
    
    raw_ica = apply_ica(raw, ica_inst, exclude_comps)
    raw_ica.plot()
    
    # %% Save preprocessed data in .fif format
    
    raw_ica.save(prep_path + prep_fn)