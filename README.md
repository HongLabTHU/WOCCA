# Codebase of the WOCCA algorithm

This codebase contains the implementation of the WOCCA algorithm proposed in the manuscript "[Spatiotemporal Decomposition of Whole-Brain Alpha Traveling Waves](https://doi.org/10.1101/2024.08.23.609472)", supplementary materials of this manuscript, and notebooks for reproduction of EEG dataset analysis, written in Python language with Jupyter Notebook.

## Quick Start
For a quick start on how to use WOCCA, refer to `grid_samples.py` for examples on synthetic grid data. For reproduction of analysis and creation of figures in the manuscript, follow the instructions in the `data` directory to set up the EEG datasets, before referring to the Jupyter Notebooks.

## Requirements
```
mne
torch
scipy
statsmodels
scikit-learn
pandas
matplotlib
```

## Contents
### The core WOCCA algorithm
- `wocca.py`: Main implementation of the WOCCA algorithm

### Supporting Code
- `core.py`: Utilities and definition of data structures
- `gradient.py`: Tools for processing phase gradient field
- `grid_samples.py`: Tools for synthetic data
- `icons.py`: Draw various icons for figures in the manuscript
- `ring.py`: Experimental dynamical analysis tools, only used for generating geodesic interpolation of synthetic data
- `templates.py`: Microstate analysis
- `visualize.py`: Various visualization tools

### Datasets
- `ds1_leipzig.py`
- `ds2_trt.py`
- `ds3_sedation.py`
- `preprocess_ds2_trt.py`: Manual preprocessing pipeline for Dataset 2

### Analysis and Figures
- `nb1_explore_ds1_leipzig.ipynb`: Exploration of Dataset 1 (Leipzig)
- `nb2_explore_ds2_trt.ipynb`: Exploration of Dataset 2 (TRT)
- `nb3_explore_wocca_consistency.ipynb`: Robustness analysis between Dataset 1 and 2
- `nb4_explore_microstate.ipynb`: Microstate analysis of Dataset 1
- `nb5_explore_ds3_sedation.ipynb`: Exploration of Dataset 3 (Sedation)
- `nb6_intro_and_toy_data.ipynb`: Introductory diagrams and synthetic data analysis

### Supplementary information
- `supplementary_information.pdf`: Supplementary details of mathematics, statistics and data analysis
