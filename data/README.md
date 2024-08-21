# Setting up Datasets

In order to correctly reproduce data analysis and figures in our manuscript, EEG datasets and other supporting files should be downloaded, correctly preprocessed and placed in `data` directory. Here are instructions for each dataset.

## Dataset 1

Dataset 1 should be stored in `data/MPI_Leipzig_MBB` directory. It consists of resting-state EEG data of 203 subjects provided in the Mind-Brain-Body dataset. For details of usage and access of the original dataset, refer to the paper "Data Descriptor: A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults" (Babayan et al., 2019). You should download the preprocessed version of EEG data (pairs of `.set` and `.fdt` files).

We separated the dataset into a training split and a test split, depending on whether the test subject has a complete list of EEG channels without any marked as "bad". The training split is placed in "data/MPI_Leipzig_MBB" directory and test split is placed in `data/MPI_Leipzig_MBB/ext` directory. The lists of test subject IDs in both splits are stored in `Labels.txt` and `Labels_ext.txt`, respectively. The `.set` and `.fdt` files should be placed directly in the corresponding directory, without creating subdirectories for each test subject.

The subject metadata file `META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv` should be placed in `/data/MPI_Leipzig_MBB/behavioral` directory. We have included the file in this codebase.

## Dataset 2

Dataset 2 should be stored in `data/test_retest` directory. It consists of resting-state and self-paced light task-state EEG data of 60 subjects provided in the Test-Retest Resting and Cognitive State EEG dataset. For details of usage and access of the original dataset, refer to the paper "A test-retest resting, and cognitive state EEG dataset during multiple subject-driven states" (Wang et al., 2022). The preprocessed data in `.set` format, as in the original dataset, should be placed in `data\test_retest\ds004148-download\derivatives\preprocessed_data\preprocessed_data` directory.

We observed that three EEG files might be broken. For each broken EEG file, we provided a `.pickle` file that contains the preprocessing ICA parameters, with filename identical to its corresponding raw data file. The raw data files (in `.vhdr` format) should be downloaded as placed in their corrresponding directories `data/test_retest/ds004148-download/sub-**/ses-session*/eeg`. Then, for each of the three broken files, open `code/preprocess_ds2_trt.py` and edit the variables `raw_fn` and `prep_fn` to match the file names, and run the script to perform automatic preprocessing.

The subject metadata file `participants.tsv` should be placed in `data/test_retest/ds004148-download` directory. We have included the file in this codebase.

## Dataset 3

Dataset 3 should be stored in `data/sedation` directory. It consists of EEG recordings of 20 subjects during four stages of sedation, provided in the Brain Connectivity during Propofol Sedation dataset. For details of usage and access of the original dataset, refer to the paper "Brain Connectivity Dissociates Responsiveness from Drug Exposure during Propofol-Induced Transitions of Consciousnes" (Chennu et al., 2016). The preprocessed data in `.set` format should be placed directly in `data/sedation` directory, without creating subdirectories for each test subject.

The subject metadata file `datainfo.mat` should also be placed in `data/sedation` directory. We have included the file in this codebase.