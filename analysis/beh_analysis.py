
from pathlib import Path
import numpy as np
import scipy
from scipy import stats
from mne import read_epochs
import pandas as pd
import slab
import os
from analysis.plotting.localization_plot import localization_accuracy as la
pd.set_option('display.max_rows', 1000, 'display.max_columns', 200, 'display.width', 99999)



eeg_dir = Path.cwd() / 'data' / 'experiment' / 'results' / 'decoding'
beh_dir = Path.cwd() / 'data' / 'experiment' / 'behavior' / 'localization'

def get_beh_df():
    columns = ['Subject', 'EF', 'M1 D1', 'M1 D2', 'M1 D3',
               'M1 D4', 'M1 D5', 'M1 D6', 'M1 D7']
    seq_df = pd.DataFrame(columns=columns)
    for sub in beh_dir.glob("P?*"):
        subject_data = []
        for filename in sub.glob('localization*'):
            trial_seq = slab.Trialsequence(filename) # get trial sequence
            elevation_gain, ele_rmse, ele_sd, az_rmse, az_sd = la(trial_seq, show=False) # get behavior data
            subject_data.append([elevation_gain, ele_rmse, ele_sd])  # append behavior data of a single subject to a list
        subject_row = {'Subject': sub.name, 'EF': subject_data[0], 'M1 D1': subject_data[1], 'M1 D2': subject_data[2],
                       'M1 D3': subject_data[3], 'M1 D4': subject_data[4], 'M1 D5': subject_data[5],
                       'M1 D6': subject_data[6], 'M1 D7': subject_data[7]}  # create new row
        seq_df = seq_df._append(subject_row, ignore_index = True)  # append row to df
    return seq_df

def get_eeg_df():
    columns = ['Subject', 'EF', 'M1 D1', 'M1 D7']
    eeg_df = pd.DataFrame(columns=columns)
    for sub in eeg_dir.glob("P?*"):
        subject_data = []
        for filename in sub.glob('*'):
            decoding_dict = np.load(filename, allow_pickle=True)
            subject_data.append(decoding_dict)  # append eeg data of a single subject to a list
        subject_row = {'Subject': sub.name, 'EF': subject_data[0], 'M1 D1': subject_data[1],
                       'M1 D7': subject_data[2]}  # create new row
        eeg_df = eeg_df._append(subject_row, ignore_index=True)  # append row to df







# paired t-test
scipy.stats.ttest_rel(con1, con2, axis=0, nan_policy='propagate', alternative='two-sided', *, keepdims=False)



# tryouts:
# dict = [{'subject': '', 'EG EF': '', 'EG M1 D1': [3], 'EG M1 D2': [4], 'EG M1 D3': [5], 'EG M1 D4': [6], 'EG M1 D5': [7], 'EG M1 D6': [8], 'EG M1 D7': [9] }]
# seq_df = pd.DataFrame(dict)
# subject_id = {'subject': ['P1', 'P2', 'P3', 'P4', 'P5']}
# seq_df['subject'] = seq_df['subject'].fill(subject_id)
# seq_df.loc[:]['subject'] = 'P1'
# new_row = {'subject': subject_id}
# subject_id = pd.DataFrame({'subject': ['P1', 'P2', 'P3', 'P4', 'P5']})
# seq_df['subject'] = subject_id
