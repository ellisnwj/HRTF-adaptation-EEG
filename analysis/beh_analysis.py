from pathlib import Path
import numpy as np
import scipy
from scipy import stats
from mne import read_epochs
import pandas as pd
import slab
import os
from analysis.plotting.localization_plot import localization_accuracy as la
import analysis.localization_analysis as loc_analysis


eeg_dir = Path.cwd() / 'data' / 'experiment' / 'results' / 'preprocessed'
beh_dir = Path.cwd() / 'data' / 'experiment' / 'behavior' / 'localization'


def get_df():    # beh and eeg path

    columns = ['EG EF', 'EG M1 D1', 'EG M1 D2', 'EG M1 D3',
               'EG M1 D4', 'EG M1 D5', 'EG M1 D6', 'EG M1 D7']
    index_ = ['P1', 'P2', 'P3', 'P4', 'P5']
    seq_df = pd.DataFrame(columns=columns, index=index_)
    print(seq_df)


    for sub in beh_dir.glob("P?*"):
        print(sub)
        for ses in enumerate(sub.glob("ses*")):
            print(ses)
            for filename in ses.glob('*'):
                print(ses)
                trial_seq =
                elevation_gain =
                ele_rmse =
                ele_sd =
                session = ses.name
                new_row = {'sessions': session}
                print(ses)
            # todo create a new row
                seq_df = seq_df._append(new_row, ignore_index = True)
                (list(loc_analysis.localization_accuracy(sequence, show=False)))


            print(ses)
            sequence = slab.Trialsequence().load_pickle(list(ses.glob('*'))[0])
            elevation_gain, ele_rmse, ele_sd, az_rmse, az_sd = la(sequence, show=False)
            # todo write into that row the sequence


df = pd.DataFrame()
df.columns


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
