import seaborn
from matplotlib import pyplot as plt
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
               'M1 D4', 'M1 D5', 'M1 D6']
    seq_df = pd.DataFrame(columns=columns)
    for sub in beh_dir.glob("P?*"):
        subject_data = []
        for filename in sub.glob('localization*'):
            trial_seq = slab.Trialsequence(filename) # get trial sequence
            elevation_gain, ele_rmse, ele_sd, az_rmse, az_sd = la(trial_seq, show=False) # get behavior data
            subject_data.append([elevation_gain, ele_rmse, ele_sd])  # append behavior data of a single subject to a list
        subject_row1 = {'Subject': sub.name, 'EF': subject_data[0], 'M1 D1': subject_data[1], 'M1 D2': subject_data[2],
                       'M1 D3': subject_data[3], 'M1 D4': subject_data[4], 'M1 D5': subject_data[5],
                       'M1 D6': subject_data[6]}  # create new row
        seq_df = seq_df._append(subject_row1, ignore_index = True)  # append row to df
    return seq_df

def get_eeg_df():
    columns = ['Subject', 'EF', 'M1 D1', 'M1 D7']
    eeg_df = pd.DataFrame(columns=columns)
    for sub in eeg_dir.glob("P?*"):
        subject_data = []
        for filename in sub.glob('*'):
            decoding_dict = np.load(filename, allow_pickle=True)
            subject_data.append(decoding_dict)  # append eeg data of a single subject to a list
        subject_row2 = {'Subject': sub.name, 'EF': subject_data[0], 'M1 D1': subject_data[1],
                       'M1 D7': subject_data[2]}  # create new row
        eeg_df = eeg_df._append(subject_row2, ignore_index=True)  # append row to df
    return eeg_df

# plotting


# extracting data from dataframe
df = seq_df
results = list()

# Iterate over rows in pandas dataframe
for index, row in df.iterrows():
    df_sub = row
    # Subset original dataframe to one row = data from one subject
    subj = row['Subject']
    # Get value from first day = first column
    d1 = row['EF'][0]
    d2 = row['M1 D1'][0]
    d3 = row['M1 D2'][0]
    d4 = row['M1 D3'][0]
    d5 = row['M1 D4'][0]
    d6 = row['M1 D5'][0]
    d7 = row['M1 D6'][0]
    result = (subj, d1, d2, d3, d4, d5, d6, d7)
    results.append(result)
    # list of tuples to dataframe
    df_ele = pd.DataFrame(results)
    # wide to long format pandas data frame
    dfr = df_ele.reset_index()
    dfr = pd.melt(dfr, id_vars=[0], value_vars=[1, 2, 3, 4, 5, 6])
    dfr.columns = ['Subject', 'day', 'elevation gain']

# Seaborn to plot data
seaborn.set(style='whitegrid')
seaborn.scatterplot(data=dfr, x='day', y='elevation gain', hue='Subject')



# my tryouts
def extract_values(dictionary):
    ele_gain = dictionary[0]
    return ele_gain

seq_df[['elevation_gain1']] = seq_df['EF'].apply(lambda x: pd.Series(extract_values(x)))
seq_df[['elevation_gain2']] = seq_df['M1 D1'].apply(lambda x: pd.Series(extract_values(x)))
print(seq_df)

# taking column and make means of ele gain, then plot the mean ele gain against days
# or plot target ele against perceived ele (aka Pauls script)

plt.plot(elevation_gain1, elevation_gain2)
pd.DataFrame.plot(seq_df['elevation_gain1'])
plt.show()
seq_df.elevation_gain1.shape
seq_df.elevation_gain1[0][4]
plt.plot(seq_df.elevation_gain1)
mean = numpy.mean(seq_df.elevation_gain1, axis = 0)
mean.shape
plt.figure()
plt.plot(x, mean)


my_dict = {'EF': subject_data[0], 'M1 D1': subject_data[1]}

names = list(my_dict.keys())
values = list(my_dict.values())

plt.bar(range(len(my_dict)), values, tick_label=names)
plt.show()

print(seq_df)



# paired t-test
# scipy.stats.ttest_rel(con1, con2, axis=0, nan_policy='propagate', alternative='two-sided', *, keepdims=False)


# 'M1 D7': subject_data[7]
# tryouts:
# dict = [{'subject': '', 'EG EF': '', 'EG M1 D1': [3], 'EG M1 D2': [4], 'EG M1 D3': [5], 'EG M1 D4': [6], 'EG M1 D5': [7], 'EG M1 D6': [8], 'EG M1 D7': [9] }]
# seq_df = pd.DataFrame(dict)
# subject_id = {'subject': ['P1', 'P2', 'P3', 'P4', 'P5']}
# seq_df['subject'] = seq_df['subject'].fill(subject_id)
# seq_df.loc[:]['subject'] = 'P1'
# new_row = {'subject': subject_id}
# subject_id = pd.DataFrame({'subject': ['P1', 'P2', 'P3', 'P4', 'P5']})
# seq_df['subject'] = subject_id
