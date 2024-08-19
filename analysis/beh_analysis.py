import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import scipy
from scipy import stats
from mne import read_epochs
import pandas as pd
import slab
import pathlib
import pickle
from analysis.plotting.localization_plot import localization_accuracy as la
pd.set_option('display.max_rows', 1000, 'display.max_columns', 200, 'display.width', 99999)


eeg_dir = Path.cwd() / 'data' / 'experiment' / 'results' / 'decoding'
beh_dir = Path.cwd() / 'data' / 'experiment' / 'behavior' / 'localization'
results_dir = Path.cwd() / 'data' / 'experiment' / 'results' / 'decoding' / 'dec'

def get_beh_df():
    columns = ['Subject', 'EF', 'M1 D1', 'M1 D2', 'M1 D3',
               'M1 D4', 'M1 D5', 'M1 D6']
    seq_df = pd.DataFrame(columns=columns)
    for sub in beh_dir.glob("P?*"):
        subject_data = []
        for filename in sub.glob('localization*'):
            trial_seq = slab.Trialsequence(filename) # get trial sequence
            print(filename)
            elevation_gain, ele_rmse, ele_sd, az_rmse, az_sd = la(trial_seq, show=False) # get behavior data
            subject_data.append([elevation_gain, ele_rmse, ele_sd])  # append behavior data of a single subject to a list
        subject_row1 = {'Subject': sub.name, 'EF': subject_data[0], 'M1 D1': subject_data[1], 'M1 D2': subject_data[2],
                       'M1 D3': subject_data[3], 'M1 D4': subject_data[4], 'M1 D5': subject_data[5],
                       'M1 D6': subject_data[6]}  # create new row
        seq_df = seq_df._append(subject_row1, ignore_index=True)  # append row to df
    return seq_df

# plotting perceived and targeted elevation

dfe = pd.DataFrame(columns=[ 'Subject', 'targets', 'responses', 'session'])
for sub in beh_dir.glob("P?*"):
    print(sub)
    # beh_data = []
    for i_ses, ses in enumerate(sub.glob("session3")):
        for filename in ses.glob('localization*'):
            trial_seq = slab.Trialsequence(filename)
            print(trial_seq)

            loc_data = np.asarray(trial_seq.data)
            loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
            targets = loc_data[:, 1]  # [az, ele]
            responses = loc_data[:, 0]
            elevations = np.unique(loc_data[:, 1, 1])
            azimuths = np.unique(loc_data[:, 1, 0])
            targets[:, 1] = loc_data[:, 1, 1]  # target elevations
            responses[:, 1] = loc_data[:, 0, 1]  # perceived elevations
            targets = list(targets[:, 1])
            responses = list(responses[:, 1])
            subject_row2 = {'Subject': sub.name, 'targets': targets, 'responses': responses, "session": ses.name}
            dfe = dfe._append(subject_row2, ignore_index=True)



# plot
# Convert array to list
target_array = dfe['targets']
# Convert NumPy array to list using a loop
list_conv = []
for item in target_array:
    list_conv.append(item)

response_array = dfe['responses']
# Convert NumPy array to list using a loop
list_conv = []
for target, response in target_array, response_array:
    list_conv.append(target, response)

df_ele = pd.DataFrame(list_conv)

list_points = []
for index, row in dfe.iterrows():
    df_e = row
    subj = row['Subject']
    targets = row['targets']
    responses = row['responses']
    session = row['session']
    results = (subj, targets, responses, session)
    list_points.append(results)

# array to list
df_e = dfe.explode('targets', ignore_index=True)
df_e['responses'] = np.tile(responses, len(dfe))
df_e['targets'] = df_e['targets'].astype(float)

# plot
sns.set(style='whitegrid')
sns.scatterplot(data=df_e, x='targets', y='responses', hue='Subject')

# add trend line
sns.relplot(data=df_e, x='targets', y='responses', kind='line')
sns.lmplot(data=df_e, x='targets', y='responses', scatter=None, line_kws={'color': 'orange'})
g = sns.lmplot(x='targets', y='responses', data=df_e, scatter=None)


# plotting learning curve

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
    # calculating mean
    m1 = df_ele[1].mean()
    m2 = df_ele[2].mean()
    m3 = df_ele[3].mean()
    m4 = df_ele[4].mean()
    m5 = df_ele[5].mean()
    m6 = df_ele[6].mean()
    m7 = df_ele[7].mean()
    means = [m1, m2, m3, m4, m5, m6, m7]
    days = [1, 2, 3, 4, 5, 6, 7]
    list_of_tuples = list(zip(means, days))
    dfm = pd.DataFrame(list_of_tuples, columns=['means', 'days'])

    # wide to long format pandas data frame
    dfr = df_ele.reset_index()
    dfr = pd.melt(dfr, id_vars=[0], value_vars=[1, 2, 3, 4, 5, 6, 7])
    dfr.columns = ['Subject', 'day', 'elevation gain']


# Seaborn to plot data
sns.set(style='whitegrid')
# sns.scatterplot(data=dfr, x='day', y='elevation gain', hue='Subject')
sns.lineplot(data=dfr, x='day', y='elevation gain', dashes=False, hue='Subject', style='Subject', markers=["o", "o"])

# add trend line
plt.plot(days, means, '-kx')
plt.show()



# df for eeg data
def get_eeg_df():
    columns_eeg = ['Subject', 'session1', 'session2', 'session3']
    eeg_df = pd.DataFrame(columns=columns_eeg)
    for sub in results_dir.glob("P?*"):
        subject_data = []
        with open(results_dir / f"{sub.name}","rb") as x:
            dict = pickle.load(x)
            dict['Subject'] = sub.name[1]
            subject_data.append(dict)   # append eeg data of a single subject to a list
            new_row = {'Subject': sub.name[1], 'session1': dict['session1'],
                       'session2': dict['session2'], 'session3': dict['session3']} # create new row
        eeg_df = eeg_df._append(new_row, ignore_index=True)  # append row to df
    return eeg_df



# df for three EGs
columns = ['Subject', 'session1', 'session2', 'session3']
    seq_df = pd.DataFrame(columns=columns)
    for sub in beh_dir.glob("P?*"):
        EG_data = []
        for filename in sub.glob('localization*'):
            trial_seq = slab.Trialsequence(filename) # get trial sequence
            print(filename)
            elevation_gain = la(trial_seq, show=False)[0] # get behavior data
            EG_data.append(elevation_gain)  # append behavior data of a single subject to a list
        subject_row1 = {'Subject': sub.name, 'session1': EG_data[0], 'session2': EG_data[1],
                        'session3': EG_data[6]}  # create new row
        seq_df = seq_df._append(subject_row1, ignore_index=True)


# plot da and eg
eg = seq_df['session1']
da = eeg_df['session1']

plt.scatter(eg, da)
plt.xlabel('Elevation gain')
plt.ylabel('Decoding accuracy')

# regression line
scipy.stats.linregress(eg, da)
reg_results = scipy.stats.linregress(eg, da)
slope = reg_results[0]
intercept = reg_results[1]
x = np.linspace(eg.min(), eg.max(), 2)
y = slope * x + intercept
plt.plot(x,y)

