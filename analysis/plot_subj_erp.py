import mne
from pathlib import Path
from copy import deepcopy
from matplotlib import pyplot as plt

data_dir = Path.cwd() / 'data'
eeg_dir = data_dir / 'experiment' / 'EEG'
subject_paths = sorted(eeg_dir.glob('*P?'), key=lambda item: item.name)

picks = ['FT10']
# picks = ['Fz']


evoked = []
conditions = ['Ears Free', 'Molds 1', 'Molds 2']
elevation_conditions = {"37.5": [], "12.5": [], "-12.5": [], "-37.5": []}
evoked_conditions = {'Ears Free': deepcopy(elevation_conditions), 'Molds 1': deepcopy(elevation_conditions),
                     'Molds 2': deepcopy(elevation_conditions),}

for condition in conditions:  # iterate over ear conditions
    for i_sub, subject_folder in enumerate(subject_paths):
        epoch_fpath = list((subject_folder / condition / 'preprocessed').glob('*epo.fif'))
        if epoch_fpath:
            epochs = mne.read_epochs(fname=epoch_fpath[0])  # take the first element from the list
            for key in elevation_conditions.keys():  # evoked response for each elevation condition
                data = epochs[key].average().pick_channels(picks).data[0] * 1e6  # convert to mV
                evoked_conditions[condition][key].append(data)

# plot
n_subj = len(list(eeg_dir.glob('*P?')))
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
times = epochs.times
tmin = -1.1
tmax = 1.0

fig, axis = plt.subplots(3,n_subj, figsize=(21,8), tight_layout=True, sharex=True)

# plot
for condition_idx, condition in enumerate(conditions):
    axis[condition_idx, 0].set_ylabel(f'{condition} \n Voltage (mV)')
    for subj_idx in range(n_subj):
        axis[0, subj_idx].set_title(subject_paths[subj_idx].name)
        axis[2, subj_idx].set_xlabel('Time (ms)')
        ax = axis[condition_idx, subj_idx]
        for i, key in enumerate(evoked_conditions['Ears Free'].keys()):
            ax.plot(times, evoked_conditions[condition][key][subj_idx], c=colors[i], label=key + '°')
        ylim = ax.get_ylim()
        ax.vlines(x=[-1, 0], ymin=ylim[0], ymax=ylim[1], color='black', linestyles='dashed', linewidth=.5)
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlim(tmin, tmax)

ax.legend(title='Elevation', loc='lower left')
fig.text(.5, .98, f'channel: {picks[0]}')