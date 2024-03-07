import mne
import numpy
from pathlib import Path
from matplotlib import pyplot as plt
samplerate = 500

data_dir = Path.cwd() / 'data'
eeg_dir = data_dir / 'experiment' /'pilot' / 'EEG'

free_ears_fname = eeg_dir / 'Free_Ears_grand_average_conditions-ave.fif'
molds_fname = eeg_dir / 'Molds_grand_average_conditions-ave.fif'

picks = ['FT10']
# picks = ['Fz']


# get data
evoked_free_ears = mne.read_evokeds(free_ears_fname)
free_ears_data = dict()
for evoked in evoked_free_ears:
    free_ears_data[evoked.comment] = evoked.pick_channels(picks).data[0]

evoked_molds = mne.read_evokeds(molds_fname)
molds_data = dict()
for evoked in evoked_molds:
    molds_data[evoked.comment] = evoked.pick_channels(picks).data[0]

# plot
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
times = evoked.times
tmin = -1.1
tmax = 1.0

# free ears
fig, axis = plt.subplots(1,1, figsize=(13,4))
for i, key in enumerate(free_ears_data.keys()):
    axis.plot(times, free_ears_data[key],c=colors[i], label=key + '°')
ylim = axis.get_ylim()
axis.vlines(x=[-1, 0], ymin=ylim[0], ymax=ylim[1], color='black', linestyles='dashed', linewidth=.5)
axis.set_ylim(ylim[0], ylim[1])
axis.set_xlim(tmin, tmax)
axis.set_xlabel('Time (ms)')
axis.set_ylabel('Voltage (mV)')
axis.legend(title='Elevation')
axis.set_title(f'condition: free ears, channel: {picks[0]}')

# molds
fig, axis = plt.subplots(1,1, figsize=(13,4))
for i, key in enumerate(molds_data.keys()):
    axis.plot(times, molds_data[key],c=colors[i], label=key + '°')
ylim = axis.get_ylim()
axis.vlines(x=[-1, 0], ymin=ylim[0], ymax=ylim[1], color='black', linestyles='dashed', linewidth=.5)
axis.set_ylim(ylim[0], ylim[1])
axis.set_xlim(tmin, tmax)
axis.set_xlabel('Time (ms)')
axis.set_ylabel('Voltage (mV)')
axis.legend(title='Elevation')
axis.set_title(f'condition: earmolds, channel: {picks[0]}')
