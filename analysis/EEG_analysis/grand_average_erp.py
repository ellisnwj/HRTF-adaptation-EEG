import mne
import numpy
import re
from pathlib import Path
from mne import grand_average, read_epochs, write_evokeds
from mne.channels import make_1020_channel_selections
from copy import deepcopy

data_dir = Path.cwd() / 'data'
eeg_dir = data_dir / 'experiment' / 'EEG'


"""
Compute the grand average evoked response (i.e the average across subjects).
For both experiments, two files are saved - one with the average across
all conditions and one with the average of each condition.
"""

evoked = []
conditions = ['Ears Free', 'Molds 1', 'Molds 2']
elevation_conditions = {"37.5": [], "12.5": [], "-12.5": [], "-37.5": []}
evoked_conditions = {'Ears Free': deepcopy(elevation_conditions), 'Molds 1': deepcopy(elevation_conditions),
                     'Molds 2': deepcopy(elevation_conditions),}

for condition in conditions:  # iterate over ear conditions
    for i_sub, subject_folder in enumerate(eeg_dir.glob('*P1')):
        epoch_fpath = list((subject_folder / condition / 'preprocessed').glob('*epo.fif'))
        if epoch_fpath:
            epochs = mne.read_epochs(fname=epoch_fpath[0])  # take the first element from the list
            evoked.append(epochs.average())  # get evoked response across all conditions
            for key in elevation_conditions.keys():  # evoked response for each elevation condition
                evoked_conditions[condition][key].append(epochs[key].average())

# compute the grand average
# across conditions
# evoked = grand_average(evoked)
# for each condition
for condition in conditions:
    for key in elevation_conditions.keys():
        evoked_conditions[condition][key] = grand_average(evoked_conditions[condition][key])
        evoked_conditions[condition][key].comment = key

evoked_conditions = list(evoked_conditions.values())

# write data
# evoked.save(eeg_dir / 'grand_average-ave.fif', overwrite=True)
write_evokeds(eeg_dir / 'Ears_Free_grand_average_conditions-ave.fif', list(evoked_conditions[0].values()), overwrite=True)
write_evokeds(eeg_dir / 'Molds_grand_average_conditions-ave.fif', list(evoked_conditions[1].values()), overwrite=True)
write_evokeds(eeg_dir / 'Molds_2_grand_average_conditions-ave.fif', list(evoked_conditions[2].values()), overwrite=True)


# group average epochs
# """
#
# channels = ["FT9", "FT10"]
# # data = np.zeros((2, 30, 4, 1251))  # why 1251 with 3 sec epoch duration at 500 Hz?
# data = numpy.zeros((2, 4, 4, 1501))
#
# for i_sub, subject_folder in enumerate(eeg_dir.glob('*')):
#     for condition in conditions:
#         epoch_fpath = list((subject_folder / condition / 'preprocessed').glob('*epo.fif'))[0]
#         epochs = mne.read_epochs(fname=epoch_fpath)
#         epochs.equalize_event_counts()
#         for i_ch, channel in enumerate(channels):
#             epochs_ch = epochs.copy()
#             epochs_ch.pick_channels([channel])
#             for ie, event in enumerate(epochs.event_id.keys()):
#                 data[i_ch, i_sub, ie] = epochs_ch[event].average().data.flatten()
# numpy.save(eeg_dir / 'group_erp.npy', data)
