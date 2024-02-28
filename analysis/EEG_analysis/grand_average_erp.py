import mne
import numpy
import re
from pathlib import Path
from mne import grand_average, read_epochs, write_evokeds
from mne.channels import make_1020_channel_selections

"""
group average epochs
"""

data_dir = Path.cwd() / 'data'
eeg_dir = data_dir / 'experiment' /'pilot' / 'EEG'

conditions = ['Free Ears', 'Molds']
channels = ["FT9", "FT10"]
# data = np.zeros((2, 30, 4, 1251))  # why 1251 with 3 sec epoch duration at 500 Hz?
data = numpy.zeros((2, 4, 4, 1501))

for i_sub, subject_folder in enumerate(eeg_dir.glob('*')):
    for condition in conditions:
        epoch_fpath = list((subject_folder / condition / 'preprocessed').glob('*epo.fif'))[0]
        epochs = mne.read_epochs(fname=epoch_fpath)
        epochs.equalize_event_counts()
        for i_ch, channel in enumerate(channels):
            epochs_ch = epochs.copy()
            epochs_ch.pick_channels([channel])
            for ie, event in enumerate(epochs.event_id.keys()):
                data[i_ch, i_sub, ie] = epochs_ch[event].average().data.flatten()
numpy.save(eeg_dir / 'group_erp.npy', data)

""" inter subject average

Compute the grand average evoked response (i.e the average across subjects).
For both experiments, two files are saved - one with the average across
all conditions and one with the average of each condition.
"""

root = Path(__file__).parent.parent.absolute()

evokeds = []

evoked_conditions = {"37.5": [], "12.5": [], "-12.5": [], "-37.5": []}
for subfolder in (eeg_dir / "preprocessed").glob("sub-*"):
    epochs = read_epochs(subfolder / f"{subfolder.name}-epo.fif")
    if int(re.search(r"\d+", subfolder.name).group()) < 100:
        evokeds1.append(epochs.average())
        for key in evoked1_conditions.keys():
            evoked1_conditions[key].append(epochs[key].average())
    else:
        evokeds2.append(epochs.average())
        for key in evoked2_conditions.keys():
            evoked2_conditions[key].append(epochs[key].average())
evoked1 = grand_average(evokeds1)
evoked2 = grand_average(evokeds2)

# compute the grand average for each condition
for key in evoked1_conditions.keys():
    evoked1_conditions[key] = grand_average(evoked1_conditions[key])
    evoked1_conditions[key].comment = key
evoked1_conditions = list(evoked1_conditions.values())
for key in evoked2_conditions.keys():
    evoked2_conditions[key] = grand_average(evoked2_conditions[key])
    evoked2_conditions[key].comment = key
evoked2_conditions = list(evoked2_conditions.values())

evoked1.save(root / "results" / "grand_averageI-ave.fif", overwrite=True)
evoked2.save(root / "results" / "grand_averageII-ave.fif", overwrite=True)
write_evokeds(
    root / "results" / "grand_averageII_conditions-ave.fif",
    evoked2_conditions,
    overwrite=True,
)
write_evokeds(
    root / "results" / "grand_averageI_conditions-ave.fif",
    evoked1_conditions,
    overwrite=True,
)