from pathlib import Path
import json
import numpy as np
from mne import events_from_annotations, compute_raw_covariance
from mne.io import read_raw_brainvision
from mne.epochs import Epochs
from autoreject import Ransac, AutoReject
from mne.preprocessing import ICA, read_ica, corrmap
from meegkit.dss import dss_line

data_dir = Path.cwd() / 'data'
eeg_dir = data_dir / 'experiment' /'pilot' / 'EEG'
condition = 'Free Ears'

electrode_names = json.load(open(data_dir / 'misc' / "electrode_names.json"))
# tmin, tmax and event_ids for both experiments
epoch_parameters = [
    -0.5,  # t_min
    2.5,    # t_max
    {"37.5": 20,  # event ids
    "12.5": 22,
    "-12.5": 24,
    "-37.5": 26}]

for subfolder in eeg_dir.glob('*'):
    print(subfolder)
header_file = subfolder / condition / 'Zofia Pilot_07.02.24_1.vhdr'
raw = read_raw_brainvision(header_file)  # read raw EEG from file
# todo make sure the header file contains the correct file names

tmin, tmax, event_ids = epoch_parameters  # get epoch parameters

outdir = subfolder / condition / 'preprocessed' # create output directory
if not outdir.exists():
    outdir.mkdir()

raw.load_data()
raw.set_montage("standard_1020")
events = events_from_annotations(raw)[0]

# remove all meaningless event codes
events = events[[not e in [99999] for e in events[:, 2]]]

# STEP 1: Remove power line noise and apply minimum-phase highpass filter
X = raw.get_data().T
X, _ = dss_line(X, fline=50, sfreq=raw.info["sfreq"], nremove=5)
raw._data = X.T  # put the data back into raw
del X
raw = raw.filter(l_freq=1, h_freq=None, phase="minimum")

# STEP 2: Epoch and downsample the data
epochs = Epochs(
    raw,
    events,
    event_id=event_ids,
    tmin=tmin,
    tmax=tmax,
    baseline=None,
    preload=True,
)

# use raw data to compute the noise covariance
tmax_noise = (events[0, 0] - 1) / raw.info["sfreq"]
raw.crop(0, tmax_noise)
cov = compute_raw_covariance(raw)
cov.save(outdir / f"{subfolder.name}_noise-cov.fif", overwrite=True)
del raw

# STEP 3: Remove target and post-target trials
idx = (
    np.genfromtxt(
        subfolder / "beh" / f"{subfolder.name}_task-oneback_beh.tsv",
        delimiter="\t",
        usecols=0,
        skip_header=1,
        dtype=int,
    )
    + 1
)
if idx[-1] == len(epochs):  # if the last trial was a target remove it
    idx = idx[:-1]
epochs.drop(idx)

# STEP 4: interpolate bad channels and re-reference to average
r = Ransac(n_jobs=4)
epochs = r.fit_transform(epochs)
epochs.set_eeg_reference("average")
del r

# STEP 5: Blink rejection with ICA
reference = read_ica(root / "code" / "reference-ica.fif")
component = reference.labels_["blinks"]
ica = ICA(n_components=0.999, method="fastica")
ica.fit(epochs)
ica.labels_["blinks"] = []
corrmap(
    [reference, ica],
    template=(0, component[0]),
    label="blinks",
    plot=False,
    threshold=0.75,
)
ica.apply(epochs, exclude=ica.labels_["blinks"])
ica.save(outdir / f"{subfolder.name}-ica.fif", overwrite=True)
del ica

# STEP 6: Reject / repair bad epochs
ar = AutoReject(n_interpolate=[0, 1, 2, 4, 8, 16], n_jobs=4)
epochs = ar.fit_transform(epochs)
epochs.save(outdir / f"{subfolder.name}-epo.fif", overwrite=True)
ar.save(outdir / f"{subfolder.name}-autoreject.h5", overwrite=True)