from pathlib import Path
import json
import numpy
from scipy import signal
from matplotlib import pyplot as plt
from mne import events_from_annotations, compute_raw_covariance
import mne
from mne.io import read_raw_brainvision
from mne.epochs import Epochs
from autoreject import Ransac, AutoReject
from mne.preprocessing import ICA, read_ica, corrmap
from meegkit.dss import dss_line

conditions = ['Free Ears', 'Molds']

data_dir = Path.cwd() / 'data'
eeg_dir = data_dir / 'experiment' /'pilot' / 'EEG'

electrode_names = json.load(open(data_dir / 'misc' / "electrode_names.json"))
# tmin, tmax and event_ids for both experiments
epoch_parameters = [
    -1.5,  # t_min
    1.5,    # t_max
    {"37.5": 20,  # event ids
    "12.5": 22,
    "-12.5": 24,
    "-37.5": 26}]

# get subject files
for condition in conditions:
    for subfolder in eeg_dir.glob('*'):
        print(subfolder)
        # concatenate raw eeg data across blocks
        if (subfolder / condition).exists():
            outdir = subfolder / condition / 'preprocessed' # create output directory
            if not outdir.exists():
                outdir.mkdir()
            # collect header files
            header_files = list((subfolder / condition).glob('*.vhdr'))
            raws = []
            # concatenate blocks
            for header_file in header_files:
                raws.append(read_raw_brainvision(header_file))
            raw = mne.concatenate_raws(raws, preload=None, events_list=None, on_mismatch='raise', verbose=None)
            raw = raw.load_data()

            # assign channel names to the data
            raw.rename_channels(electrode_names)
            raw.set_montage("standard_1020")  # ------ use brainvision montage instead?
            # montage_path = data_dir / 'misc' / "AS-96_REF.bvef"  # original version
            # montage = mne.channels.read_custom_montage(fname=montage_path)
            # raw.set_montage(montage)

            # inspect raw data
            # raw.plot()
            # fig = raw.plot_psd(xscale='linear', fmin=0, fmax=80)
            # fig.suptitle(header_file.name)
            # raw.compute_psd().plot(average=True)

            print('STEP 1: Remove power line noise and apply minimum-phase highpass filter')  # Cheveigné, 2020
            X = raw.get_data().T
            X, _ = dss_line(X, fline=50, sfreq=raw.info["sfreq"], nremove=30)

            # # plot changes made by the filter:
            # # plot before / after zapline denoising
            # # power line noise is not fully removed with 5 components, remove 10
            # f, ax = plt.subplots(1, 2, sharey=True)
            # f, Pxx = signal.welch(raw.get_data().T, 500, nperseg=500, axis=0, return_onesided=True)
            # ax[0].semilogy(f, Pxx)
            # f, Pxx = signal.welch(X, 500, nperseg=500, axis=0, return_onesided=True)
            # ax[1].semilogy(f, Pxx)
            # ax[0].set_xlabel("frequency [Hz]")
            # ax[1].set_xlabel("frequency [Hz]")
            # ax[0].set_ylabel("PSD [V**2/Hz]")
            # ax[0].set_title("before")
            # ax[1].set_title("after")
            # plt.show()

            # put the data back into raw
            raw._data = X.T
            del X

            # remove line noise (eg. stray electromagnetic signals)
            raw = raw.filter(l_freq=.5, h_freq=None, phase="minimum")

            print('STEP 2: Epoch and downsample the data')
            # get events
            events = events_from_annotations(raw)[0]
            events = events[[not e in [99999, 10, 12, 14, 16] for e in events[:, 2]]]
            # remove all meaningless event codes, including post trial events
            tmin, tmax, event_ids = epoch_parameters  # get epoch parameters
            epochs = Epochs(
                raw,
                events,
                event_id=event_ids,
                tmin=tmin,
                tmax=tmax,
                baseline=None,
                preload=True,
            )

            # extra: use raw data to compute the noise covariance  # for later analysis?
            tmax_noise = (events[0, 0] - 1) / raw.info["sfreq"]  # cut raw data before first stimulus
            raw.crop(0, tmax_noise)
            cov = compute_raw_covariance(raw)  # compute covariance matrix
            cov.save(outdir / f"{subfolder.name}_noise-cov.fif", overwrite=True)  # save to file
            del raw

            fs = 100  # resample data to effectively drop frequency components above fs / 3
            decim = int(epochs.info["sfreq"] / fs)
            epochs.filter(None, fs / 3, n_jobs=4)
            epochs.decimate(decim)

            print('STEP 4: interpolate bad channels and re-reference to average')  # Bigdely-Shamlo et al., 2015
            r = Ransac(n_jobs=4, min_corr=0.85)
            epochs_clean = epochs.copy()
            epochs_clean = r.fit_transform(epochs_clean)
            del r
            # 1 Interpolate all channels from a subset of channels (fraction denoted as min_channels),
            # repeat n_resample times.
            # 2 See if correlation of interpolated channels to original channel is above 75% per epoch (min_corr)
            # 3 If more than unbroken_time fraction of epochs have a lower correlation than that,
            # add channel to self.bad_chs_


            print('STEP 5: Blink rejection with ICA')  # Viola et al., 2009
            # reference = read_ica(data_dir / 'misc' / 'reference-ica.fif')
            # component = reference.labels_["blinks"]
            ica = ICA(n_components=0.999, method="fastica")
            ica.fit(epochs)
            ica.labels_["blinks"] = [0, 1]
            # corrmap([ica, ica], template=(0, component[0]), label="blinks", plot=False, threshold=0.75)
            # ica.plot_components(picks=range(10))
            # ica.plot_sources(epochs)
            ica.apply(epochs, exclude=ica.labels_["blinks"])
            ica.save(outdir / f"{subfolder.name}-ica.fif", overwrite=True)
            del ica

            epochs_clean.set_eeg_reference("average", projection=True)
            epochs.add_proj(epochs_clean.info["projs"][0])
            epochs.apply_proj()
            del epochs_clean

            print('STEP 6: Reject / repair bad epochs')  # Jas et al., 2017
            # ar = AutoReject(n_interpolate=[0, 1, 2, 4, 8, 16], n_jobs=4)
            ar = AutoReject(n_jobs=4)
            epochs = ar.fit_transform(epochs)  # Bigdely-Shamlo et al., 2015)?
            # apply threshold \tau_i to reject trials in the train set
            # calculate the mean of the signal( for each sensor and timepoint) over the GOOD (= not rejected)
            # trials in the train set
            # calculate the median of the signal(for each sensor and timepoint) over ALL trials in the test set
            # compare both of these signals and calculate the error
            # the candidate threshold with the lowest error is the best rejection threshold for a global rejection

            # plot preprocessing results
            fig = epochs.plot_psd(xscale='linear', fmin=0, fmax=50)
            fig.suptitle(header_file.name)

            #  save peprocessed epochs to file
            epochs.save(outdir / f"{subfolder.name}-epo.fif", overwrite=True)
            ar.save(outdir / f"{subfolder.name}-autoreject.h5", overwrite=True)
