from pathlib import Path
import json
import numpy as np
from mne.io import read_raw_brainvision
from mne.epochs import Epochs
from mne.preprocessing import ICA, read_ica, corrmap
from mne import events_from_annotations, merge_events
from meegkit.dss import dss_line
from meegkit.detrend import detrend
from autoreject import Ransac, AutoReject

root = Path(__file__).parent.parent.absolute()
p = json.load(open(root / "code" / "parameters.json"))
electrode_names = json.load(open(root / "code" / "electrode_names.json"))
reference_ica = read_ica(root / "code" / "reference-ica.fif")


for sub in (root / "data" / "bids").glob("sub*"):
    for ses in sub.glob("ses*"):
        for run in (ses / "eeg").glob("*.vhdr"):
            run_id = run.name.split("_")[-2]
            raw = read_raw_brainvision(run, preload=True)
            raw.rename_channels(electrode_names)
            raw.set_montage("standard_1020")

            # Remove power line noise
            X = raw._data.T
            X, _ = dss_line(
                X,
                fline=p["zapline"]["fline"],
                sfreq=raw.info["sfreq"],
                nremove=p["zapline"]["nremove"],
            )
            X, _, _ = detrend(X, 1)
            X, _, _ = detrend(X, 8)
            raw._data = X.T

            # remove target and post-target trials
            events, event_id = events_from_annotations(raw)
            idx = np.where(
                np.isin(
                    events[:, 2],
                    [
                        p["epochs"]["event_id"]["+37.5_t"],
                        p["epochs"]["event_id"]["+12.5_t"],
                        p["epochs"]["event_id"]["-12.5_t"],
                        p["epochs"]["event_id"]["-37.5_t"],
                    ],
                )
            )[0]
            idx = np.concatenate([idx, idx + 2])
            events = np.delete(events, idx, axis=0)

            # create epochs
            epochs = Epochs(
                raw,
                events,
                event_id=dict(
                    (k, p["epochs"]["event_id"][k])
                    for k in ["+37.5", "+12.5", "-12.5", "-37.5"]
                ),
                tmin=p["epochs"]["tmin"],
                tmax=p["epochs"]["tmax"],
                preload=True,
            )
            del raw

            # downsample
            decim = int(epochs.info["sfreq"] / p["fs"])
            epochs.filter(None, p["fs"] / 3, n_jobs=4)
            epochs.decimate(decim)

            # remove blinks
            ica = ICA(n_components=0.999, method="fastica")
            ica.fit(epochs.copy().filter(l_freq=2, h_freq=None, n_jobs=4))
            for key, value in reference_ica.labels_.items():
                corrmap(
                    [reference_ica, ica],
                    (0, value[0]),
                    label=key,
                    threshold=p["corrmap_thresh"],
                    plot=False,
                )
            if len(ica.labels_["blinks"]) > 0:
                bad_ics = [v[0] for v in ica.labels_.values()]
                epochs = ica.apply(epochs, exclude=bad_ics)

            # re-reference to robust average
            r = Ransac(n_jobs=4, min_corr=p["ransac_thresh"])
            epochs_clean = epochs.copy()
            epochs_clean = r.fit_transform(epochs_clean)
            epochs_clean.set_eeg_reference("average", projection=True)
            epochs.add_proj(epochs_clean.info["projs"][0])
            epochs.apply_proj()
            del epochs_clean

            # remove and repair epochs
            ar = AutoReject()
            epochs, log = ar.fit_transform(epochs, return_log=True)

            # save result
            outdir = root / "data" / "epochs" / sub.name / ses.name
            if not outdir.exists():
                outdir.mkdir(parents=True)
            epochs.save(
                outdir / f"{sub.name}_{ses.name}_{run_id}-epo.fif", overwrite=True
            )
