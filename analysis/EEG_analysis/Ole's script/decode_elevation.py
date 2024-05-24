from pathlib import Path
import json
import itertools
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne import read_epochs, concatenate_epochs
from mne.decoding import SlidingEstimator, cross_val_multiscore

eeg_dir = Path.cwd() / 'data' / 'experiment' / 'EEG'

# p = json.load(open(root / "code" / "parameters.json"))
for sub in eeg_dir.glob("P?*"):
    print(sub)
    for i_ses, ses in enumerate(sub.glob("ses*")):
        print(ses)
        epochs = [read_epochs(e) for e in ses.glob("*-epo.fif")][0]

        events, event_id = epochs.events, epochs.event_id
        n_times = len(epochs.times)
        combinations = [
            ",".join(map(str, c))
            for c in itertools.combinations(epochs.event_id.keys(), 2)
        ]
        combinations = [comb.split(",") for comb in combinations]
        results = {}
        for i, (con1, con2) in enumerate(combinations):
            mask = np.logical_or(
                events[:, 2] == event_id[con1], events[:, 2] == event_id[con2]
            )
            X = epochs.get_data()[mask]
            y = events[mask][:, 2]
            clf = make_pipeline(
                StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000)
            )
            decoder = SlidingEstimator(clf, scoring="roc_auc", verbose=True, n_jobs=4)
            scores = cross_val_multiscore(decoder, X, y, cv=p["cv_folds"])      #create a parameters list
            results[f"{con1} vs {con2}"] = scores.mean(axis=0)
            np.save(
                eeg_dir
                / "results"
                / "decoding"
                / f"{sub.name}_{ses.name}_decoding_score.npy",
                results,
            )
