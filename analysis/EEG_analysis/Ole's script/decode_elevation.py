from pathlib import Path
import json
import itertools
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne import read_epochs, concatenate_epochs
from mne.decoding import SlidingEstimator, cross_val_multiscore
import pickle

eeg_dir = Path.cwd() / 'data' / 'experiment' / 'EEG'
results_dir = Path.cwd() / 'data' / 'experiment' / 'results' / 'decoding'
p = 100
start_idx = 166
stop_idx = 239
# p = json.load(open(root / "code" / "parameters.json"))
for sub in eeg_dir.glob("P?*"):
    print(sub)
    results = {}
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

        condition_scores = []
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
            scores = cross_val_multiscore(decoder, X, y, cv=p)      #create a parameters list

            condition_score = scores.mean(axis=0)  # mean across 100 folds of cross validation
            condition_scores.append(condition_score)
        # subject_score = subject_scores.mean(axis=1) # mean across conditions
        subject_score = np.mean(condition_scores, axis=0)
        subject_score = subject_score[start_idx:stop_idx].mean() # mean across time

        results[f"{ses.name}"] = subject_score
    # save dict as pickle
    # create a binary pickle file
    with open(results_dir / f"{sub.name}_scores.pkl","wb") as f:
        # write the python object (dict) to pickle file
        pickle.dump(results,f)
        # close file
    f.close()
