from pathlib import Path
import json
import pandas as pd
import numpy as np

root = Path(__file__).parent.parent.absolute()

sub_id_offset = 129
task = "oneback"
top_dir = root / "data" / "original" / "behavior" / "EEG"
columns = ["trial", "target_ele", "response_ele"]
bids_root = Path("/home/obi/projects/elevation-bids")

beh_dict = {
    "trial": {"Description": "Trial number"},
    "target_ele": {"Description": "Target elevation in degree"},
    "response_ele": {"Description": "Response elevation in degree"},
}

# get behavior for the task during EEG recording
for sub in top_dir.glob("*"):
    sub_id = str(int(sub.name[-1]) + sub_id_offset)
    for i_c, cond in enumerate(["Ears Free", "Molds 1"]):
        folder = sub / cond
        runs = list(folder.glob("*"))
        runs.sort()
        dfs = []
        for i_r, run in enumerate(runs):
            seq = np.load(run, allow_pickle=True)["data"]
            if any([len(x) == 0 for x in seq]):
                print(f"empty responses in {run}")
            is_target = [x[0][0][0] is not None for x in seq if len(x) > 0]
            df = pd.DataFrame(
                0, index=np.arange(sum(is_target)), columns=columns, dtype=float
            )
            count = 0
            for i, (s, it) in enumerate(zip(seq, is_target)):
                if it:
                    df.iloc[count].trial = i + 1
                    df.iloc[count].target_ele = s[0][1][1]
                    df.iloc[count].response_ele = s[0][0][1]
                    count += 1
            df.trial = df.trial.astype(int)
            df.response_ele = df.response_ele.round(1)
            dfs.append(df)
        df = pd.concat(dfs)

        out = (
            bids_root
            / f"sub-{sub_id}"
            / f"ses-{i_c+1}"
            / "beh"
            / f"sub-{sub_id}_ses-{i_c+1}_task-oneback_beh.tsv"
        )

        if not out.parent.exists():
            out.parent.mkdir(parents=True)

        df.to_csv(out, index=False, sep="\t")

        # Save task description dictionary
        out = (
            bids_root
            / f"sub-{sub_id}"
            / f"ses-{i_c+1}"
            / "beh"
            / f"sub-{sub_id}_ses-{i_c+1}_task-oneback_beh.json"
        )
        json.dump(beh_dict, open(out, "w"))

# get behavior for the initial localization task
task = "loctest"
columns = ["target_ele", "target_azi", "response_ele", "response_azi"]
beh_dict = {
    "target_ele": {"Description": "Target elevation in degree"},
    "response_ele": {"Description": "Response elevation in degree"},
    "target_azi": {"Description": "Target azimuth in degree"},
    "response_azi": {"Description": "Response azimuth in degree"},
}
top_dir = root / "data" / "original" / "behavior" / "localization"
for sub in top_dir.glob("*"):
    sub_id = str(int(sub.name[-1]) + sub_id_offset)
    print(sub_id)
    for i_c, cond in enumerate(["Ears Free", "Molds"]):
        folder = sub / cond
        runs = list(folder.glob("*localization*"))
        runs.sort()
        run = runs[0]
        # check if there are multiple recordings from the same day
        # if so, always use the last one
        original_name = run.name
        done = 0
        count = 1
        while not done:
            next_run = run.parent / f"{original_name}_{count}"
            if next_run.exists():
                run = next_run
                count += 1
            else:
                done = 1
        seq = np.load(run, allow_pickle=True)["data"]
        df = pd.DataFrame(0, index=np.arange(len(seq)), columns=columns, dtype=float)
        df.target_azi = [s[0][1][0] for s in seq]
        df.target_ele = [s[0][1][1] for s in seq]
        df.response_azi = [s[0][0][0] for s in seq]
        df.response_ele = [s[0][0][1] for s in seq]
        df.response_ele = df.response_ele.round(1)
        df.response_azi = df.response_azi.round(1)

        out = (
            bids_root
            / f"sub-{sub_id}"
            / f"ses-{i_c+1}"
            / "beh"
            / f"sub-{sub_id}_ses-{i_c+1}_task-loctest_beh.tsv"
        )

        if not out.parent.exists():
            out.parent.mkdir(parents=True)
        df.to_csv(out, index=False, sep="\t")

        # Save task description dictionary
        out = (
            bids_root
            / f"sub-{sub_id}"
            / f"ses-{i_c+1}"
            / "beh"
            / f"sub-{sub_id}_ses-{i_c+1}_task-loctest_beh.json"
        )
        json.dump(beh_dict, open(out, "w"))
