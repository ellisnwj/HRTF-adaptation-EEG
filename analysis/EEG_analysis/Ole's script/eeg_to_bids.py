"""
Convert EEG data to the bids format.
For each participant there are three folders:
    Ears Free: initial recording before molds were inserted. In the BIDS format, this task is called "free_ears".
    Molds 1: recording after insertion of molds. In the BIDS format, this task is called "molds_naive"
    Molds 2: recoring after 5 days of adaptation. In the BIDS format, this task is called "molds_trained".
Each recording consists of 5 runs.

There are 9 different event codes:
    1: onset of the adapter
    20: Onset of the probe at 37.5 degree
    22: Onset of the probe at 12.5 degree
    24: Onset of the probe at -12.5 degree
    26: Onset of the probe at -37.5 degree
    10, 12, 14, 16: Onset of probe at the same location as codes 20-26 but followed by a tone which identified this trial as a probe
"""

from pathlib import Path
from mne import events_from_annotations, annotations_from_events, concatenate_raws
from mne.io import read_raw_brainvision
from mne_bids import BIDSPath, write_raw_bids

root = Path(__file__).parent.parent.absolute()
bids_root = Path("/home/obi/projects/elevation-bids")

sub_id_offset = 129
event_desc = {
    20: "+37.5",
    22: "+12.5",
    24: "-12.5",
    26: "-37.5",
    10: "+37.5_t",
    12: "+12.5_t",
    14: "-12.5_t",
    16: "-37.5_t",
    1: "adapter",
}

for sub in (root / "data" / "original" / "EEG").glob("P*"):
    sub_id = str(int(sub.name[-1]) + sub_id_offset)
    for session, folder_name in enumerate(["Ears Free", "Molds 1", "Molds 2"]):
        raw = []
        for run in (sub / folder_name).glob("*.vhdr"):
            run_id = str(int(run.name.split(".")[0][-1]) + 1)
            raw.append(read_raw_brainvision(run))
        raw = concatenate_raws(raw)
        events, event_id = events_from_annotations(raw)
        events = events[events[:, 2] != 99999]

        annotations = annotations_from_events(events, raw.info["sfreq"], event_desc)
        raw = raw.set_annotations(annotations)
        event_id = {v: k for k, v in event_desc.items()}
        events, event_id = events_from_annotations(raw, event_id=event_id)

        bids_path = BIDSPath(
            subject=sub_id,
            session=str(session + 1),
            task="oneback",
            root=bids_root,
        )

        write_raw_bids(
            raw,
            bids_path,
            events,
            event_id,
            overwrite=True,
            allow_preload=True,
            format="BrainVision",
        )
