import shutil, errno
from mne_bids import (BIDSPath,read_raw_bids)

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc:
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise

def plot_invalid_trials_over_time_stacked(participant_data):
    plt.figure(figsize=(10, 6))

    for idx, (participant, trials) in enumerate(participant_data.items()):
        invalid_trials = [i for i, invalid in enumerate(trials) if invalid == 1]
        plt.scatter(invalid_trials, [idx + 1] * len(invalid_trials), marker='o', label=participant)

    plt.title('Invalid Trials Over Time')
    plt.xlabel('Trial Number')
    plt.ylabel('Participant')
    plt.yticks(list(range(1, len(participant_data) + 1)), list(participant_data.keys()))
    plt.legend()
    plt.grid(True)
    plt.show()

def ptp_annotate(subject_id):
    # annotate points with >40uV difference between successive sample points
    bids_root = "Dataset/ds004147-filtered"    
    bids_path = BIDSPath(subject=subject_id,task="casinos",
                         datatype='eeg', suffix='eeg',
                         root=bids_root)
    raw = read_raw_bids(bids_path)
    raw2 = raw.copy()
    raw2.load_data()
    raw2.filter(l_freq=0.1,h_freq=50)
    # for each channel raw[chan]: set bad annotation at indices where the difference is greater than the 40e-6 threshold
    all_onsets = []
    all_durations = []
    all_descriptions = []
    new_anno = raw2.annotations.copy()
    for chan in raw2.ch_names:
        bad_indices = np.where(abs(np.diff(raw[chan][0][0]))>40e-6)[0]
        n_anno = len(bad_indices)
        times = raw[chan][1][bad_indices]
        new_anno = new_anno + mne.Annotations(
            onset=times,  # in seconds
            duration=[0.001]*n_anno,  # in seconds, too
            description=["BAD_" + chan + ": 40uV between samples"]*n_anno,
            orig_time=raw2.annotations.orig_time
        )
    print(new_anno)
    raw2.set_annotations(new_anno)