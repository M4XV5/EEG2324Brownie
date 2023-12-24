study_name: str = "EEG2324Brownie"
bids_root = "./Dataset/ds004147-filtered"
deriv_root = "./Dataset/derivatives/mne-bids-pipeline/ds004147-filtered"

task = "casinos"
interactive = False
subjects = ['27']  # analysing only subject 27 for the initial test run
ch_types = ['eeg']
data_type = 'eeg'
raw_resample_sfreq = 150
conditions = ["S  2"]
epochs_tmin = -0.2
epochs_tmax = 0.6
task_is_rest = False  # assuming resting state for initial testing
baseline = (None, 0)
