# Default settings for data processing and analysis.

from typing import Optional, Union, Iterable, List, Tuple, Dict, Callable, Literal

from mne import Covariance
from mne_bids import BIDSPath

from mne_bids_pipeline.typing import (
    PathLike,
    ArbitraryContrast,
    FloatArrayLike,
    DigMontageType,
)

#######################################
# Custom Flags
#--------------------------------------

# True = change events/epochs to ICA training
ica_train_step: bool = False # False by default, to be set as true just while training ICA
print('ica train flag: ', ica_train_step)

# Renaming Stimuli toggle
# default False = no renaming
renaming_flag: bool = True
print('renaming flag: ', renaming_flag)

#--------------------------------------
#######################################

#######################################
# Config parameters
study_name: str = "EEG2324Brownie"

bids_root = "./Dataset/ds004147-filtered"

interactive = False

sessions: Union[List, Literal["all"]] = "all"

task: str = "casinos"

plot_psd_for_runs: Union[Literal["all"], Iterable[str]] = "all"

# no subjects excluded - our analysis did not return any subjects that satisfied the exclusion criteria
subjects = 'all'

ch_types = ['eeg']

data_type = 'eeg'

eog_channels: Optional[Iterable[str]] = ['Fp1', 'Fp2']

# we use Fp1 and Fp2 as virtual EOG channels, as explained below:

"""
Specify EOG channels to use, or create virtual EOG channels.

Allows the specification of custom channel names that shall be used as
(virtual) EOG channels. For example, say you recorded EEG **without** dedicated
EOG electrodes, but with some EEG electrodes placed close to the eyes, e.g.
Fp1 and Fp2. These channels can be expected to have captured large quantities
of ocular activity, and you might want to use them as "virtual" EOG channels,
while also including them in the EEG analysis. By default, MNE won't know that
these channels are suitable for recovering EOG, and hence won't be able to
perform tasks like automated blink removal, unless a "true" EOG sensor is
present in the data as well. Specifying channel names here allows MNE to find
the respective EOG signals based on these channels.
"""

eeg_reference = ['TP9', 'TP10'] #rereference to the average of the mastoid signals

eeg_template_montage = 'standard_1020'

analyze_channels: Union[
    Literal["all"], Literal["ch_types"], Iterable["str"]
] = ['all']

# FREQUENCY FILTERING & RESAMPLING
l_freq = 0.1

h_freq = 50.0

notch_freq = 50.0

raw_resample_sfreq = None

# RENAME EXPERIMENTAL EVENTS
if renaming_flag:
    rename_events: dict = {
        'Stimulus/S  6': 'Win LL', 
        'Stimulus/S  7': 'Loss LL',
        'Stimulus/S 16': 'Win ML', 
        'Stimulus/S 17': 'Loss ML',
        'Stimulus/S 26': 'Win MH', 
        'Stimulus/S 27': 'Loss MH',
        'Stimulus/S 36': 'Win HH', 
        'Stimulus/S 37': 'Loss HH',
        'Stimulus/S  2':'Cue LL',
        'Stimulus/S 12':'Cue ML',
        'Stimulus/S 22':'Cue MH',
        'Stimulus/S 32':'Cue HH',
        # not directly required for analysis:
        'Stimulus/S  1': 'Fixation LL',
        'Stimulus/S 11': 'Fixation ML',
        'Stimulus/S 21': 'Fixation MH',
        'Stimulus/S 31': 'Fixation HH',
        'Stimulus/S  3':'Beep LL',        
        'Stimulus/S 13':'Beep ML',        
        'Stimulus/S 23':'Beep MH',        
        'Stimulus/S 33':'Beep HH',
        'Stimulus/S  4': 'Valid Left LL',
        'Stimulus/S 14': 'Valid Left ML',
        'Stimulus/S 24': 'Valid Left MH',
        'Stimulus/S 34': 'Valid Left HH',
        'Stimulus/S  5':'Valid Right LL',        
        'Stimulus/S 15':'Valid Right ML',        
        'Stimulus/S 25':'Valid Right MH',        
        'Stimulus/S 35':'Valid Right HH',
    }

conditions = [
    'Stimulus/S  6',
    'Stimulus/S  7',
    'Stimulus/S 16',
    'Stimulus/S 17',
    'Stimulus/S 26',
    'Stimulus/S 27',
    'Stimulus/S 36',
    'Stimulus/S 37'
]

if ica_train_step:
    conditions = [
        'Stimulus/S  2',
        'Stimulus/S 12',
        'Stimulus/S 22',
        'Stimulus/S 32'
        ]
    
if renaming_flag:
    conditions = [
    'Win LL',
    'Loss LL',
    'Win ML',
    'Loss ML',
    'Win MH',
    'Loss MH',
    'Win HH',
    'Loss HH'
    ]

    if ica_train_step:
        conditions = [
            'Cue LL',
            'Cue ML',
            'Cue MH',
            'Cue HH'
            ]

# epochs relative to the stimulus events
epochs_tmin: float = -0.2
if ica_train_step:
    epochs_tmin = 0

epochs_tmax: float = 0.6
if ica_train_step:
    epochs_tmax = 3

task_is_rest: bool = False

# this is the default parameter but the pipeline failed without it commented in:
baseline: Optional[Tuple[Optional[float], Optional[float]]] = (None, 0)

contrasts = [('Stimulus/S  6','Stimulus/S  7'),('Stimulus/S 16','Stimulus/S 17'),('Stimulus/S 26','Stimulus/S 27'),('Stimulus/S 36','Stimulus/S 37')]

if renaming_flag:
    contrasts = [('Win LL','Loss LL'),('Win ML','Loss ML'),('Win MH','Loss MH'),('Win HH','Loss HH')]

# ARTIFACT REMOVAL
spatial_filter: Optional[Literal["ssp", "ica"]] = 'ica'

# Rejection based on ICA
ica_reject: Optional[Union[Dict[str, float], Literal["autoreject_local"]]] = 'autoreject_local'

ica_l_freq: Optional[float] = 1.0

ica_max_iterations: int = 500

ica_n_components: Optional[Union[float, int]] = 0.95

ica_eog_threshold: float = 1.0

# Rejection based on peak-to-peak amplitude
reject = {"eeg":150e-6} #'autoreject_global' #   

reject_tmin: Optional[float] = None

reject_tmax: Optional[float] = None

# DECODING
decode: bool = False

# Execution
n_jobs: int = 3