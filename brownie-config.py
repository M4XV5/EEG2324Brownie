from typing import Optional, Union, Iterable, List, Tuple, Dict, Callable, Literal

from mne import Covariance
from mne_bids import BIDSPath

from mne_bids_pipeline.typing import (
    PathLike,
    ArbitraryContrast,
    FloatArrayLike,
    DigMontageType,
)

study_name: str = "EEG2324Brownie"
bids_root = "./Dataset/ds004147"
interactive = False
task: str = "casinos"

subjects = ['all']
# our analysis did not return any subjects that satisfied the exclusion criteria
# exclude_subjects: Iterable[str] = [] 

ch_types = ['eeg']
data_type = 'eeg'

eeg_reference = ['TP9', 'TP10'] #rereference to the average of the mastoid signals

eeg_template_montage = 'standard_1020'

# drop_channels: Iterable[str] = []

l_freq = 0.1
h_freq = 50.0
notch_freq = 50.0

raw_resample_sfreq = 150

###############################################################################
# EPOCHING
# --------

# epochs_metadata_tmin: Optional[float] = None
# """
# The beginning of the time window for metadata generation, in seconds,
# relative to the time-locked event of the respective epoch. This may be less
# than or larger than the epoch's first time point. If `None`, use the first
# time point of the epoch.
# """

# epochs_metadata_tmax: Optional[float] = None
# """
# Same as `epochs_metadata_tmin`, but specifying the **end** of the time
# window for metadata generation.
# """

# epochs_metadata_keep_first: Optional[Iterable[str]] = None
# """
# Event groupings using hierarchical event descriptors (HEDs) for which to store
# the time of the **first** occurrence of any event of this group in a new column
# with the group name, and the **type** of that event in a column named after the
# group, but with a `first_` prefix. If `None` (default), no event
# aggregation will take place and no new columns will be created.

# ???+ example "Example"
#     Assume you have two response events types, `response/left` and
#     `response/right`; in some trials, both responses occur, because the
#     participant pressed both buttons. Now, you want to keep the first response
#     only. To achieve this, set
#     ```python
#     epochs_metadata_keep_first = ['response']
#     ```
#     This will add two new columns to the metadata: `response`, indicating
#     the **time** relative to the time-locked event; and `first_response`,
#     depicting the **type** of event (`'left'` or `'right'`).

#     You may also specify a grouping for multiple event types:
#     ```python
#     epochs_metadata_keep_first = ['response', 'stimulus']
#     ```
#     This will add the columns `response`, `first_response`, `stimulus`,
#     and `first_stimulus`.
# """

# epochs_metadata_keep_last: Optional[Iterable[str]] = None
# """
# Same as `epochs_metadata_keep_first`, but for keeping the **last**
# occurrence of matching event types. The columns indicating the event types
# will be named with a `last_` instead of a `first_` prefix.
# """

# epochs_metadata_query: Optional[str] = None
# """
# A [metadata query][https://mne.tools/stable/auto_tutorials/epochs/30_epochs_metadata.html]
# specifying which epochs to keep. If the query fails because it refers to an
# unknown metadata column, a warning will be emitted and all epochs will be kept.

# ???+ example "Example"
#     Only keep epochs without a `response_missing` event:
#     ```python
#     epochs_metadata_query = ['response_missing.isna()']
#     ```
# """  # noqa: E501

conditions = ['S  2', 'S  6', 'S  7',
'S 12', 'S 16', 'S 17',
'S 22', 'S 26', 'S 27',
'S 32', 'S 36', 'S 37'
]

# epochs relative to the stimulus (/ # beep) events
epochs_tmin: float = -0.2 # -1.2
epochs_tmax: float = 0.6 # -0.6

task_is_rest: bool = False

# baseline = (-0.2, 0)
# """
# Specifies which time interval to use for baseline correction of epochs;
# if `None`, no baseline correction is applied.

# ???+ example "Example"
#     ```python
#     baseline = (None, 0)  # beginning of epoch until time point zero
#     ```
# """

contrasts = [('S  6','S  7'),('S 16','S 17'),('S 26','S 27'),('S 36','S 37')]
# contrasts = [
#         {
#             'name': 'winVsLossContrast',
#             'conditions': [
#                 'S  6','S 16','S 26','S 36','S  7','S 17','S 27','S 37'
#             ],
#             'weights': [1, 1, 1, 1, -1, -1, -1, -1]
#         }
#     ]
# """
# The conditions to contrast via a subtraction of ERPs / ERFs. The list elements
# can either be tuples or dictionaries (or a mix of both). Each element in the
# list corresponds to a single contrast.

# A tuple specifies a one-vs-one contrast, where the second condition is
# subtracted from the first.

# If a dictionary, must contain the following keys:

# - `name`: a custom name of the contrast
# - `conditions`: the conditions to contrast
# - `weights`: the weights associated with each condition.

# Pass an empty list to avoid calculation of any contrasts.

# For the contrasts to be computed, the appropriate conditions must have been
# epoched, and therefore the conditions should either match or be subsets of
# `conditions` above.

# ???+ example "Example"
#     Contrast the "left" and the "right" conditions by calculating
#     `left - right` at every time point of the evoked responses:
#     ```python
#     contrasts = [('left', 'right')]  # Note we pass a tuple inside the list!
#     ```

#     Contrast the "left" and the "right" conditions within the "auditory" and
#     the "visual" modality, and "auditory" vs "visual" regardless of side:
#     ```python
#     contrasts = [('auditory/left', 'auditory/right'),
#                  ('visual/left', 'visual/right'),
#                  ('auditory', 'visual')]
#     ```

#     Contrast the "left" and the "right" regardless of side, and compute an
#     arbitrary contrast with a gradient of weights:
#     ```python
#     contrasts = [
#         ('auditory/left', 'auditory/right'),
#         {
#             'name': 'gradedContrast',
#             'conditions': [
#                 'auditory/left',
#                 'auditory/right',
#                 'visual/left',
#                 'visual/right'
#             ],
#             'weights': [-1.5, -.5, .5, 1.5]
#         }
#     ]
#     ```
# """

reject = {"eeg":150e-6}
# """
# Peak-to-peak amplitude limits to mark epochs as bad. This allows you to remove
# epochs with strong transient artifacts.

# !!! info
#       The rejection is performed **after** SSP or ICA, if any of those methods
#       is used. To reject epochs **before** fitting ICA, see the
#       [`ica_reject`][mne_bids_pipeline._config.ica_reject] setting.

# If `None` (default), do not apply artifact rejection.

# If a dictionary, manually specify rejection thresholds (see examples). 
# The thresholds provided here must be at least as stringent as those in
# [`ica_reject`][mne_bids_pipeline._config.ica_reject] if using ICA. In case of
# `'autoreject_global'`, thresholds for any channel that do not meet this
# requirement will be automatically replaced with those used in `ica_reject`.

# If `"autoreject_global"`, use [`autoreject`](https://autoreject.github.io) to find
# suitable "global" rejection thresholds for each channel type, i.e., `autoreject`
# will generate a dictionary with (hopefully!) optimal thresholds for each
# channel type.

# If `"autoreject_local"`, use "local" `autoreject` to detect (and potentially repair) bad
# channels in each epoch. Use [`autoreject_n_interpolate`][mne_bids_pipeline._config.autoreject_n_interpolate]
# to control how many channels are allowed to be bad before an epoch gets dropped.

# ???+ example "Example"
#     ```python
#     reject = {"grad": 4000e-13, 'mag': 4e-12, 'eog': 150e-6}
#     reject = {"eeg": 100e-6, "eog": 250e-6}
#     reject = None  # no rejection based on PTP amplitude
#     reject = "autoreject_global"  # find global (per channel type) PTP thresholds
#     reject = "autoreject_local"  # find local (per channel) thresholds and repair epochs
#     ```
# """

# reject_tmin: Optional[float] = None
# """
# Start of the time window used to reject epochs. If `None`, the window will
# start with the first time point. Has no effect if
# [`reject`][mne_bids_pipeline._config.reject] has been set to `"autoreject_local"`.

# ???+ example "Example"
#     ```python
#     reject_tmin = -0.1  # 100 ms before event onset.
#     ```
# """

# reject_tmax: Optional[float] = None
# """
# End of the time window used to reject epochs. If `None`, the window will end
# with the last time point. Has no effect if
# [`reject`][mne_bids_pipeline._config.reject] has been set to `"autoreject_local"`.

# ???+ example "Example"
#     ```python
#     reject_tmax = 0.3  # 300 ms after event onset.
#     ```
# """

# autoreject_n_interpolate: FloatArrayLike = [4, 8, 16]
# """
# The maximum number of bad channels in an epoch that `autoreject` local will try to
# interpolate. The optimal number among this list will be estimated using a
# cross-validation procedure; this means that the more elements are provided here, the
# longer the `autoreject` run will take. If the number of bad channels in an epoch
# exceeds this value, the channels won't be interpolated and the epoch will be dropped.

# !!! info
#     This setting only takes effect if [`reject`][mne_bids_pipeline._config.reject] has
#     been set to `"autoreject_local"`.

# !!! info
#     Channels marked as globally bad in the BIDS dataset (in `*_channels.tsv)`) will not
#     be considered (i.e., will remain marked as bad and not analyzed by autoreject).
# """