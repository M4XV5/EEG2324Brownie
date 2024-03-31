# import all needed libraries
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
import shutil, errno
import pandas as pd
import os
import mne
import tqdm
import pickle
import numpy as np
import re
import pandas as pd
from mne_bids import (BIDSPath,read_raw_bids)

################################

# Process the raw data before passing it to the pipeline

################################

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


# Removing trials/participants
def drop_learning_nonlearner_invalid_trials(n=4, threshold=0.7, subjects_dir='Dataset/ds004147-filtered'):
    # copy dataset an add "-filtered" suffix
    subject_dirs = [d for d in os.listdir(subjects_dir) if os.path.isdir(os.path.join(subjects_dir, d))]
    removed_trial_dict = {}
    for subject_dir in subject_dirs:
        beh_file_path = os.path.join(subjects_dir, subject_dir, 'beh', f'{subject_dir}_task-casinos_beh.tsv')
    
        if os.path.exists(beh_file_path):
            df = pd.read_csv(beh_file_path, sep='\t')
            # remove the first four trials for every cue and block
            filtered_df = df.groupby(['block', 'cue']).apply(lambda x: x.iloc[4:]).reset_index(drop=True)
            filtered_df = filtered_df[filtered_df['invalid'] != 1]
            removed_trials = df.groupby(['block', 'cue']).apply(lambda x: x.iloc[:4]).reset_index(drop=True)
            removed_trials["trial"] = removed_trials["trial"] + (removed_trials["block"] - 1) * 144
            removed_trial_dict[subject_dir] = removed_trials
            df_sorted = filtered_df.sort_values(by=['block', 'trial'])
            # aggregate learnable trials and mark participants that did not meet the threshold
            learnable_trials = filtered_df[filtered_df['prob'] == 80]
            correct_choices = learnable_trials['outcome'] == learnable_trials['optimal']
            success_rate = correct_choices.mean()
            threshold_met = success_rate >= threshold
            if threshold_met:
                print(f"{subject_dir} does meet the threshold and should be included in the analysis.")
                df_sorted.to_csv(os.path.join(subjects_dir, subject_dir, 'beh', f'{subject_dir}_task-casinos_beh.tsv'), sep='\t', index=False)
            else:
                print(f"{subject_dir} does  not meet the threshold and should be not included in the analysis.")
                df_sorted.to_csv(os.path.join(subjects_dir, subject_dir, 'beh', f'{subject_dir}_REJECT_task-casinos_beh.tsv'), sep='\t', index=False)
        else:
            print(f'Behavioral file for {subject_dir} does not exist.')

def drop_trials_events(subject_dirs, subjects_dir, removed_trial_dict): #after removing the trials from the behavioural tsv file, also update the events and marker files
    for subject_dir in subject_dirs:
        eeg_vmrk_path = os.path.join(subjects_dir, subject_dir, 'eeg', f'{subject_dir}_task-casinos_eeg.vmrk')
        eee_events_tsv_path = os.path.join(subjects_dir, subject_dir, 'eeg', f'{subject_dir}_task-casinos_events.tsv')
        if os.path.exists(eeg_vmrk_path):
            with open(eeg_vmrk_path, 'r') as file:
                lines = file.readlines()
            start_index = 13  
            groups = []
            current_group = []
            for i, line in enumerate(lines[start_index:], start=start_index):
                if any(stim in line for stim in ['S  1', 'S 11', 'S 21', 'S 31']):
                    if current_group:
                        groups.append(current_group)
                        current_group = []
                current_group.append(i)
            if current_group:
                groups.append(current_group)
            groups = [(idx+1, element) for idx, element in enumerate(groups)]
            invalid = [(idx, element) for idx, element in groups if len(element) != 5]  # filter invalid trials, i.e. sublist != 5
            print(f"Subject: {subject_dir} had {len(invalid)} invalid trials!")
            # filter the 4 trials for every slot in every casino
            first_four = [(idx, item) for idx, item in groups if idx in removed_trial_dict[subject_dir]["trial"].values]
            first_four_invalid= [item for item in first_four if len(item[1]) != 5]
            # remove duplicate in invalid (invalid trials could be in the first four trials of a slot)
            invalid = [item for item in invalid if item not in first_four_invalid]
            first_four.extend(invalid)
            # convert to list of lines to remove
            first_four = [element[1] for element in first_four]
            lines_to_remove = []
            [lines_to_remove.extend(element) for element in first_four]
            filtered_lines = [line for idx, line in enumerate(lines) if idx not in lines_to_remove]
            
            # create filtered vrmk file
            #print(f"Adapting eeg.vmrk and events.tsv file for {subject_dir}")
            with open(os.path.join(subjects_dir, subject_dir, 'eeg', f'{subject_dir}_task-casinos_eeg.vmrk'), "w") as file:
                # Write each item to the file
                for item in filtered_lines:
                    file.write(item)
            
            
            # do the same for events.tsv file
            with open(eee_events_tsv_path, 'r') as file:
                lines = file.readlines()
            lines_to_remove = [idx - 10 for idx in lines_to_remove]
            filtered_lines = [line for idx, line in enumerate(lines) if idx not in lines_to_remove]
            with open(os.path.join(subjects_dir, subject_dir, 'eeg', f'{subject_dir}_task-casinos_events.tsv'), "w") as file:
                # Write each item to the file
                for item in filtered_lines:
                    file.write(item)
        else:
            print(f'eeg.vmrk file for {subject_dir} does not exist.')


################################

# Linear Modelling

################################


# Key is stimulus id, Value is (task, prob, outcome)
# prob is used as substitute for cue to know which cues are high (prob = 80) and low (prob = 50)
CONDITION_BEH_MAPPING = {
    6: (1, 50, 1),
    16: (2, 50, 1),
    26: (2, 80, 1),
    36: (3, 80, 1),
    7: (1, 50, 0),
    17: (2, 50, 0),
    27: (2, 80, 0),
    37: (3, 80, 0),
}

CONDITION_MAPPING = {
    6: ("low", "low", "win"),
    16: ("mid", "low", "win"),
    26: ("mid", "high", "win"),
    36: ("high", "high", "win"),
    7: ("low", "low", "loss"),
    17: ("mid", "low", "loss"),
    27: ("mid", "high", "loss"),
    37: ("high", "high", "loss"),
}

OUR_CONDITION_MAPPING = {
    'Win LL': 6,
    'Loss LL': 7,
    'Win ML': 16,
    'Loss ML': 17,
    'Win MH': 26,
    'Loss MH': 27,
    'Win HH': 36,
    'Loss HH': 37,
}

# import all needed helper functions
def create_df_from_beh_tsv(tsv_name):
    df = pd.read_csv(tsv_name, sep='\t')
    df = df[['block', 'trial', 'task', 'prob', 'outcome', 'optimal', 'rt']]
    df['id'] = df['block'] * df['trial']
    # Calculate Z-score of RT within each task
    task_mean_rt = df.groupby('task')['rt'].transform('mean')
    task_std_rt = df.groupby('task')['rt'].transform('std')
    df['rt_z_score'] = (df['rt'] - task_mean_rt) / task_std_rt
    # Calculate percentile of RT within each task
    task_percentiles = df.groupby('task')['rt'].transform(lambda x: (x.rank() - 1) / len(x) * 100)
    df['rt_percentile'] = task_percentiles
    # Calculate residuals
    residuals = df.groupby('task')['rt'].transform(lambda x: x - x.mean())
    df['residuals'] = residuals
    return df[['id', 'task', 'prob', 'outcome', 'optimal', 'rt_z_score', 'rt_percentile', 'residuals']]


def create_outcome_figure(df1, df2, title, ax):
    ax.boxplot([df1["RewP"], df2["RewP"]])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Loss (prev)', 'Win (prev)'])
    ax.set_ylabel("RewP")
    ax.set_title(title)


def analyze_categorical_feature(feature, data, title, second_feature=None):
    if second_feature:
        # Pair up the first and second features into one
        data[f'{feature}-{second_feature}'] = data[feature] + '-' + data[second_feature]
        feature = f'{feature}-{second_feature}'

    # Grouping data by the feature and extracting 'RewP' values for each group
    feature_groups = data.groupby(feature)["RewP"].apply(list)

    # Plotting boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(feature_groups.values, labels=feature_groups.index)
    plt.xlabel(feature)
    plt.ylabel("RewP")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Fit ANOVA model
    model = ols(f'{"RewP"} ~ ' + feature, data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # Calculate Eta-squared
    ss_between = anova_table['sum_sq'][feature]
    ss_total = anova_table['sum_sq']['Residual'] + ss_between
    eta_squared = ss_between / ss_total
    print(f"\nEta-squared (η²): {eta_squared}")


def perform_chi_squared_test(data, categorical_var1, categorical_var2):
    contingency_table = pd.crosstab(data[categorical_var1], data[categorical_var2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print("Chi-squared test results:")
    print(f"Chi-square statistic: {chi2}")
    print(f"P-value: {p}")
    print(f"Degrees of freedom: {dof}")
    print("Contingency Table (Observed frequencies):")
    print(contingency_table)
    print("Expected frequencies:")
    print(expected)


def analyze_numerical_feature(feature, data, title):
    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(data[feature], data["RewP"], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel("RewP")
    plt.title(f'Scatter Plot of RewP vs {feature} for {title}')
    plt.grid(True)
    plt.show()

    # Correlation analysis
    correlation, p_value = pearsonr(data[feature], data["RewP"])
    print(f"Correlation between {feature} and RewP: {correlation}")
    print(f"P-value: {p_value}")

    # Calculate R-squared
    r_squared = correlation ** 2
    print(f"R-squared: {r_squared}")


def convert_condition_id_to_beh_names(condition_id):
    return CONDITION_BEH_MAPPING[condition_id]


def filter_df_by_kwargs(df, **kwargs):
    mask = pd.Series([True] * len(df))
    for col, value in kwargs.items():
        mask = mask & (df[col] == value)
    return df[mask]


def extract_number(value):
    try:
        import re
        return float(re.findall(r'\d+', value)[0])
    except (IndexError, ValueError):
        return np.nan


def retrieve_data(selected_channel, filt_ds_path, raw_ds_path, output_dir, subjects_to_include, use_beh):
    df = pd.DataFrame(columns=["max_voltage", "mean_voltage", "subject", "outcome", "task", "cue", "prev_outcome"])
    max_voltages = []
    mean_voltages = []
    subjects = []
    outcomes = []
    tasks = []
    cues = []
    prev_outcomes = []
    subject_names = [d for d in os.listdir(filt_ds_path) if os.path.isdir(os.path.join(filt_ds_path, d)) and "sub" in d]
    for subject_name in subject_names:
        if subjects_to_include is not None:
            if subject_name not in subjects_to_include:
                continue
        # load paths
        good_epochs_path = os.path.join(filt_ds_path, "derivatives", "mne-bids-pipeline", subject_name, "eeg",
                                        f"{subject_name}_task-casinos_proc-clean_epo.fif")
        if use_beh:
            raw_beh_path = os.path.join(raw_ds_path, subject_name, "beh", f"{subject_name}_task-casinos_beh.tsv")
            raw_casinos_events = os.path.join(raw_ds_path, subject_name, "eeg",
                                              f"{subject_name}_task-casinos_events.tsv")
            if any([not os.path.exists(good_epochs_path), not os.path.exists(raw_casinos_events),
                    not os.path.exists(raw_beh_path)]):
                print(f"{subject_name} is missing either good_epochs_file, raw_beh_path or raw_casinos_events!")
                continue
            # load files
            beh = pd.read_csv(raw_beh_path, sep='\t')
            beh['prev_outcome'] = beh['outcome'].shift(1)
            raw_condition_event_ids = pd.read_csv(raw_casinos_events, sep='\t')
            raw_condition_event_ids["value"] = raw_condition_event_ids["value"].astype(str)

        epochs = mne.read_epochs(good_epochs_path, preload=True)
        # pick channel
        epochs = epochs.pick(picks=[selected_channel])
        event_ids = epochs.events

        for condition in epochs.event_id.keys():
            condition_epochs = epochs[condition]
            condition_id = OUR_CONDITION_MAPPING[condition]
            if use_beh:
                condition_event_ids = event_ids[event_ids[:, 2] == epochs.event_id[condition]]
                raw_condition_event_ids["number"] = raw_condition_event_ids["value"].apply(extract_number)
                condition_condition_event_ids = raw_condition_event_ids[
                    raw_condition_event_ids["number"] == condition_id]
                filtered_condition_idxs = [i for i, sample in enumerate(condition_condition_event_ids["sample"]) if
                                           int(sample) - 1 in condition_event_ids[:, 0]]
                task, prob, outcome = convert_condition_id_to_beh_names(condition_id)
                filtered_beh = filter_df_by_kwargs(beh, task=task, prob=prob, outcome=outcome)
                filtered_beh.index = range(len(filtered_beh))
                filtered_beh = filtered_beh.loc[filtered_condition_idxs]
            time_series = []
            for i, epoch in enumerate(condition_epochs):
                epoch = epoch.flatten() * 1000000
                epoch = epoch[440:540]
                max_voltage = np.max(epoch)
                mean_voltage = np.mean(epoch)
                if use_beh:
                    outcome = "win" if filtered_beh.iloc[i]["outcome"] == 1 else "loss"
                    if filtered_beh.iloc[i]["prev_outcome"] == 1:
                        prev_outcome = "win"
                    elif filtered_beh.iloc[i]["prev_outcome"] == 0:
                        prev_outcome = "loss"
                    elif filtered_beh.iloc[i]["prev_outcome"] == -1:
                        prev_outcome = "invalid"
                    else:
                        raise Exception("Something went wrong with the prev_outcome assignment")
                    if filtered_beh.iloc[i]["task"] == 1:
                        task = "low"
                    elif filtered_beh.iloc[i]["task"] == 2:
                        task = "mid"
                    elif filtered_beh.iloc[i]["task"] == 3:
                        task = "high"
                    else:
                        raise Exception("Something went wrong with the task assignment")
                    cue = "low" if filtered_beh.iloc[i]["prob"] == 50 else "high"
                    prev_outcomes.append(prev_outcome)
                else:
                    task, cue, outcome = CONDITION_MAPPING[condition_id]
                mean_voltages.append(mean_voltage)
                max_voltages.append(max_voltage)
                cues.append(cue)
                outcomes.append(outcome)
                subjects.append(subject_name)
                tasks.append(task)
    if use_beh:
        df = pd.DataFrame({
            "subject": subjects,
            "task": tasks,
            "cue": cues,
            "prev_outcome": prev_outcomes,
            "outcome": outcomes,
            "mean_voltage": mean_voltages,
            "max_voltage": max_voltages
        })
    else:
        df = pd.DataFrame({
            "subject": subjects,
            "task": tasks,
            "cue": cues,
            "outcome": outcomes,
            "mean_voltage": mean_voltages,
            "max_voltage": max_voltages,
        })
    performance = pd.read_csv("data/performances.csv")
    df = pd.merge(df, performance, on="subject", how="left")
    if output_dir is not None:
        df.to_csv(output_dir, index=False)
    return df


def calculate_RewP_isolated_by_feature(df, feature=None):
    # Group data by fixed features and calculate average mean voltage
    if feature is not None:
        grouped_data = df.groupby([feature, 'task', 'cue', 'performance', "outcome"])['max_voltage'].mean().reset_index()
        rew_p = df.pivot_table(index=[feature, 'task', 'cue', 'performance'],
                               columns='outcome',
                               values='max_voltage',
                               aggfunc='mean').reset_index()
    else:
        grouped_data = df.groupby(['task', 'cue', 'performance', "outcome"])['max_voltage'].mean().reset_index()
        rew_p = df.pivot_table(index=['task', 'cue', 'performance'],
                               columns='outcome',
                               values='max_voltage',
                               aggfunc='mean').reset_index()

    rew_p['RewP'] = rew_p['win'] - rew_p['loss']
    rew_p = rew_p.dropna(subset=['RewP'])
    if feature is not None:
        rew_p[[feature, 'task', 'cue', 'performance', 'RewP']]
    else:
        rew_p[['task', 'cue', 'performance', 'RewP']]
    return rew_p


def create_scatter_from_categorical_and_numerical_feature(df, y_col, x_col, categorical_col):
    categories = df[categorical_col].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']

    plt.figure(figsize=(8, 6))
    for i, category in enumerate(categories):
        category_data = df[df[categorical_col] == category]
        plt.scatter(category_data[x_col], category_data[y_col], label=category, color=colors[i % len(colors)],
                    alpha=0.5, s=1)

    plt.xlabel('Numerical Variable')
    plt.ylabel('Dependent Variable')
    plt.title('Scatter Plot of Numerical Variable vs. Dependent Variable by Category')
    plt.legend(title='Category')
    plt.grid(True)
    plt.show()


def plot_boxplot_and_sse(data, column_name, title):
    plt.figure(figsize=(8, 6))
    plt.boxplot(data[column_name])
    plt.ylabel(column_name)
    plt.title(title)
    plt.show()

    data['squared_errors'] = (data[column_name] - data[column_name].mean()) ** 2
    sse = data['squared_errors'].sum()
    print("Sum of Squared Errors (SSE):", sse)
    data[column_name].describe()


################################

# Miscellaneous

################################

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc:
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise

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