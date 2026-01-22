import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from scipy.stats import mannwhitneyu


def extract_day(filename):
    """
Extract the day label (e.g., 'Day03') from a filename.

Args:
    filename (str): Filename containing a substring like 'DayXX'.

Returns:
    str or None: The matched day string, or None if no day pattern is found.
"""

    match = re.search(r"(Day\d{2})", filename)
    return match.group(1) if match else None  




def extract_trial(filename):
    """
Extract the trial number from a filename.

Args:
    filename (str): Filename containing a pattern like '_<trialNumber>'.

Returns:
    int: Extracted trial number, or 0 if no valid pattern is found.
"""

    match = re.search(r'_(\d+)[^0-9]', filename)  
    return int(match.group(1)) if match else 0  


def plot_avg_trial_durations(directory_paths, fsamp=2048, save_path=None):
    """
    Plots the average trial duration per day for multiple subjects.

    Parameters:
    - directory_paths: dict, keys are labels (e.g., "S1") and values are paths to folders
    - fsamp: int, sampling frequency in Hz
    - save_path: str or None, if given, saves figure to this path
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Adjust size/grid if more subjects
    axes = axes.flatten()

    for idx, (label, directory_path) in enumerate(directory_paths.items()):
        # Load files
        pkl_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]

        # Group by day
        files_by_day = {}
        for file in pkl_files:
            day = extract_day(file)
            if day:
                files_by_day.setdefault(day, []).append(file)

        session_days = sorted(files_by_day.keys())

        for day in session_days:
            files_by_day[day].sort(key=extract_trial)

        sorted_pkl_files = [file for day in session_days for file in files_by_day[day]]
        trial_durations_by_day = {day: [] for day in session_days}

        for file_name in sorted_pkl_files:
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            force = np.array(data['Force'].tolist())[0]
            duration = len(force) / fsamp
            day = extract_day(file_name)
            if day:
                trial_durations_by_day[day].append(duration)

        avg_durations = [np.mean(trial_durations_by_day[day]) for day in session_days]
        std_durations = [np.std(trial_durations_by_day[day]) for day in session_days]

        ax = axes[idx]
        ax.bar(session_days, avg_durations, yerr=std_durations, color='gray',
               edgecolor='black', alpha=0.7, capsize=5)

        ax.set_xticks(np.arange(len(session_days)))
        ax.set_xticklabels([str(int(day[3:])) for day in session_days], fontsize=14, rotation=45)
        ax.set_yticks(np.arange(0, 36, 5))
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=6, width=2)
        ax.set_title(label, fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



def plot_first_last_two_days_duration_box(directory_path, fsamp):
    from matplotlib import rcParams
    rcParams['font.family'] = 'Arial'
    rcParams['axes.labelsize'] = 16
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14

    pkl_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]
    files_by_day = {}
    for file in pkl_files:
        day = extract_day(file)
        if day:
            files_by_day.setdefault(day, []).append(file)

    session_days = sorted(files_by_day.keys())
    if len(session_days) < 4:
        raise ValueError("Need at least 4 days.")

    first_days = session_days[:2]
    last_days = session_days[-2:]

    records = []
    for label, days in zip(["First sessions", "Last sessions"], [first_days, last_days]):
        for day in days:
            files_by_day[day].sort(key=extract_trial)
            for file in files_by_day[day]:
                file_path = os.path.join(directory_path, file)
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                force = np.array(data["Force"].tolist())[0]
                duration = len(force) / fsamp
                records.append({"Group": label, "Day": day, "Duration": duration})

    df = pd.DataFrame(records)

    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    sns.stripplot(data=df, x="Group", y="Duration", hue="Day",
                  dodge=True, jitter=0.25, alpha=0.6, size=8,
                  palette="tab10", edgecolor='gray', linewidth=0.3, ax=ax)

    sns.boxplot(data=df, x="Group", y="Duration",
                showcaps=True, showfliers=False,
                whiskerprops={'linewidth': 1.2},
                boxprops={'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 1.5},
                medianprops={'color': 'black', 'linewidth': 2},
                width=0.3, ax=ax)

    ax.set_ylabel("Trial duration (s)", fontsize=18)
    ax.set_xlabel("")
    ax.set_ylim([0, 40])
    ax.set_xticklabels(["First sessions", "Last sessions"], fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    sns.despine(trim=True)
    plt.legend().remove()
    plt.tight_layout()
    return plt.gcf()


def run_duration_stats(directory_path, fsamp):
    pkl_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]
    files_by_day = {}
    for file in pkl_files:
        day = extract_day(file)
        if day:
            files_by_day.setdefault(day, []).append(file)

    session_days = sorted(files_by_day.keys())
    if len(session_days) < 4:
        raise ValueError("Need at least 4 days.")

    first_days = session_days[:2]
    last_days = session_days[-2:]

    durations_first = []
    durations_last = []

    # Step 2: Compute durations
    for group_days, storage in zip([first_days, last_days], [durations_first, durations_last]):
        for day in group_days:
            files_by_day[day].sort(key=extract_trial)
            for file in files_by_day[day]:
                file_path = os.path.join(directory_path, file)
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                force = np.array(data['Force'].tolist())[0]
                duration = len(force) / fsamp
                storage.append(duration)

    # Step 3: Mann-Whitney U test
    stat, p_value = mannwhitneyu(durations_first, durations_last, alternative='two-sided')
    print(f"Mann-Whitney U test:")
    print(f"U statistic = {stat:.3f}")
    print(f"p-value = {p_value:.6f}")

    return stat, p_value




def collect_duration_single_participant(directory_path, fsamp=2048):
    """
    For a single participant: build a DataFrame with one row per trial.

    Columns:
      - day_label  (e.g. 'Day01')
      - day_index  (1, 2, 3, ...)
      - trial      (trial number within day)
      - duration   (trial duration in seconds)
    """
    pkl_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]

    # group files by day
    files_by_day = {}
    for file in pkl_files:
        day_label = extract_day(file)
        if day_label:
            files_by_day.setdefault(day_label, []).append(file)

    session_days = sorted(files_by_day.keys())
    if len(session_days) == 0:
        raise ValueError("No valid days found in this directory.")

    # map day label -> integer index (1, 2, 3, ...)
    day_to_index = {day_label: i + 1 for i, day_label in enumerate(session_days)}

    rows = []

    for day_label in session_days:
        trial_files = files_by_day[day_label]
        # sort trials within the day
        trial_files.sort(key=extract_trial)

        for file in trial_files:
            file_path = os.path.join(directory_path, file)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # assuming data["Force"] is array-like
            force = np.array(data["Force"].tolist())[0]
            duration = len(force) / fsamp  # seconds

            rows.append({
                "day_label": day_label,
                "day_index": day_to_index[day_label],
                "trial": extract_trial(file),
                "duration": duration
            })

    df = pd.DataFrame(rows)
    return df



def fit_duration_trend_single_participant(directory_path, fsamp=2048):
    """
    For a single participant:
      - build trial-level DataFrame
      - fit linear model: duration ~ day_index
      - print slope and p-value
    """
    df = collect_duration_single_participant(directory_path, fsamp)

    # Fit simple linear regression (OLS): duration ~ day_index
    model = smf.ols("duration ~ day_index", data=df)
    result = model.fit()

    print("=== Linear model: duration ~ day_index ===")
    print(result.summary())

    slope = result.params["day_index"]
    p_val = result.pvalues["day_index"]

    print(f"\nEffect of day_index (slope): {slope:.4f} s/day")
    print(f"p-value for day_index: {p_val:.6f}")

    return df, result
