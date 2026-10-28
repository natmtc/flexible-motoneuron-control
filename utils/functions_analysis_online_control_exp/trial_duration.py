import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from scipy.stats import mannwhitneyu
import statsmodels.formula.api as smf
from matplotlib import rcParams


def extract_day(filename):
    """
    Extract the day label (e.g., 'Day03') from a filename.
    """
    match = re.search(r"(Day\d{2})", filename)
    return match.group(1) if match else None  


def extract_trial(filename):
    """
    Extract the trial number from a filename.
    """
    match = re.search(r'_(\d+)[^0-9]', filename)  
    return int(match.group(1)) if match else 0  


def load_duration_data(directory_path, fsamp):
    """
    Helper function to load data and compute trial durations for first/last sessions.
    Returns a DataFrame with Group, Day, and Duration columns.
    """
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

    return pd.DataFrame(records)


def plot_avg_trial_durations(directory_paths, fsamp=2048, save_path=None):
    """
    Plots the average trial duration per day for multiple subjects.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (label, directory_path) in enumerate(directory_paths.items()):
        pkl_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]

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
    """
    Creates boxplot visualization (no statistical test).
    """
    rcParams['font.family'] = 'Arial'
    rcParams['axes.labelsize'] = 16
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14

    df = load_duration_data(directory_path, fsamp)

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


def run_duration_stats_mannwhitney(directory_path, fsamp):
    """
    Compares trial durations between early and late sessions using Mann-Whitney U test.
    
    Note: This test treats all trials as independent observations.
    Be aware of potential pseudoreplication if trials within days are correlated.
    """
    df = load_duration_data(directory_path, fsamp)

    early_vals = df[df["Group"] == "First sessions"]["Duration"].values
    late_vals = df[df["Group"] == "Last sessions"]["Duration"].values

    # Mann-Whitney U test
    stat, pval = mannwhitneyu(early_vals, late_vals, alternative='two-sided')

    # Effect size: rank-biserial correlation
    n1, n2 = len(early_vals), len(late_vals)
    rank_biserial = 1 - (2 * stat) / (n1 * n2)

    print("=" * 50)
    print("Mann-Whitney U Test")
    print("=" * 50)
    print(f"U = {stat:.3f}")
    print(f"p-value = {pval:.4g}")
    print(f"n (early) = {n1}, n (late) = {n2}")
    print(f"Rank-biserial correlation = {rank_biserial:.3f}")
    print(f"Median early = {np.median(early_vals):.4f} s")
    print(f"Median late = {np.median(late_vals):.4f} s")
    print("=" * 50)

    return stat, pval, rank_biserial


def run_duration_stats_lmm(directory_path, fsamp):
    """
    Compares trial durations between early and late sessions using 
    Linear Mixed Model with Day as random intercept.
    
    This approach accounts for the correlation between trials within the same day.
    """
    df = load_duration_data(directory_path, fsamp)

    # Set categorical order (First sessions as reference)
    df["Group"] = pd.Categorical(
        df["Group"],
        categories=["First sessions", "Last sessions"],
        ordered=True
    )

    # Fit Linear Mixed Model
    model = smf.mixedlm(
        "Duration ~ Group",
        data=df,
        groups=df["Day"]
    )
    result = model.fit()

    # Extract key statistics
    effect = result.fe_params["Group[T.Last sessions]"]
    pval = result.pvalues["Group[T.Last sessions]"]
    ci = result.conf_int().loc["Group[T.Last sessions]"]

    print("=" * 50)
    print("Linear Mixed Model (Day as random intercept)")
    print("=" * 50)
    print(result.summary())
    print("\n" + "-" * 50)
    print(f"Effect (Late - Early): {effect:.4f} s")
    print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"p-value: {pval:.4g}")
    print("=" * 50)

    return result, effect, pval


def plot_duration_mannwhitney(directory_path, fsamp):
    """
    Creates boxplot with Mann-Whitney U test results in title.
    """
    rcParams['font.family'] = 'Arial'
    rcParams['axes.labelsize'] = 16
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14

    df = load_duration_data(directory_path, fsamp)

    # Run Mann-Whitney
    early_vals = df[df["Group"] == "First sessions"]["Duration"].values
    late_vals = df[df["Group"] == "Last sessions"]["Duration"].values
    stat, pval = mannwhitneyu(early_vals, late_vals, alternative='two-sided')

    # Plot
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
    ax.set_title(f"Mann-Whitney U: p = {pval:.4g}", fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    sns.despine(trim=True)
    plt.legend().remove()
    plt.tight_layout()

    return plt.gcf(), stat, pval


def plot_duration_lmm(directory_path, fsamp):
    """
    Creates boxplot with LMM results in title.
    """
    rcParams['font.family'] = 'Arial'
    rcParams['axes.labelsize'] = 16
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14

    df = load_duration_data(directory_path, fsamp)

    # Set categorical order
    df["Group"] = pd.Categorical(
        df["Group"],
        categories=["First sessions", "Last sessions"],
        ordered=True
    )

    # Fit LMM
    model = smf.mixedlm("Duration ~ Group", data=df, groups=df["Day"])
    result = model.fit()

    effect = result.fe_params["Group[T.Last sessions]"]
    pval = result.pvalues["Group[T.Last sessions]"]

    # Plot
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
    ax.set_title(f"LMM: β = {effect:.2f} s, p = {pval:.4g}", fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    sns.despine(trim=True)
    plt.legend().remove()
    plt.tight_layout()

    return plt.gcf(), result, effect, pval


def collect_duration_single_participant(directory_path, fsamp=2048):
    """
    For a single participant: build a DataFrame with one row per trial.
    """
    pkl_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]

    files_by_day = {}
    for file in pkl_files:
        day_label = extract_day(file)
        if day_label:
            files_by_day.setdefault(day_label, []).append(file)

    session_days = sorted(files_by_day.keys())
    if len(session_days) == 0:
        raise ValueError("No valid days found in this directory.")

    day_to_index = {day_label: i + 1 for i, day_label in enumerate(session_days)}

    rows = []
    for day_label in session_days:
        trial_files = files_by_day[day_label]
        trial_files.sort(key=extract_trial)

        for file in trial_files:
            file_path = os.path.join(directory_path, file)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            force = np.array(data["Force"].tolist())[0]
            duration = len(force) / fsamp

            rows.append({
                "day_label": day_label,
                "day_index": day_to_index[day_label],
                "trial": extract_trial(file),
                "duration": duration
            })

    return pd.DataFrame(rows)


def fit_duration_trend_single_participant(directory_path, fsamp=2048):
    """
    For a single participant: fit linear model duration ~ day_index.
    """
    df = collect_duration_single_participant(directory_path, fsamp)

    model = smf.ols("duration ~ day_index", data=df)
    result = model.fit()

    print("=== Linear model: duration ~ day_index ===")
    print(result.summary())

    slope = result.params["day_index"]
    p_val = result.pvalues["day_index"]

    print(f"\nEffect of day_index (slope): {slope:.4f} s/day")
    print(f"p-value for day_index: {p_val:.6f}")

    return df, result
