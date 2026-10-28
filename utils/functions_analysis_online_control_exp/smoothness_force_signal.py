import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy.stats import mannwhitneyu
import statsmodels.formula.api as smf


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


def compute_smoothness(force_signal, fsamp, high_freq_threshold=5.0):
    """
    Compute a smoothness metric based on high-frequency content of a force signal.
    """
    freqs = np.fft.rfftfreq(len(force_signal), d=1/fsamp)
    amp_spectrum = np.abs(np.fft.rfft(force_signal)) / len(force_signal)
    smoothness = np.sum(amp_spectrum[freqs > high_freq_threshold])
    return smoothness


def load_smoothness_data(directory_path, fsamp):
    """
    Helper function to load data and compute smoothness values.
    Returns a DataFrame with Group, Day, and Smoothness columns.
    """
    pkl_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]
    files_by_day = {}
    for file in pkl_files:
        day = extract_day(file)
        if day is not None:
            files_by_day.setdefault(day, []).append(file)

    session_days = sorted(files_by_day.keys())
    if len(session_days) < 4:
        raise ValueError("Need at least 4 days to compute first vs last two.")

    first_days = session_days[:2]
    last_days = session_days[-2:]

    records = []
    for group_label, group_days in zip(["First sessions", "Last sessions"], [first_days, last_days]):
        for day in group_days:
            files_by_day[day].sort(key=extract_trial)
            for file_name in files_by_day[day]:
                file_path = os.path.join(directory_path, file_name)
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

                force_signal = np.array(data['Force'].tolist())[0]
                smoothness = compute_smoothness(force_signal, fsamp)
                records.append({"Group": group_label, "Day": day, "Smoothness": smoothness})

    return pd.DataFrame(records)


def compute_signal_smoothness(directory_path, fsamp):
    """
    Computes smoothness values and creates visualization (no statistical test).
    """
    rcParams['font.family'] = 'Arial'
    rcParams['axes.labelsize'] = 16
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14

    df = load_smoothness_data(directory_path, fsamp)

    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    sns.stripplot(data=df, x="Group", y="Smoothness", hue="Day", dodge=True, 
                  jitter=0.25, alpha=0.5, size=8, ax=ax, palette="tab10", 
                  edgecolor='gray', linewidth=0.2)

    sns.boxplot(data=df, x="Group", y="Smoothness", showcaps=True, showfliers=False,
                whiskerprops={'linewidth': 1.2},
                boxprops={'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 1.5},
                medianprops={'color': 'black', 'linewidth': 2}, width=0.3, ax=ax)

    ax.set_ylabel("Jitter", fontsize=18, labelpad=10)
    ax.set_xlabel("")
    ax.set_xticklabels(["First sessions", "Last sessions"], fontsize=16)
    sns.despine(trim=True)
    plt.tight_layout()

    return plt.gcf()


def compute_signal_smoothness_mannwhitney(directory_path, fsamp):
    """
    Computes smoothness values and compares early vs late sessions
    using Mann-Whitney U test.
    
    Note: This test treats all trials as independent observations.
    Be aware of potential pseudoreplication if trials within days are correlated.
    """
    rcParams['font.family'] = 'Arial'
    rcParams['axes.labelsize'] = 16
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14

    df = load_smoothness_data(directory_path, fsamp)

    # Extract values for each group
    early_vals = df[df["Group"] == "First sessions"]["Smoothness"].values
    late_vals = df[df["Group"] == "Last sessions"]["Smoothness"].values

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
    print(f"Median early = {np.median(early_vals):.4f}")
    print(f"Median late = {np.median(late_vals):.4f}")
    print("=" * 50)

    # Plot
    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    sns.stripplot(data=df, x="Group", y="Smoothness", hue="Day", dodge=True, 
                  jitter=0.25, alpha=0.5, size=8, ax=ax, palette="tab10", 
                  edgecolor='gray', linewidth=0.2)

    sns.boxplot(data=df, x="Group", y="Smoothness", showcaps=True, showfliers=False,
                whiskerprops={'linewidth': 1.2},
                boxprops={'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 1.5},
                medianprops={'color': 'black', 'linewidth': 2}, width=0.3, ax=ax)

    ax.set_ylabel("Jitter", fontsize=18, labelpad=10)
    ax.set_xlabel("")
    ax.set_xticklabels(["First sessions", "Last sessions"], fontsize=16)
    ax.set_title(f"Mann-Whitney U: p = {pval:.4g}", fontsize=14)
    sns.despine(trim=True)
    plt.tight_layout()

    return plt.gcf(), stat, pval, rank_biserial


def compute_signal_smoothness_lmm(directory_path, fsamp):
    """
    Computes smoothness values and compares early vs late sessions
    using a Linear Mixed Model with Day as random intercept.
    
    This approach accounts for the correlation between trials within the same day.
    """
    rcParams['font.family'] = 'Arial'
    rcParams['axes.labelsize'] = 16
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14

    df = load_smoothness_data(directory_path, fsamp)

    # Set categorical order (First sessions as reference)
    df["Group"] = pd.Categorical(
        df["Group"],
        categories=["First sessions", "Last sessions"],
        ordered=True
    )

    # Fit Linear Mixed Model
    model = smf.mixedlm(
        "Smoothness ~ Group",
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
    print(f"Effect (Late - Early): {effect:.4f}")
    print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"p-value: {pval:.4g}")
    print("=" * 50)

    # Plot
    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    sns.stripplot(data=df, x="Group", y="Smoothness", hue="Day", dodge=True, 
                  jitter=0.25, alpha=0.5, size=8, ax=ax, palette="tab10", 
                  edgecolor='gray', linewidth=0.2)

    sns.boxplot(data=df, x="Group", y="Smoothness", showcaps=True, showfliers=False,
                whiskerprops={'linewidth': 1.2},
                boxprops={'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 1.5},
                medianprops={'color': 'black', 'linewidth': 2}, width=0.3, ax=ax)

    ax.set_ylabel("Jitter", fontsize=18, labelpad=10)
    ax.set_xlabel("")
    ax.set_xticklabels(["First sessions", "Last sessions"], fontsize=16)
    ax.set_title(f"LMM: β = {effect:.4f}, p = {pval:.4g}", fontsize=14)
    sns.despine(trim=True)
    plt.tight_layout()

    return plt.gcf(), result, effect, pval
