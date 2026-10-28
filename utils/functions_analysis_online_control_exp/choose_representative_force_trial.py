import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_day(filename):
    """
Extract the day label (e.g., 'Day01') from a filename.

Args:
    filename (str): Filename containing a day pattern like 'DayXX'.

Returns:
    str or None: The extracted day string, or None if no match is found.
"""

    m = re.search(r"(Day\d{2})", filename)
    return m.group(1) if m else None




def extract_trial(filename):
    """
Extract the trial number from a filename.

Args:
    filename (str): Filename containing a pattern like '_<trialNumber>'.

Returns:
    int: Extracted trial number, or 0 if no numeric trial identifier is found.
"""

    m = re.search(r'_(\d+)[^0-9]', filename)
    return int(m.group(1)) if m else 0




def load_force(file_path):
    """
Load and sanitize force data from a pickle file.

Args:
    file_path (str): Path to a pickle file containing a dict with key 'Force'.

Process:
    - Loads the pickle dictionary.
    - Extracts the 'Force' entry, handling cases where it may be a list-like
      structure or nested array.
    - Squeezes to a 1D array and converts to float.

Returns:
    np.ndarray: 1D force signal as floats.
"""

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    try:
        x = np.array(data['Force']).squeeze()
        if x.ndim > 1:
            x = np.array(data['Force'].tolist())[0].squeeze()
    except Exception:
        x = np.array(data['Force'].tolist())[0].squeeze()
    return np.asarray(x, float)




def duration_vs_smoothness_scatter_with_median(directory_path, fsamp=2048, high_freq_threshold=5.0,
    keep_first_last=True, annotate=True, save_path=None):
    """
Scatter plot of trial duration vs. smoothness across days, with per-day
representative trials and median lines.

Args:
    directory_path (str): Directory containing .pkl files with 'Force' data.
    fsamp (float): Sampling frequency of the force signal in Hz.
    high_freq_threshold (float): Cutoff in Hz for smoothness metric
        (higher value = more noise).
    keep_first_last (bool): If True and ≥4 days exist, plot only the first
        2 and last 2 days; otherwise plot all days.
    annotate (bool): Unused (placeholder for future annotation features).
    save_path (str or None): If given, save the figure to this path.

Process:
    - Collects all .pkl files and groups them by day.
    - For each file, computes trial duration and smoothness.
    - For each day, selects a representative trial closest to the
      (duration, smoothness) median (in normalized distance).
    - Plots duration vs. smoothness for selected days, highlighting
      representative trials and drawing median lines for the first plotted day.
    - Prints the table of representative trials across all days.

Returns:
    tuple:
        df_all (pd.DataFrame): All trials with columns:
            ["Day", "Trial", "TrialDur_s", "Smoothness", "File"].
        selected_all (pd.DataFrame): One representative trial per day.
"""

    # Collect files by day (ALL days)
    pkl_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]
    files_by_day = {}
    for f in pkl_files:
        day = extract_day(f)
        if day is not None:
            files_by_day.setdefault(day, []).append(f)
    for day in files_by_day:
        files_by_day[day].sort(key=extract_trial)

    days_all = sorted(files_by_day.keys())
    # Days to plot (first 2 + last 2), but compute for all
    if keep_first_last and len(days_all) >= 4:
        days_to_plot = days_all[:2] + days_all[-2:]
    else:
        days_to_plot = days_all

    # Build records for ALL days
    records = []
    for day in days_all:
        for fname in files_by_day[day]:
            fpath = os.path.join(directory_path, fname)
            try:
                x = load_force(fpath)
            except Exception:
                continue
            
            trial_dur = len(x) / fsamp
            smoothness = compute_smoothness(x, fsamp, high_freq_threshold)
            records.append({
                "Day": day,
                "Trial": extract_trial(fname),
                "TrialDur_s": trial_dur,
                "Smoothness": smoothness,
                "File": fname
            })
    if not records:
        print("No valid trials found.")
        return None, None

    df_all = pd.DataFrame(records)

    # Picks (per-day median) for ALL days
    picks = []
    for day, g in df_all.groupby("Day", sort=True):
        mx = g["TrialDur_s"].median()
        my = g["Smoothness"].median()
        iqr_x = np.percentile(g["TrialDur_s"], 75) - np.percentile(g["TrialDur_s"], 25)
        iqr_y = np.percentile(g["Smoothness"], 75) - np.percentile(g["Smoothness"], 25)
        sx = iqr_x if iqr_x > 0 else (g["TrialDur_s"].std() or 1.0)
        sy = iqr_y if iqr_y > 0 else (g["Smoothness"].std() or 1.0)
        dist = np.sqrt(((g["TrialDur_s"]-mx)/sx)**2 + ((g["Smoothness"]-my)/sy)**2)
        pick = g.loc[dist.idxmin(), ["Day","Trial","File","TrialDur_s","Smoothness"]]
        picks.append(pick)
    selected_all = pd.DataFrame(picks)

    # === Plot ONLY the 4 days (or all, if <4) ===
    df_plot = df_all[df_all["Day"].isin(days_to_plot)].copy()
    selected_plot = selected_all[selected_all["Day"].isin(days_to_plot)].copy()

    fig, ax = plt.subplots(figsize=(8,5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for color_idx, (day, g) in enumerate(df_plot.groupby("Day")):
        day_color = colors[color_idx % len(colors)]  # unique color per day
        first = True
        for _, row in g.iterrows():
            if row["TrialDur_s"] > 17:
                ax.scatter(
                    row["TrialDur_s"], row["Smoothness"],
                    s=60, facecolor=day_color,
                    edgecolor="gray", linewidth=3,
                    alpha=0.9,
                    label=day if first else None
                )
            else:
                ax.scatter(
                    row["TrialDur_s"], row["Smoothness"],
                    s=60, facecolor=day_color,
                    edgecolor="black", linewidth=0.5,
                    alpha=0.9,
                    label=day if first else None
                )
            first = False


    # highlight chosen trials (red ring) for plotted days
    for _, r in selected_plot.iterrows():
        ax.scatter(r["TrialDur_s"], r["Smoothness"], s=200, facecolors="none",edgecolors="red", linewidths=2.0, zorder=6)

    # Median lines for the FIRST plotted day
    first_day = sorted(days_to_plot)[0]
    median_smoothness_first = df_plot[df_plot["Day"] == first_day]["Smoothness"].median()
    median_dur_first = df_plot[df_plot["Day"] == first_day]["TrialDur_s"].median()
    ax.axhline(median_smoothness_first, ls="--", color="red", lw=1.5, label=f"{first_day} median smoothness")
    ax.axvline(median_dur_first, ls="--", color="blue", lw=1.5, label=f"{first_day} median duration")

    ax.set_xlabel("Trial duration (s)")
    ax.set_ylabel("Smoothness (higher value means noisier)")
    ax.legend(title="Session day", bbox_to_anchor=(1.02,1), loc="upper left")
    ax.tick_params(axis='both', which='minor', length=4, width=1, color='black')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Print ALL selected (not just the 4)
    print(selected_all)
    return df_all, selected_all





def compute_smoothness(force_signal, fsamp, high_freq_threshold=5.0):
    """
Compute a simple smoothness metric from the force signal's frequency spectrum.

Args:
    force_signal (array-like): 1D force trace.
    fsamp (float): Sampling frequency in Hz.
    high_freq_threshold (float): Frequencies above this (in Hz) are treated as
        “high-frequency noise”.

Process:
    - Computes the one-sided FFT amplitude spectrum.
    - Sums the amplitudes above the chosen frequency threshold.
      (Higher values indicate noisier, less smooth signals.)

Returns:
    float: Smoothness metric (total high-frequency amplitude).
"""

    freqs = np.fft.rfftfreq(len(force_signal), d=1/fsamp)
    amp_spectrum = np.abs(np.fft.rfft(force_signal)) / len(force_signal)
    smoothness = np.sum(amp_spectrum[freqs > high_freq_threshold])
    return smoothness