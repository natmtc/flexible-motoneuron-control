import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns


def extract_day(filename):
    match = re.search(r"(Day\d{2})", filename)
    return match.group(1) if match else None  

def extract_trial(filename):
    match = re.search(r'_(\d+)[^0-9]', filename)  
    return int(match.group(1)) if match else 0  

def compute_smoothness(force_signal, fsamp, high_freq_threshold=5.0):
    freqs = np.fft.rfftfreq(len(force_signal), d=1/fsamp)
    amp_spectrum = np.abs(np.fft.rfft(force_signal)) / len(force_signal)
    smoothness = np.sum(amp_spectrum[freqs > high_freq_threshold])
    return smoothness

def compute_signal_smoothness(directory_path, fsamp):
    """
    Computes smoothness values of force signals across sessions.
    
    This function loads all `.pkl` files from a directory, extracts the 
    force signals, computes a smoothness metric for each trial, and groups 
    results into "First sessions" (first two days) and "Last sessions" 
    (last two days) for later statistical comparison or visualization.
    """
    rcParams['font.family'] = 'Arial'
    rcParams['axes.labelsize'] = 16
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14

    # 1. Load and group files
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
                records.append({"Group": group_label,"Day": day,"Smoothness": smoothness  })

    df = pd.DataFrame(records)
    

    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    # Transparent colored dots by session
    sns.stripplot(data=df, x="Group", y="Smoothness", hue="Day",dodge=True, jitter=0.25, alpha=0.5, size=8,ax=ax, palette="tab10", edgecolor='gray', linewidth=0.2)

    # Standard boxplot (Q1–Q3 + median)
    sns.boxplot(data=df, x="Group", y="Smoothness",showcaps=True, showfliers=False,whiskerprops={'linewidth': 1.2},boxprops={'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 1.5},medianprops={'color': 'black', 'linewidth': 2}, width=0.3,ax=ax)

    group_stats = df.groupby("Group")["Smoothness"].agg(["mean", "std", "count"])
    for i, group in enumerate(["First sessions", "Last sessions"]):
        mean_val = group_stats.loc[group, "mean"]
        std_val = group_stats.loc[group, "std"]
        n = group_stats.loc[group, "count"]
        sem = std_val / np.sqrt(n)


    ax.set_ylabel("Learning metric", fontsize=18, labelpad=10)
    ax.set_xlabel("")
    ax.set_xticklabels(["First sessions", "Last sessions"], fontsize=16)
    sns.despine(trim=True)


    plt.tight_layout()
    return plt.gcf()