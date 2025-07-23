import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import re
import pickle

def extract_day(filename):
    match = re.search(r"(Day\d{2})", filename)
    return match.group(1) if match else None

def extract_trial(filename):
    match = re.search(r'_(\d+)[^0-9]', filename)
    return int(match.group(1)) if match else 0

def plot_prolonged_firing_duration(directory_path, fsamp=2048,scatter_save_path=None, trend_save_path=None):
    
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

    num_days = len(session_days)
    if num_days <= 20:
        color_map = {day: plt.cm.get_cmap("tab20")(i) for i, day in enumerate(session_days)}
    else:
        color_map = {day: plt.cm.viridis(i / num_days) for i, day in enumerate(session_days)}

    total_dur = []
    colors = []
    day_durations = {day: [] for day in session_days}

    for file_name in sorted_pkl_files:
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        data_MU = np.array(data['MU'].tolist())
        idx_pair = data['Idx_MU_Pair'].tolist()

        spike_small = np.where(data_MU[idx_pair[0]] == 1)[0]
        spike_large = np.where(data_MU[idx_pair[1]] == 1)[0]

        if len(spike_small) == 0 or len(spike_large) == 0:
            continue

        duration = (spike_large[-1] - spike_small[-1]) / fsamp
        total_dur.append(duration)

        day = extract_day(file_name)
        colors.append(color_map[day])
        day_durations[day].append(duration)

    day_avg_durations = {day: np.mean(durations)for day, durations in day_durations.items() if durations}

    time = np.arange(len(total_dur))
    plt.figure(figsize=(10, 6))
    plt.scatter(time, total_dur, color=colors, alpha=0.8)
    plt.xlabel("File Index")
    plt.ylabel("Prolongated Firing Duration (s)")
    plt.title("Prolongated Firing Duration Across Sessions")

    for day, color in color_map.items():
        plt.scatter([], [], color=color, label=day)

    plt.legend(title="Session Day", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if scatter_save_path:
        plt.savefig(scatter_save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Step 7: Bar plot with smoothed trend
    sorted_days = sorted(day_avg_durations.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
    sorted_avg = [day_avg_durations[day] for day in sorted_days]
    smoothed_avg = gaussian_filter1d(sorted_avg, sigma=1.2)

    plt.figure(figsize=(12, 6))
    plt.bar(sorted_days, sorted_avg, color=[color_map[day] for day in sorted_days], alpha=0.6, label="Avg Duration")
    plt.plot(sorted_days, smoothed_avg, marker='o', linestyle='-', color='black', linewidth=2, label="Trend")

    plt.xlabel("Session Day")
    plt.ylabel("Average Prolongated Firing Duration (s)")
    plt.title("Average Prolongated Firing Duration Per Session (with Trend)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    if trend_save_path:
        plt.savefig(trend_save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_groupwise_prolonged_firing(session_files_dict, base_data_dir, fsamp=2048,save_path=None):
    """
    Plot average prolongated firing durations with smoothed trend across multiple subjects.

    Parameters:
    - session_files_dict: dict of dicts, e.g. {'S1': {'Day01': [...], 'Day02': [...]}, ...}
    - base_data_dir: str, path to the directory containing subject folders
    - fsamp: int, sampling frequency
    - save_path: str or None, if given, figure is saved there
    """
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()

    for idx, (subject, subject_sessions) in enumerate(session_files_dict.items()):
        directory_path = os.path.join(base_data_dir, subject)
        day_durations = {day: [] for day in subject_sessions.keys()}

        for day, files in subject_sessions.items():
            for file_name in files:
                file_path = os.path.join(directory_path, file_name)
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

                data_MU = np.array(data['MU'].tolist())
                idx_pair = data['Idx_MU_Pair']

                spikes_small = np.where(data_MU[idx_pair[0]] == 1)[0]
                spikes_large = np.where(data_MU[idx_pair[1]] == 1)[0]

                if len(spikes_small) == 0 or len(spikes_large) == 0:
                    continue

                dur = (spikes_large[-1] - spikes_small[-1]) / fsamp
                day_durations[day].append(dur)

        # Compute averages
        day_avg_durations = {day: np.mean(durs) for day, durs in day_durations.items() if durs}
        sorted_days = sorted(day_avg_durations.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
        sorted_avg_durations = [day_avg_durations[day] for day in sorted_days]
        smoothed_avg_durations = gaussian_filter1d(sorted_avg_durations, sigma=1.2)

        # Plotting
        ax = axes[idx]
        ax.bar(range(len(sorted_days)), sorted_avg_durations, color='#8FBC8F', alpha=0.8)
        ax.plot(range(len(sorted_days)), smoothed_avg_durations, marker='o',linestyle='-', color='black', linewidth=2)

        ax.set_title(f"Subject {subject[1]}", fontsize=20)
        day_numbers = [int(day[3:]) for day in sorted_days]
        ax.set_xticks(range(len(sorted_days)))
        ax.set_xticklabels(day_numbers, fontsize=25, rotation=45)
        ax.tick_params(axis='both', which='major', width=1.5, length=6, labelsize=25)

        y_ticks = ax.get_yticks()
        y_ticks_int = [int(t) for t in y_ticks if np.isclose(t, round(t))]
        ax.set_yticks(y_ticks_int)
        ax.set_yticklabels([str(t) for t in y_ticks_int], fontsize=25)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(1.5)

        ax.set_xlabel("")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
