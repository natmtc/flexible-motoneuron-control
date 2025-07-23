import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from scipy.signal import windows, convolve
import math
    
    
def plot_fr_trajectories(directory_path, output_path, clip_thresh=35):

    def conv_spikes(array, pos1, pos2, fs, window_duration=0.3):
        window_samples = int(round(window_duration * fs))
        hanning_window = windows.hann(window_samples)
        hanning_window /= np.sum(hanning_window)  
        s1 = convolve(array[pos1, :], hanning_window, mode='same') * fs
        s2 = convolve(array[pos2, :], hanning_window, mode='same') * fs
        return s1, s2

    def extract_day(filename):
        match = re.search(r"(Day\d{2})", filename)
        return match.group(1) if match else None  

    def extract_trial(filename):
        match = re.search(r'_(\d+)[^0-9]', filename)  
        return int(match.group(1)) if match else 0  

    # === Load and organize files ===
    fsamp = 2048
    pkl_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]

    files_by_day = {}
    for file in pkl_files:
        day = extract_day(file)
        if day:
            files_by_day.setdefault(day, []).append(file)

    session_days = sorted(files_by_day.keys())

    for day in session_days:
        files_by_day[day].sort(key=extract_trial)

    early_days = session_days[:2]
    late_days = session_days[-2:]

    early_files = [f for d in early_days for f in files_by_day[d]]
    late_files = [f for d in late_days for f in files_by_day[d]]
    grouped_files = early_files + late_files

    num_days = len(session_days)
    color_map = {day: cm.get_cmap("tab20")(i) for i, day in enumerate(session_days)}  

    # === Plot ===
    num_files = len(grouped_files)
    cols = 4
    rows = math.ceil(len(grouped_files) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    if rows == 1:
        axes = axes[np.newaxis, :]  # Always 2D

    xlim = (-2, 35)
    ylim = (-2, 35)

    for i, file_name in enumerate(grouped_files):
        is_early = i < len(early_files)
        local_idx = i if is_early else i - len(early_files)
        row_idx = local_idx // 2
        col_idx = local_idx % 2 + (0 if is_early else 2)
        ax = axes[row_idx, col_idx]

        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        data_MU = np.array(data['MU'].tolist())
        neuron_1, neuron_2 = data['Idx_MU_Pair'].tolist()
        s1, s2 = conv_spikes(data_MU, neuron_1, neuron_2, fs=data['fsamp'])

        s1 = np.clip(s1, 0, clip_thresh)
        s2 = np.clip(s2, 0, clip_thresh)

        ax.plot(s1, s2, color="black", linewidth=2)

        day_info = extract_day(file_name)
        day_color = color_map.get(day_info, "gray")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.add_patch(plt.Rectangle((xlim[0], ylim[0]),xlim[1] - xlim[0],ylim[1] - ylim[0],edgecolor=day_color,facecolor='none',lw=4, zorder=-10))

        ax.set_xticks([5, 15, 25, 35])
        ax.set_yticks([5, 15, 25, 35])
        ax.tick_params(axis='x', labelsize=25)
        ax.tick_params(axis='y', labelsize=25)

        for spine in ax.spines.values():
            spine.set_visible(False)

        if col_idx == 0:
            ax.set_ylabel("Neuron 2 FR (Hz)", fontsize=25)
        else:
            ax.set_yticklabels([])

        if row_idx == rows - 1:
            ax.set_xlabel("Neuron 1 FR (Hz)", fontsize=25)
        else:
            ax.set_xticklabels([])

    for j in range(i + 1, rows * cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
