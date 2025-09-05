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

    fsamp = 2048
    pkl_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]

    # Group by day, filtering broken/incomplete files
    files_by_day = {}
    for file in pkl_files:
        try:
            with open(os.path.join(directory_path, file), 'rb') as f:
                data = pickle.load(f)
            if 'MU' in data and 'Idx_MU_Pair' in data:
                day = extract_day(file)
                if day:
                    files_by_day.setdefault(day, []).append(file)
        except Exception:
            continue

    session_days = sorted(files_by_day.keys())
    if len(session_days) < 4:
        print("Not enough valid days with data.")
        return

    # Sort trials and define early/late
    for day in session_days:
        files_by_day[day].sort(key=extract_trial)

    early_days = session_days[:2]
    late_days = session_days[-2:]

    early_files = [f for d in early_days for f in files_by_day[d]]
    late_files = [f for d in late_days for f in files_by_day[d]]
    grouped_files = early_files + late_files

    num_files = len(grouped_files)
    rows_needed = max(math.ceil(len(early_files) / 2), math.ceil(len(late_files) / 2))

    fig, axes = plt.subplots(rows_needed, 4, figsize=(16, rows_needed * 3))
    axes = np.atleast_2d(axes)

    xlim = (-2, 35)
    ylim = (-2, 35)

    color_map = {day: cm.get_cmap("tab20")(i) for i, day in enumerate(session_days)}

    for i, file_name in enumerate(grouped_files):
        is_early = i < len(early_files)
        local_idx = i if is_early else i - len(early_files)
        row_idx = local_idx // 2
        col_offset = 0 if is_early else 2
        col_idx = local_idx % 2 + col_offset

        if row_idx >= rows_needed or col_idx >= 4:
            continue

        ax = axes[row_idx, col_idx]

        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        data_MU = np.array(data['MU'].tolist())
        neuron_1, neuron_2 = data['Idx_MU_Pair']
        s1, s2 = conv_spikes(data_MU, neuron_1, neuron_2, fs=data['fsamp'])

        s1 = np.clip(s1, 0, clip_thresh)
        s2 = np.clip(s2, 0, clip_thresh)

        ax.plot(s1, s2, color="black", linewidth=2)

        day_info = extract_day(file_name)
        day_color = color_map.get(day_info, "gray")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.add_patch(plt.Rectangle((xlim[0], ylim[0]),xlim[1] - xlim[0],ylim[1] - ylim[0], edgecolor=day_color, facecolor='none',lw=4, zorder=-10))

        ax.set_xticks([5, 15, 25, 35])
        ax.set_yticks([5, 15, 25, 35])
        ax.tick_params(axis='x', labelsize=25)
        ax.tick_params(axis='y', labelsize=25)

        for spine in ax.spines.values():
            spine.set_visible(False)

        if col_idx == 0 or col_idx == 2:
            ax.set_ylabel("FR (Hz)", fontsize=25)
        else:
            ax.set_yticklabels([])

        if row_idx == rows_needed - 1:
            ax.set_xlabel("FR (Hz)", fontsize=25)

        else:
            ax.set_xticklabels([])
       
    # Hide any unused axes
    total_slots = rows_needed * 4
    for j in range(len(grouped_files), total_slots):
        axes[j // 4, j % 4].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
