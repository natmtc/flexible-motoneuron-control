import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def extract_day(filename):
    match = re.search(r"(Day\d{2})", filename)
    return match.group(1) if match else None  

def extract_trial(filename):
    match = re.search(r'_(\d+)[^0-9]', filename)  
    return int(match.group(1)) if match else 0  

def create_force_plots_by_day(directory_path, fsamp, highlight_samples, extract_day, extract_trial):
    """
    Create a multi-panel force plot figure with trials grouped and colour-coded by day.

    Parameters:
        directory_path (str): Path to the directory containing .pkl files.
        fsamp (float): Sampling frequency (Hz).
        highlight_samples (int): Number of samples to highlight at start and end.
        extract_day (function): Function to extract day from filename.
        extract_trial (function): Function to extract trial number from filename.

    Returns:
        matplotlib.figure.Figure: The generated figure (not shown or saved).
    """

    # Group files by day
    pkl_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]
    files_by_day = {}
    for file in pkl_files:
        day = extract_day(file)
        if day is not None:
            files_by_day.setdefault(day, []).append(file)

    # Sort days and files within each day
    session_days = sorted(files_by_day.keys())
    for day in session_days:
        files_by_day[day].sort(key=extract_trial)
    
    # Flatten the sorted list
    sorted_pkl_files = [file for day in session_days for file in files_by_day[day]]

    # Colour map per day
    num_days = len(session_days)
    if num_days <= 20:
        cmap = cm.get_cmap("tab20")
        color_map = {day: cmap(i) for i, day in enumerate(session_days)}
    else:
        cmap = cm.get_cmap("viridis")
        color_map = {day: cmap(i / num_days) for i, day in enumerate(session_days)}

    # Layout
    num_files = len(sorted_pkl_files)
    cols = 6
    rows = int(np.ceil(num_files / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3), sharex=False, sharey=False)
    axes = axes.flatten()

    for i, file_name in enumerate(sorted_pkl_files):
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        data_FORCE = np.array(data['Force'].tolist())[0]
        time_axis = np.arange(len(data_FORCE)) / fsamp
        start_region = np.arange(highlight_samples)
        end_region = np.arange(len(data_FORCE) - highlight_samples, len(data_FORCE))

        day_info = extract_day(file_name)
        day_color = color_map.get(day_info, "white")

        ax = axes[i]
        ax.plot(time_axis, data_FORCE, color="blue", alpha=0.8)
        ax.axvspan(time_axis[0], time_axis[-1], color=day_color, alpha=0.3, zorder=-10)
        ax.plot(time_axis[start_region], data_FORCE[start_region], color="green", alpha=1)
        ax.plot(time_axis[end_region], data_FORCE[end_region], color="red", alpha=1)

        ax.set_title(file_name.split('.')[0], fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force")

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig


def plot_force_trials(directory_path, output_path, fsamp=2048, highlight_sec=3):


    highlight_samples = int(highlight_sec * fsamp)

    def extract_day(filename):
        match = re.search(r"(Day\d{2})", filename)
        return match.group(1) if match else None

    def extract_trial(filename):
        match = re.search(r'_(\d+)[^0-9]', filename)
        return int(match.group(1)) if match else 0

    # Get all .pkl files in directory
    pkl_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]

    # Group files by day
    files_by_day = {}
    for file in pkl_files:
        day = extract_day(file)
        if day:
            files_by_day.setdefault(day, []).append(file)

    session_days = sorted(files_by_day.keys())
    selected_sessions = session_days[:2] + session_days[-2:]
    sorted_pkl_files = []
    for day in selected_sessions:
        sorted_pkl_files.extend(files_by_day[day])

    # Color map
    num_days = len(session_days)
    if num_days <= 20:
        color_map = {day: cm.get_cmap("tab20")(i) for i, day in enumerate(session_days)}
    else:
        color_map = {day: cm.viridis(i / num_days) for i, day in enumerate(session_days)}

    # Group early/late trials
    early_days = session_days[:2]
    late_days = session_days[-2:]
    early_trials = [f for f in sorted_pkl_files if extract_day(f) in early_days]
    late_trials = [f for f in sorted_pkl_files if extract_day(f) in late_days]
    grouped_files = early_trials + late_trials

    # Layout
    num_files = len(grouped_files)
    cols = 4
    rows = int(np.ceil(num_files / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    if rows == 1:
        axes = axes[np.newaxis, :]

    # Global y-limits
    all_forces = []
    for file_name in grouped_files:
        with open(os.path.join(directory_path, file_name), 'rb') as f:
            data = pickle.load(f)
        all_forces.extend(np.array(data['Force'].tolist())[0])
    all_forces = np.array(all_forces)
    ymin, ymax = np.min(all_forces), np.max(all_forces)

    # Plot
    for i, file_name in enumerate(grouped_files):
        col_offset = 0 if i < len(early_trials) else 2
        local_idx = i if i < len(early_trials) else i - len(early_trials)
        row_idx = local_idx // 2
        col_idx = local_idx % 2 + col_offset
        ax = axes[row_idx, col_idx]

        with open(os.path.join(directory_path, file_name), 'rb') as f:
            data = pickle.load(f)
        data_FORCE = np.array(data['Force'].tolist())[0]
        time_axis = np.arange(len(data_FORCE)) / fsamp
        start_region = np.arange(highlight_samples)
        end_region = np.arange(len(data_FORCE) - highlight_samples, len(data_FORCE))
        day_info = extract_day(file_name)
        day_color = color_map.get(day_info, "white")

        ax.plot(time_axis, data_FORCE, color="black", alpha=0.8)
        ax.add_patch(plt.Rectangle((0, ymin), time_axis[-1], ymax - ymin, color='white',
                                   ec=day_color, lw=4, fill=False, zorder=-30))
        ax.plot(time_axis[start_region], data_FORCE[start_region], color="black", alpha=1)
        ax.plot(time_axis[end_region], data_FORCE[end_region], color="black", alpha=1)

        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis='y', which='both', length=0, labelsize=0)
        ax.tick_params(axis='x', which='both', length=0, labelsize=25)
        for spine in ax.spines.values():
            spine.set_visible(False)

        if col_idx == 0:
            ax.set_ylabel("Force", fontsize=25)
            ax.tick_params(axis='y', length=5, labelsize=25)

        if row_idx == rows - 1:
            ax.set_xlabel("Time (s)", fontsize=25)
            ax.tick_params(axis='x', length=5, labelsize=25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
