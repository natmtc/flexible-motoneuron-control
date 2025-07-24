import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import interp1d

def extract_day(filename):
    match = re.search(r"(Day\d{2})", filename)
    return match.group(1) if match else None

def extract_trial(filename):
    match = re.search(r'_(\d+)[^0-9]', filename)
    return int(match.group(1)) if match else 0

def compute_and_save_avg_force(directory_path, save_dir, file_prefix="avg_force", target_max=0.5, fsamp=2048, plot=True):
    """
    Compute average force trajectories per day, scale to target_max, and save to .npy files.

    Parameters:
        directory_path (str): Path to the directory containing .pkl files.
        save_dir (str): Path where scaled .npy files will be saved.
        file_prefix (str): Prefix for saved filenames (default: "avg_force").
        target_max (float): Target maximum force value after scaling (default: 0.5).
        fsamp (int): Sampling rate in Hz (default: 2048).
        plot (bool): Whether to generate force plots (default: True).
    """
    
    pkl_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]
    files_by_day = {}
    for file in pkl_files:
        day = extract_day(file)
        if day is not None:
            files_by_day.setdefault(day, []).append(file)

    session_days = sorted(files_by_day.keys())
    for day in session_days:
        files_by_day[day].sort(key=extract_trial)

    average_force_by_day = {}
    time_by_day = {}

    for day in session_days:
        session_files = files_by_day[day]
        force_trials = []
        trial_lengths = []
        raw_forces = []

        for file_name in session_files:
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            force = np.array(data['Force'].tolist())[0]
            raw_forces.append(force)
            trial_lengths.append(len(force))

        min_length = min(trial_lengths)
        new_time = np.linspace(0, 1, min_length)

        for force in raw_forces:
            old_time = np.linspace(0, 1, len(force))
            interp_func = interp1d(old_time, force, kind='linear')
            downsampled = interp_func(new_time)
            force_trials.append(downsampled)

        force_trials = np.array(force_trials)
        mean_force = np.mean(force_trials, axis=0)
        std_force = np.std(force_trials, axis=0)
        time_axis = np.linspace(0, len(raw_forces[0]) / fsamp, min_length)

        average_force_by_day[day] = {'mean': mean_force, 'std': std_force}
        time_by_day[day] = time_axis

    # Global max for scaling
    all_max = max(np.max(average_force_by_day[day]['mean']) for day in session_days)
    scaling_factor = target_max / all_max

    print(f"Global maximum force (before scaling): {all_max:.4f}")
    print(f"Scaling factor to apply: {scaling_factor:.4f}")

    os.makedirs(save_dir, exist_ok=True)

    for day in session_days:
        mean_force = average_force_by_day[day]['mean']
        scaled_force = mean_force * scaling_factor      
        filename = f"{file_prefix}_{day}_scaled{int(target_max * 10):02d}.npy"
        save_path = os.path.join(save_dir, filename)
        np.save(save_path, scaled_force)
        print(f"Saved scaled force for {day} to {save_path}")

    if plot:
        cols = 3
        rows = int(np.ceil(len(session_days) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=False, sharey=True)
        axes = axes.flatten()
        cmap = cm.get_cmap("tab10")

        for i, day in enumerate(session_days):
            mean = average_force_by_day[day]['mean']
            std = average_force_by_day[day]['std']
            t = time_by_day[day]
            color = cmap(i % 10)

            ax = axes[i]
            ax.plot(t, mean, color=color, label='Mean Force')
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.3, label='±1 SD')
            ax.set_title(f"{day}", fontsize=12)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Force")
            ax.set_ylim([0, 0.2])
            ax.grid(True)
            ax.legend(fontsize=8)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
