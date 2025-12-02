import os
import re
import pickle
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def extract_day(filename):
    match = re.search(r"(Day\d{2})", filename)
    return match.group(1) if match else None

def extract_trial(filename):
    match = re.search(r'_(\d+)[^0-9]', filename)
    return int(match.group(1)) if match else 0

def compute_representative_trial(directory_path, save_dir, file_prefix="represetative_trial", target_max=0.4, fsamp=2048, plot=True,trial_numbers_by_day=None):

    pkl_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]
    files_by_day = {}
    for file in pkl_files:
        day = extract_day(file)
        if day is not None:
            files_by_day.setdefault(day, []).append(file)

    session_days = sorted(files_by_day.keys())
    for day in session_days:
        files_by_day[day].sort(key=extract_trial)

    # if list is given, align it to session_days
    day_to_trialnum = None
    if isinstance(trial_numbers_by_day, list):
        if len(trial_numbers_by_day) != len(session_days):
            raise ValueError(
                f"Length of trial_numbers_by_day list ({len(trial_numbers_by_day)}) "
                f"must match number of days ({len(session_days)})."
            )
        day_to_trialnum = {day: tn for day, tn in zip(session_days, trial_numbers_by_day)}
    elif isinstance(trial_numbers_by_day, dict):
        day_to_trialnum = trial_numbers_by_day
    else:
        raise ValueError("Provide trial_numbers_by_day (list or dict) to select a specific trial per day.")

    average_force_by_day = {}
    time_by_day = {}

    for day in session_days:
        session_files = files_by_day[day]

        # Load all trial forces for the day
        raw_forces, trial_lengths, trial_nums = [], [], []
        for file_name in session_files:
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            force = np.array(data['Force'].tolist())[0]
            raw_forces.append(force)
            trial_lengths.append(len(force))
            trial_nums.append(extract_trial(file_name))

        # Use the specific trial only (no averaging)
        selected_trialnum = int(day_to_trialnum[day])
        candidates = [i for i, tn in enumerate(trial_nums) if tn == selected_trialnum]
        if not candidates:
            raise ValueError(f"Day {day}: trial number {selected_trialnum} not found in filenames {trial_nums}.")
        idx = candidates[0]
        selected_force = raw_forces[idx]

        min_length = len(selected_force)
        new_time = np.linspace(0, 1, min_length)
        old_time = np.linspace(0, 1, len(selected_force))
        interp_func = interp1d(old_time, selected_force, kind='linear')
        downsampled = interp_func(new_time)

        mean_force = downsampled  # using the selected trial only
        std_force = np.zeros_like(mean_force)  # no variability when not averaging
        time_axis = np.linspace(0, len(selected_force) / fsamp, min_length)

        average_force_by_day[day] = {
            'mean': mean_force,
            'std': std_force,
            'selected_trialnum': selected_trialnum
        }
        time_by_day[day] = time_axis

    all_max = max(np.max(average_force_by_day[day]['mean']) for day in session_days)
    scaling_factor = target_max / all_max if all_max > 0 else 1.0
    print(f"Global maximum force (before scaling): {all_max:.4f}")
    print(f"Scaling factor to apply: {scaling_factor:.4f}")

    os.makedirs(save_dir, exist_ok=True)

    for day in session_days:
        mean_force = average_force_by_day[day]['mean']
        scaled_force = mean_force * scaling_factor

        sel_trial = average_force_by_day[day]['selected_trialnum']
        suffix = f"_trial{sel_trial:02d}"

        filename = f"{file_prefix}_{day}.npy"
        save_path = os.path.join(save_dir, filename)
        # np.save(save_path , scaled_force)
        print(f"Saved scaled force for {day} to {save_path}")

    if plot:
        cols = 3
        rows = int(np.ceil(len(session_days) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=False, sharey=True)
        axes = axes.flatten()
        cmap = cm.get_cmap("tab10")

        for i, day in enumerate(session_days):
            mean = average_force_by_day[day]['mean'] * scaling_factor
            t    = time_by_day[day]
            color = cmap(i % 10)
            sel_trial = average_force_by_day[day]['selected_trialnum']

            ax = axes[i]
            ax.plot(t, mean, color=color, label='Scaled Force')
            title_tag = f"{day} (trial {sel_trial})"
            ax.set_title(title_tag, fontsize=12)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Scaled Force")
            ax.set_ylim([0, target_max * 1.1])
            ax.grid(True)
            ax.legend(fontsize=8)

        # remove unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
        
        