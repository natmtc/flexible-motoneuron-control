import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import re
from scipy import signal
from matplotlib.ticker import MaxNLocator


def extract_day(filename):
    match = re.search(r"(Day\d{2})", filename)
    return match.group(1) if match else None  

# Extract trial number from filename
def extract_trial(filename):
    match = re.search(r'_(\d+)[^0-9]', filename)  
    return int(match.group(1)) if match else 0  

# Define Low-pass filter (Butterworth)
def lowpass_filter(data, cutoff=0.5, fs=2048, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)



def create_peak_count_summary(directory_paths, fsamp, extract_day, extract_trial, lowpass_filter):
    """
    Create a figure summarising the average number of peaks per session across multiple folders.

    Parameters:
        directory_paths (dict): Dictionary mapping folder names to their paths.
        fsamp (float): Sampling frequency.
        extract_day (function): Function to extract day identifier from filename.
        extract_trial (function): Function to extract trial number from filename.
        lowpass_filter (function): Function to low-pass filter the force signal.

    Returns:
        matplotlib.figure.Figure: The assembled figure (not saved or shown).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Can adjust layout for number of folders
    axes = axes.flatten()

    for idx, (folder_name, directory_path) in enumerate(directory_paths.items()):
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
        peak_counts_by_day = {day: [] for day in session_days}

        for file_name in sorted_pkl_files:
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            data_FORCE = np.array(data['Force'].tolist())[0]
            filtered_force = lowpass_filter(data_FORCE)
            peaks, _ = find_peaks(filtered_force)
            num_peaks = len(peaks)

            day = extract_day(file_name)
            if day:
                peak_counts_by_day[day].append(num_peaks)

        avg_peaks_per_day = [np.mean(peak_counts_by_day[day]) for day in session_days]
        std_peaks_per_day = [np.std(peak_counts_by_day[day]) for day in session_days]

        ax = axes[idx]
        bars = ax.bar(
            session_days,
            avg_peaks_per_day,
            yerr=std_peaks_per_day,
            color='gray',
            edgecolor='black',
            alpha=0.7,
            capsize=5
        )

        ax.set_xlabel("Session (Day)", fontsize=25)
        ax.set_ylabel("Average Number of Peaks", fontsize=25)
        ax.set_title(f"({folder_name})", fontsize=25)
        ax.set_ylim([0, 13])
        ax.set_xticks(np.arange(len(session_days)))
        ax.set_xticklabels([str(int(day[3:])) for day in session_days])
        ax.tick_params(axis='x', labelsize=20, which='both', length=0)
        ax.tick_params(axis='y', labelsize=20, which='both', length=0)

    # Remove unused axes if fewer folders than subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig
