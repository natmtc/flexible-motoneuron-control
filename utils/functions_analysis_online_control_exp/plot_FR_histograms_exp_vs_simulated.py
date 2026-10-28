import os
import re
import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt


def extract_day(filename):
    """
Extract the numeric day index from a filename (e.g., 'Day07' → 7).

Args:
    filename (str): Filename containing a substring like 'DayXX'.

Returns:
    int or None: The day number as an integer, or None if no match is found.
"""

    match = re.search(r"(Day\d{2})", filename)
    return int(match.group(1)[3:]) if match else None




def extract_trial(filename):
    """
Extract the trial number from a filename.

Args:
    filename (str): Filename containing a substring like '_<trialNumber>'.

Returns:
    int: Extracted trial number, or 0 if no valid pattern is found.
"""

    match = re.search(r'_(\d+)[^0-9]', filename)
    return int(match.group(1)) if match else 0



def compute_avg_firing_rate(spike_times, sampling_rate):
    """
Compute the average firing rate from spike times.

Args:
    spike_times (array-like): Indices of spike occurrences (sample numbers).
    sampling_rate (float): Sampling rate in Hz.

Process:
    - Computes inter-spike intervals (ISIs) in seconds.
    - Returns the reciprocal of the mean ISI, i.e., the average firing rate.

Returns:
    float or None: Average firing rate in Hz, or None if fewer than two spikes.
"""

    if len(spike_times) < 2:
        return None
    ISIs = np.diff(spike_times) / sampling_rate
    return 1 / np.mean(ISIs)



def load_simulated_rates(jld2_file_paths):
    """
Load simulated average firing rates from multiple .jld2 files.

Args:
    jld2_file_paths (list of str): Paths to .jld2 files whose names contain a
        pattern like 'forceX_firing_rates.jld2', where X is the session day index.

Process:
    - Extracts the day number from each filename using the pattern 'force(\d+)_firing_rates.jld2'.
    - Opens each file with h5py and reads the dataset 'avg_rates_per_neuron'.
    - Stores the firing-rate array under its corresponding day.

Returns:
    dict: Mapping {day_number (int): avg_rates_per_neuron (np.ndarray)}.
"""

    sim_rates_by_day = {}

    for path in jld2_file_paths:
        day_match = re.search(r'force(\d+)_firing_rates\.jld2', path)
        if not day_match:
            continue
        day = int(day_match.group(1))

        with h5py.File(path, 'r') as f:
            if 'avg_rates_per_neuron' in f:
                rates = f['avg_rates_per_neuron'][:]
                sim_rates_by_day[day] = rates
                print(sim_rates_by_day[day])
            else:
                print(f"'avg_rates_per_neuron' not found in {path}")

    return sim_rates_by_day



def compute_firing_rate_histograms_with_sim(folder_path, Fsamp, jld2_file_paths):
    """
Compare experimental and simulated firing-rate distributions across days.

Args:
    folder_path (str): Directory containing experimental .pkl files with 'MU' data.
    Fsamp (float): Sampling frequency of the spike trains (Hz).
    jld2_file_paths (list of str): Paths to .jld2 files with simulated
        'avg_rates_per_neuron' datasets.

Process:
    - Loads simulated average firing rates per day.
    - Groups experimental .pkl files by day and extracts MU spike trains.
    - Computes average firing rate per neuron (excluding neurons with <2 spikes
      and clipping at 80 Hz).
    - For each day, plots overlaid histograms (density) of experimental vs
      simulated firing rates in a multi-panel figure.

Returns:
    matplotlib.figure.Figure: Figure containing histograms for all days.
"""

    # Load simulated data once
    sim_data = load_simulated_rates(jld2_file_paths)

    # Load and group experimental files
    pkl_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.pkl')],
        key=lambda x: (extract_day(x), extract_trial(x))
    )
    day_files = {}
    for file in pkl_files:
        day = extract_day(file)
        if day is not None:
            day_files.setdefault(day, []).append(file)

    # Prepare subplots
    n_days = len(day_files)
    n_cols = 4
    n_rows = int(np.ceil(n_days / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, (day, files) in enumerate(sorted(day_files.items())):
        all_firing_rates = []

        for file in files:
            with open(os.path.join(folder_path, file), 'rb') as f:
                data = pickle.load(f)
            MU_array = np.array(data['MU'])

            for neuron_idx in range(MU_array.shape[0]):
                spike_times = np.where(MU_array[neuron_idx] == 1)[0]
                fr = compute_avg_firing_rate(spike_times, Fsamp)
                if fr is not None and fr <= 80:
                    all_firing_rates.append(fr)

        ax = axes[i]
        bin_edges = np.linspace(0, 40, 21)  # 20 bins, 4 Hz each

        ax.hist(all_firing_rates, bins=bin_edges, color='blue', alpha=0.4, label='Experimental',density=True)

        # Plot simulated data if it exists for this day
        if day in sim_data:
            ax.hist(sim_data[day], bins=bin_edges, color='red', alpha=0.4, label='Simulated',density=True)

        ax.set_title(f'Day {day}: {len(all_firing_rates)} neurons')
        ax.set_xlabel('Avg firing rate (Hz)')
        ax.set_ylabel('Count')
        ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig
