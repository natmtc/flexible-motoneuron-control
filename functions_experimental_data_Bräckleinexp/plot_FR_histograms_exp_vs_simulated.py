import os
import re
import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt


def extract_day(filename):
    match = re.search(r"(Day\d{2})", filename)
    return int(match.group(1)[3:]) if match else None


def extract_trial(filename):
    match = re.search(r'_(\d+)[^0-9]', filename)
    return int(match.group(1)) if match else 0


def compute_avg_firing_rate(spike_times, sampling_rate):
    if len(spike_times) < 2:
        return None
    ISIs = np.diff(spike_times) / sampling_rate
    return 1 / np.mean(ISIs)


def load_simulated_rates(jld2_file_paths):
    """
    Load avg_rates_per_neuron from multiple .jld2 files using h5py.
    Assumes the files are named ...forceX_firing_rates6.jld2 and X corresponds to DayXX.
    Returns a dict: day -> list of firing rates.
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
