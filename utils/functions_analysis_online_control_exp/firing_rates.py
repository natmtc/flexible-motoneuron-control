import os
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, windows


def compute_firing_rates(spike_array, fs, window_duration=0.3):
    """
Compute smoothed firing rates for each neuron using a Hanning window.

Args:
    spike_array (np.ndarray): Binary spike matrix of shape (neurons, samples).
    fs (float): Sampling frequency in Hz.
    window_duration (float): Duration of the smoothing window in seconds.

Process:
    - Converts window duration to samples.
    - Builds a normalized Hanning window.
    - Convolves each spike train with the window.
    - Multiplies by fs to convert spikes/sample → spikes/second (Hz).

Returns:
    np.ndarray: Smoothed firing-rate matrix with same shape as spike_array.
"""

    window_samples = int(round(window_duration * fs))  # Convert duration to samples
    hanning_window = windows.hann(window_samples)  # Create Hanning window
    hanning_window /= np.sum(hanning_window)  # Normalize window
    
    smoothed_firing_rates = np.zeros_like(spike_array, dtype=float)
    
    for i in range(spike_array.shape[0]):  
        smoothed_firing_rates[i, :] = convolve(spike_array[i, :], hanning_window, mode='same') * fs  
    
    return smoothed_firing_rates




def plot_firing_rates(folder_path, pkl_files, fsamp=2048, window_duration=0.3, session_label=''):
    """
Plot smoothed firing rates for selected trials, highlighting the larger-threshold MU.

Args:
    folder_path (str): Directory containing the .pkl trial files.
    pkl_files (list of str): Filenames to plot (already sorted externally).
    fsamp (float): Sampling frequency of the MU data (Hz).
    window_duration (float): Smoothing window duration for firing rate estimation (s).
    session_label (str): Optional text prefix for subplot titles.

Process:
    - Loads MU spike trains from each file.
    - Computes smoothed firing rates using a Hanning window.
    - Extracts the MU pair indices and plots only the larger-threshold neuron.
    - Arranges plots in a multi-panel grid with shared axes.

Returns:
    None (the function generates and displays a matplotlib figure).
"""

    num_files = len(pkl_files)      
    cols = 3  # Number of columns for subplots
    rows = math.ceil(num_files / cols)  # Compute number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, file in enumerate(pkl_files):
        file_path = os.path.join(folder_path, file)

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        data_MU = np.array(data['MU'].tolist())  
        smoothed_firing_rates = compute_firing_rates(data_MU, fsamp, window_duration)
        idx_pair = data['Idx_MU_Pair'].tolist()

        ax = axes[idx]
        time_axis = np.arange(data_MU.shape[1]) / fsamp  # Convert sample index to time in seconds

        for neuron_idx in range(data_MU.shape[0]):
            if neuron_idx==idx_pair[1]:
                ax.plot(time_axis, smoothed_firing_rates[neuron_idx, :], label=f'Neuron {neuron_idx}', alpha=0.7)

        num_neurons = data_MU.shape[0]  # Count the number of motoneurons
        ax.set_title(f'{session_label} - Trial: {file}', fontsize=10)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing Rate (Hz)')

    # Remove unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])  

    plt.tight_layout()
    plt.ylim([0, 40])
    plt.show()