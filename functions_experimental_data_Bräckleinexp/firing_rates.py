import os
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, windows


def compute_firing_rates(spike_array, fs, window_duration=0.3):
    window_samples = int(round(window_duration * fs))  # Convert duration to samples
    hanning_window = windows.hann(window_samples)  # Create Hanning window
    hanning_window /= np.sum(hanning_window)  # Normalize window
    
    smoothed_firing_rates = np.zeros_like(spike_array, dtype=float)
    
    for i in range(spike_array.shape[0]):  
        smoothed_firing_rates[i, :] = convolve(spike_array[i, :], hanning_window, mode='same') * fs  
    
    return smoothed_firing_rates

def plot_firing_rates(folder_path, pkl_files, fsamp=2048, window_duration=0.3, session_label=''):
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

        ax = axes[idx]
        time_axis = np.arange(data_MU.shape[1]) / fsamp  # Convert sample index to time in seconds

        for neuron_idx in range(data_MU.shape[0]):
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