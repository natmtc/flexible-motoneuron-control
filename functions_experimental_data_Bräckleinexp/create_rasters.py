
import os
import math
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt


def extract_day(filename):
    match = re.search(r"(Day\d{2})", filename)
    return int(match.group(1)[3:]) if match else None  # Extract numeric part of "DayXX"

def extract_trial(filename):
    match = re.search(r'_(\d+)[^0-9]', filename)  # Extracts numeric trial number before a non-digit
    return int(match.group(1)) if match else 0  # Default to 0 if no match found



def create_raster(folder_path, Fsamp, extract_day, extract_trial):
    """
    Create a matplotlib figure with spike rasters for all trial files in the folder.
    
    Returns:
        matplotlib.figure.Figure
    """
    pkl_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.pkl')],
        key=lambda x: (extract_day(x), extract_trial(x))
    )

    num_files = len(pkl_files)
    cols = 3
    rows = math.ceil(num_files / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), sharex=False, sharey=True)
    axes = axes.flatten()

    for idx, file in enumerate(pkl_files):
        file_path = os.path.join(folder_path, file)

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        data_MU = np.array(data['MU'].tolist())
        idx_pair = data['Idx_MU_Pair'].tolist()

        ax = axes[idx]

        for i in range(data_MU.shape[0]):
            spike_times = np.where(data_MU[i] == 1)[0] / Fsamp

            if i == idx_pair[0]:
                colorline = 'red'
            elif i == idx_pair[1]:
                colorline = 'blue'
            else:
                colorline = 'black'

            ax.vlines(spike_times, i + 0.5, i + 1.5, color=colorline, linewidth=0.5)

        ax.set_title(f'{file}', fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neurons')

    # Remove unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig




