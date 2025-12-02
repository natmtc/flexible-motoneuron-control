
import os
import math
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt


def extract_day(filename):
    """
Extract the numeric day index from a filename (e.g., 'Day03' → 3).

Args:
    filename (str): Filename containing a substring like 'DayXX'.

Returns:
    int or None: The integer day number, or None if no day pattern is found.
"""

    match = re.search(r"(Day\d{2})", filename)
    return int(match.group(1)[3:]) if match else None 




def extract_trial(filename):
    """
Extract the trial number from a filename.

Args:
    filename (str): Filename containing a pattern like '_<trialNumber>'.

Returns:
    int: Extracted trial number, or 0 if no trial number is found.
"""

    
    match = re.search(r'_(\d+)[^0-9]', filename)  
    return int(match.group(1)) if match else 0  




def create_raster(folder_path, Fsamp, extract_day, extract_trial):
    """
Create a grid of spike raster plots for all trials in a folder.

Args:
    folder_path (str): Path to the folder containing .pkl trial files.
    Fsamp (float): Sampling frequency of the spike trains (Hz).
    extract_day (callable): Function mapping filename -> sortable day index.
    extract_trial (callable): Function mapping filename -> trial number.

Process:
    - Sorts .pkl files by day and trial.
    - Loads MU spike trains and highlighted MU pair indices from each file.
    - Plots spike rasters (time vs neuron) in a multi-panel figure.
      * First neuron in the pair is red, second is blue, others are black.
    - Removes any unused subplot axes.

Returns:
    matplotlib.figure.Figure: Figure containing all raster subplots.
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




