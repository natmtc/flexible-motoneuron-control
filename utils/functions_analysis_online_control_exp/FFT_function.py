import os, re, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def extract_day(filename):
    """
Extract the day label (e.g., 'Day03') from a filename.

Args:
    filename (str): Filename containing a substring like 'DayXX'.

Returns:
    str or None: The matched day string, or None if no day pattern is found.
"""

    m = re.search(r"(Day\d{2})", filename)
    return m.group(1) if m else None




def extract_trial(filename):
    """
Extract the trial number from a filename.

Args:
    filename (str): Filename containing a substring like '_<trialNumber>'.

Returns:
    int: Extracted trial number, or 0 if no match is found.
"""

    m = re.search(r'_(\d+)[^0-9]', filename)
    return int(m.group(1)) if m else 0




def load_force(path):
    """
Load and clean force data from a pickle file.

Args:
    path (str): Path to a pickle file containing a dict with key 'Force'.

Process:
    - Loads the pickle dictionary.
    - Extracts the 'Force' entry, handling cases where it is nested or list-like.
    - Squeezes to 1D and converts to float.
    - Replaces NaN and ±inf values with 0 for robustness.

Returns:
    np.ndarray: 1D force signal as floats.
"""

    with open(path, 'rb') as f:
        d = pickle.load(f)
    try:
        x = np.array(d['Force']).squeeze()
        if x.ndim > 1:
            x = np.array(d['Force'].tolist())[0].squeeze()
    except Exception:
        x = np.array(d['Force'].tolist())[0].squeeze()
    return np.nan_to_num(np.asarray(x, float), nan=0.0, posinf=0.0, neginf=0.0)

def spectrum_psd(x, fs, nperseg_secs=2.0, window='hann', detrend=False):
    nperseg = max(8, int(round(nperseg_secs * fs)))
    nperseg = min(nperseg, len(x))
    freqs, psd = welch(x, fs=fs, window=window, nperseg=nperseg,
                       noverlap=nperseg//2,
                       detrend='constant' if detrend else False,
                       return_onesided=True, scaling='density')  # V^2/Hz
    return freqs, psd




def area_above(freqs, spec, fmin):
    """
Compute the spectral area above a given frequency threshold.

Args:
    freqs (np.ndarray): Frequency vector (Hz).
    spec (np.ndarray): Spectral amplitude or power values corresponding to `freqs`.
    fmin (float): Minimum frequency (Hz) above which the area is integrated.

Process:
    - Selects all frequencies ≥ fmin.
    - Computes the integral of the spectrum over this region using trapezoidal rule.

Returns:
    float: Integrated spectral area (e.g., V² or arbitrary units).
"""

    mask = freqs >= fmin
    if not np.any(mask):
        return 0.0
    return np.trapz(spec[mask], freqs[mask])  # V^2




def plot_fft_and_compute_metric_per_trial(directory, fsamp=2048, fmin=5.0, max_freq=30):
    """
Compute and plot PSD-based high-frequency metrics for each trial, grouped by day.

Args:
    directory (str): Folder containing .pkl files with force data.
    fsamp (float): Sampling frequency of the force signal (Hz).
    fmin (float): Frequency threshold above which power is considered high-frequency.
    max_freq (float or None): Maximum frequency to display in the PSD plots
        (if None, plot full spectrum).

Process:
    - Groups .pkl files by day and sorts them by trial number.
    - For each trial:
        * Loads the force trace.
        * Computes PSD via `spectrum_psd`.
        * Computes total power and power above `fmin`.
        * Computes the fraction of high-frequency power.
        * Plots PSD curves per day (optionally truncated to `max_freq`).
    - Draws a vertical line at `fmin` in each plot.

Returns:
    list of dict: One entry per trial with keys:
        'filename', 'day', 'trial',
        'hf_power_V2', 'total_power_V2', 'fraction_hf'.
"""

    # Group files by day and sort by trial
    pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    by_day = {}
    for f in pkl_files:
        day = extract_day(f)
        if day:
            by_day.setdefault(day, []).append(f)
    for day in by_day:
        by_day[day].sort(key=extract_trial)

    # Collect results per trial and plot per day
    results = []
    for day in sorted(by_day.keys()):
        plt.figure(figsize=(8, 5))
        plt.title(f"{day} — PSD per trial")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (V²/Hz)')
        plt.yscale('log')

        first_freqs = None  # to reuse x-limits
        for fname in by_day[day]:
            path = os.path.join(directory, fname)
            x = load_force(path)

            freqs, psd = spectrum_psd(x, fsamp, nperseg_secs=2.0, window='hann', detrend=False)
            total_power = np.trapz(psd, freqs)
            hf_power = area_above(freqs, psd, fmin)
            frac_hf = hf_power / total_power if total_power > 0 else 0.0

            # Plot this trial's PSD
            if max_freq is not None:
                keep = freqs <= max_freq
                f_plot, psd_plot = freqs[keep], psd[keep]
            else:
                f_plot, psd_plot = freqs, psd

            trial_num = extract_trial(fname)
            plt.plot(f_plot, psd_plot, linewidth=1.2, alpha=0.8, label=f"Trial {trial_num}")

            results.append({
                'filename': fname,
                'day': day,
                'trial': trial_num,
                'hf_power_V2': hf_power,
                'total_power_V2': total_power,
                'fraction_hf': frac_hf
            })

            if first_freqs is None:
                first_freqs = f_plot

        # Vertical split line
        if max_freq is None or fmin <= max_freq:
            plt.axvline(5, linestyle='--', linewidth=1)

        plt.legend(title='Trials', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    return results