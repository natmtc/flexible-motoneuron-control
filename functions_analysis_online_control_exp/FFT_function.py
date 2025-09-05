import os, re, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def extract_day(filename):
    m = re.search(r"(Day\d{2})", filename)
    return m.group(1) if m else None

def extract_trial(filename):
    m = re.search(r'_(\d+)[^0-9]', filename)
    return int(m.group(1)) if m else 0

def load_force(path):
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
    mask = freqs >= fmin
    if not np.any(mask):
        return 0.0
    return np.trapz(spec[mask], freqs[mask])  # V^2

def plot_fft_and_compute_metric_per_trial(directory, fsamp=2048, fmin=5.0, max_freq=30):

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