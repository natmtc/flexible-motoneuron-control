import os
import re
import h5py
import numpy as np
from scipy.special import gammaln


def compute_pR2(y_true, y_pred, eps: float = 1e-12) -> float:
    """
    Poisson deviance–based pseudo-R² (Perich-style).

    pR² = 1 − (LL_sat − LL_model) / (LL_sat − LL_null)

    Parameters
    ----------
    y_true : array-like
        Observed spike counts per bin.
    y_pred : array-like
        Predicted mean counts per bin (nonnegative).
    eps : float, optional
        Small clamp to avoid log(0); default 1e-12.

    Returns
    -------
    float
        pR² in (-∞, 1]. Returns NaN if the denominator is zero/invalid.
    """
    y = np.asarray(y_true, dtype=float)
    lam = np.clip(np.asarray(y_pred, dtype=float), eps, None)

    def logL(y_, lam_):
        return np.sum(y_ * np.log(lam_) - lam_ - gammaln(y_ + 1))

    nz = y > 0
    ll_sat   = np.sum(y[nz] * np.log(y[nz])) - np.sum(y) - np.sum(gammaln(y + 1))
    ll_model = logL(y, lam)
    ll_null  = logL(y, np.full_like(y, y.mean()))
    denom = ll_sat - ll_null
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return 1.0 - (ll_sat - ll_model) / denom


def list_trials_spike(spike_base_dir: str) -> dict:
    """
    Discover spike files arranged as: spike_base_dir/DayXX/*.jld2.

    Returns a nested mapping per day and trial key (e.g., 'Day03_trial14').

    Parameters
    ----------
    spike_base_dir : str
        Root directory containing DayXX subfolders with .jld2 files.

    Returns
    -------
    dict
        { 'Day01': { 'Day01_trial01': '/.../Day01/....jld2', ... }, ... }
    """
    out = {}
    for d in sorted(os.listdir(spike_base_dir)):
        if not d.lower().startswith('day'):
            continue
        day_dir = os.path.join(spike_base_dir, d)
        if not os.path.isdir(day_dir):
            continue
        per_day = {}
        for f in sorted(os.listdir(day_dir)):
            if not f.endswith('.jld2'):
                continue
            stem = os.path.splitext(f)[0]
            m = re.search(r'(Day\d+_trial\d+)', stem, re.IGNORECASE)
            key = m.group(1) if m else stem
            per_day[key] = os.path.join(day_dir, f)
        if per_day:
            out[d] = per_day
    return out


def list_trials_force(force_base_dir: str) -> dict:
    """
    Discover force files stored as .npy with names like 'DayXX_trialYY.npy'.

    Parameters
    ----------
    force_base_dir : str
        Directory containing .npy force files.

    Returns
    -------
    dict
        { 'Day01': { 'Day01_trial01': '/.../Day01_trial01.npy', ... }, ... }
    """
    out = {}
    for f in sorted(os.listdir(force_base_dir)):
        if not f.endswith('.npy'):
            continue
        stem = os.path.splitext(f)[0]  # e.g., 'Day03_trial14'
        m = re.match(r'(Day\d+_trial\d+)$', stem, re.IGNORECASE)
        if not m:
            continue
        day = stem.split('_')[0]       # 'DayXX'
        out.setdefault(day, {})
        out[day][stem] = os.path.join(force_base_dir, f)
    return out


def load_spikes(path: str, key: str = 'spike_array') -> np.ndarray:
    """
    Load spikes from a HDF5-backed .jld2 file.

    Parameters
    ----------
    path : str
        Path to the .jld2 file.
    key : str, optional
        Dataset inside the file; default 'spike_array'.

    Returns
    -------
    np.ndarray
        Spike matrix with shape (samples, neurons).

    Raises
    ------
    KeyError
        If the dataset key is not found.
    ValueError
        If the dataset is not 2D.
    """
    with h5py.File(path, 'r') as f:
        if key not in f:
            raise KeyError(f"{key} not in {path}; keys: {list(f.keys())[:10]}")
        MU = np.array(f[key])
    if MU.ndim != 2:
        raise ValueError(f"Expected 2D (samples, neurons), got {MU.shape} in {path}")
    return MU


def load_force_npy(path: str) -> np.ndarray:
    """
    Load a 1D force trace from a .npy file; squeezes extra dims if present.

    Parameters
    ----------
    path : str
        Path to the .npy force file.

    Returns
    -------
    np.ndarray
        1D float array of force samples.
    """
    F = np.load(path)
    if F.ndim > 1:
        F = F.squeeze()
    return np.asarray(F, dtype=float)


def bin_spikes_samples_by_neurons(MU: np.ndarray,
                                  bin_ms: int = 100,
                                  fs: int = 20000) -> np.ndarray | None:
    """
    Bin spikes from (samples, neurons) into (neurons, bins) by summing within each bin.

    Parameters
    ----------
    MU : np.ndarray
        Raw spikes, shape (samples, neurons).
    bin_ms : int, optional
        Bin width in milliseconds; default 100 ms.
    fs : int, optional
        Sampling rate (Hz) of MU; default 20000.

    Returns
    -------
    np.ndarray or None
        Binned spike counts, shape (neurons, bins), or None if fewer than one full bin.
    """
    MU = np.asarray(MU)
    if MU.ndim != 2:
        raise ValueError(f"Expected 2D (samples, neurons), got {MU.shape}")
    n_samples, n_neurons = MU.shape
    bin_len = int(round(fs * bin_ms / 1000.0))
    if bin_len <= 0:
        raise ValueError("bin_len <= 0; check fs/bin_ms")
    n_bins = n_samples // bin_len
    if n_bins == 0:
        return None
    L = n_bins * bin_len
    return MU[:L, :].reshape(n_bins, bin_len, n_neurons).sum(axis=1).T


def bin_force(F: np.ndarray,
              bin_ms: int = 100,
              fs: int = 2048) -> np.ndarray | None:
    """
    Bin a force trace by averaging within each bin.

    Parameters
    ----------
    F : np.ndarray
        1D force array (samples,).
    bin_ms : int, optional
        Bin width in milliseconds; default 100 ms.
    fs : int, optional
        Sampling rate (Hz) of the force; default 2048.

    Returns
    -------
    np.ndarray or None
        Binned force (bins,), or None if fewer than one full bin.
    """
    f = np.asarray(F, dtype=float)
    bin_len = int(round(fs * bin_ms / 1000.0))
    if bin_len <= 0:
        raise ValueError("bin_len <= 0; check fs/bin_ms")
    n_bins = f.shape[0] // bin_len
    if n_bins == 0:
        return None
    L = n_bins * bin_len
    return f[:L].reshape(n_bins, bin_len).mean(axis=1)


def day_num(name: str) -> int:
    """
    Extract the numeric part from a label like 'Day07' for sorting.

    Parameters
    ----------
    name : str
        Day label.

    Returns
    -------
    int
        Parsed integer (e.g., 7). Returns a large fallback if not found.
    """
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else 999_999
