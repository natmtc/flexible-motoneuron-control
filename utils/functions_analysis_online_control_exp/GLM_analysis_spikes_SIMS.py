import os
import re
import h5py
import numpy as np
from scipy.special import gammaln


def compute_pR2(y_true, y_pred, eps: float = 1e-12) -> float:
    """
    Poisson deviance-based pseudo-R² (Perich-style).

    pR² = 1 − (LL_sat − LL_model) / (LL_sat − LL_null)

    Parameters
    ----------
    y_true : array-like
        Observed spike counts per bin.
    y_pred : array-like
        Predicted mean counts per bin (must be nonnegative).
    eps : float, optional
        Small constant to avoid log(0); defaults to 1e-12.

    Returns
    -------
    float
        pR² value in (-∞, 1]. Returns NaN if denominator is zero or invalid.
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


def build_session_files(base_dir: str) -> dict:
    """
    Scan a directory tree structured as base_dir/DayXX/*.jld2 and
    return a mapping of day -> list of full file paths.

    Parameters
    ----------
    base_dir : str
        Root directory containing DayXX subfolders with .jld2 files.

    Returns
    -------
    dict
        { 'Day01': [ '/.../Day01/file1.jld2', ... ], 'Day02': [...], ... }
    """
    session_files = {}
    for d in sorted(os.listdir(base_dir)):
        if not d.lower().startswith('day'):
            continue
        day_dir = os.path.join(base_dir, d)
        if not os.path.isdir(day_dir):
            continue
        files = sorted(f for f in os.listdir(day_dir) if f.endswith('.jld2'))
        if files:
            session_files[d] = [os.path.join(day_dir, f) for f in files]
    return session_files


def load_jld2_h5(path: str, key: str = 'spike_array') -> np.ndarray:
    """
    Load spikes from a HDF5-backed .jld2 file.

    Parameters
    ----------
    path : str
        Path to the .jld2 file.
    key : str, optional
        Dataset name inside the file containing spikes; default 'spike_array'.

    Returns
    -------
    np.ndarray
        Spike array with shape (samples, neurons).

    Raises
    ------
    KeyError
        If the dataset key is not found in the file.
    ValueError
        If the dataset is not 2D.
    """
    with h5py.File(path, 'r') as f:
        if key not in f:
            raise KeyError(f"{key} not found in {path}. Keys: {list(f.keys())[:10]}")
        MU = np.array(f[key])
    if MU.ndim != 2:
        raise ValueError(f"Expected 2D (samples, neurons), got {MU.shape} in {path}")
    return MU


def bin_spikes_samples_by_neurons(MU: np.ndarray, bin_size_ms: int = 100, fsamp: int = 20000) -> np.ndarray | None:
    """
    Bin spikes from (samples x neurons) into (neurons x bins) by summing within each bin.

    Parameters
    ----------
    MU : np.ndarray
        Spike matrix (samples, neurons).
    bin_size_ms : int, optional
        Bin width in milliseconds; default 100 ms.
    fsamp : int, optional
        Sampling rate (Hz) of MU; default 20000.

    Returns
    -------
    np.ndarray or None
        Binned spike counts (neurons, bins), or None if fewer than one full bin.
    """
    MU = np.asarray(MU)
    if MU.ndim != 2:
        raise ValueError(f"Expected 2D (samples, neurons), got {MU.shape}")
    n_samples, n_neurons = MU.shape
    bin_len = int(round(fsamp * bin_size_ms / 1000.0))
    if bin_len <= 0:
        raise ValueError("bin_len <= 0; check fsamp/bin_size_ms")
    n_bins = n_samples // bin_len
    if n_bins == 0:
        return None
    L = n_bins * bin_len
    # (samples, neurons) -> (bins, bin_len, neurons) -> sum over bin_len -> (bins, neurons) -> transpose
    return MU[:L, :].reshape(n_bins, bin_len, n_neurons).sum(axis=1).T


def create_glm_data_pop(spikes_T: np.ndarray, target_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build GLM design for spike->spike using instantaneous population activity.

    Parameters
    ----------
    spikes_T : np.ndarray
        Binned spikes with shape (bins, neurons) — time along rows.
    target_idx : int
        Index of the target neuron to predict.

    Returns
    -------
    (X, Y) : tuple of np.ndarray
        X : (bins, predictors) = all other neurons (columns) excluding target.
        Y : (bins,)            = target neuron's spike counts.
    """
    if spikes_T.ndim != 2:
        raise ValueError(f"spikes_T must be 2D (bins, neurons), got {spikes_T.shape}")
    if not (0 <= target_idx < spikes_T.shape[1]):
        raise IndexError(f"target_idx {target_idx} out of range 0..{spikes_T.shape[1]-1}")

    Y = spikes_T[:, target_idx].astype(float)
    X = np.delete(spikes_T, target_idx, axis=1).astype(float)
    return X, Y


def day_sort_key(name: str) -> int:
    """
    Extract a sortable integer from a day label like 'Day07'.

    Parameters
    ----------
    name : str
        Day string (e.g., 'Day01', 'Day7', 'DayABC').

    Returns
    -------
    int
        Extracted integer for sorting, or a large fallback if not found.
    """
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else 999_999
