"""
Force scaling and firing-rate estimation for the online control experiment.

Two things live here, so that the experimental and simulated sides of an
analysis are treated identically:

  1. FORCE SCALING.  `scale_forces` writes each scaled trace together with a row
     in a manifest CSV recording how it was produced -- policy, baseline
     removal, scale factor, resulting min/mean/max.  The manifest is what the
     Julia simulation driver reads, so amplitude is decided once, here.

  2. FIRING-RATE ESTIMATION.  `firing_rates` is applied unchanged to
     experimental MU spike trains and to simulated spike arrays.  It requires
     the sampling rate rather than assuming one, takes explicit inclusion
     criteria, and returns excluded units alongside the reason.

Usage
-----
    from utils.functions_analysis_online_control_exp import unified_pipeline as up

    man = up.scale_forces("S1", src_dir, out_dir, policy="global_max", target=0.4)
    rates = up.firing_rates(spikes, fs=2048.0, max_isi=0.2)
"""

from __future__ import annotations

import os
import re
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Sampling rate of a simulated `spike_array`, in Hz.
#:
#: `run_sim` integrates with a fixed step of 0.05 ms (`hi = 0.05` in
#: creation_model.jl, `tspan = 0:0.05:sim_length` in the run loops), so each
#: column of `spike_array` is 0.05 ms => 20 kHz.  Prefer the `fs` stored in each
#: simulation file over this constant; it is the fallback for older files.
SIM_FS_HZ = 20_000.0

#: Force sampling rate of the experimental recordings, in Hz.  Prefer the
#: `fsamp` field stored inside each .pkl; this is only the fallback.
EXP_FS_HZ = 2048.0

#: Scaling policies accepted by :func:`scale_forces`.
POLICIES = ("global_max", "per_trial_max", "per_trial_mean")

MANIFEST_COLUMNS = [
    "subject", "day", "trial", "src_file", "out_file",
    "policy", "target", "remove_baseline",
    "baseline", "scale_factor", "extra_gain",
    "raw_min", "raw_mean", "raw_max",
    "out_min", "out_mean", "out_max",
    "n_samples", "fs", "duration_s",
]


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def extract_day(filename: str) -> Optional[int]:
    """Return the day number from a filename containing ``DayXX``, else None."""
    m = re.search(r"Day0*(\d+)", filename)
    return int(m.group(1)) if m else None


def extract_trial(filename: str) -> Optional[int]:
    r"""Return the trial number from a filename, else None.

    Handles both conventions in use: the raw recordings end in ``..._17.pkl``,
    while the files written by :func:`scale_forces` and by the Julia driver are
    named ``S1_Day01_trial21.npy`` / ``.jld2``.  The explicit ``trial`` form is
    tried first, since ``_(\d+)`` would otherwise match the day.
    """
    stem = os.path.splitext(filename)[0]
    m = re.search(r"trial0*(\d+)", stem)
    if m:
        return int(m.group(1))
    m = re.search(r"_(\d+)(?!\d)", stem)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Experimental trial loading
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    """One experimental trial loaded from a .pkl file."""
    subject: str
    day: int
    trial: int
    src_file: str
    force: np.ndarray      # (n_samples,) raw force, arbitrary units
    mu: np.ndarray         # (n_units, n_samples) binary spike raster
    fs: float              # sampling rate in Hz
    pair: Optional[Tuple[int, int]]   # 1-based indices of the task-controlled MU pair

    @property
    def duration_s(self) -> float:
        return self.force.size / self.fs


def _as_2d(x) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype == object:
        a = np.asarray(a.tolist())
    return np.atleast_2d(a)


def load_trial(path: str, subject: Optional[str] = None) -> Trial:
    """Load one experimental .pkl trial.

    Reads the sampling rate from the file's own ``fsamp`` field rather than
    assuming 2048 Hz, and picks up ``Idx_MU_Pair`` (the two motor units the
    subject was controlling in the task).
    """
    with open(path, "rb") as fh:
        data = pickle.load(fh)

    name = os.path.basename(path)

    force = _as_2d(data["Force"])[0].astype(float).ravel()
    mu = _as_2d(data["MU"]).astype(np.uint8)

    fs = float(data.get("fsamp", EXP_FS_HZ))

    pair = None
    if "Idx_MU_Pair" in data:
        p = np.asarray(data["Idx_MU_Pair"]).ravel()
        if p.size >= 2:
            pair = (int(p[0]), int(p[1]))

    if subject is None:
        m = re.search(r"MB0*(\d+)", name)
        subject = f"S{int(m.group(1))}" if m else "S?"

    return Trial(
        subject=subject,
        day=extract_day(name) or -1,
        trial=extract_trial(name) or -1,
        src_file=path,
        force=force,
        mu=mu,
        fs=fs,
        pair=pair,
    )


def list_trials(src_dir: str, subject: Optional[str] = None) -> List[Tuple[int, int, str]]:
    """Return ``(day, trial, path)`` for every parseable .pkl in ``src_dir``, sorted."""
    out = []
    for name in os.listdir(src_dir):
        if not name.endswith(".pkl"):
            continue
        day, trial = extract_day(name), extract_trial(name)
        if day is None or trial is None:
            continue
        out.append((day, trial, os.path.join(src_dir, name)))
    return sorted(out)


# ---------------------------------------------------------------------------
# Force scaling
# ---------------------------------------------------------------------------

def scale_forces(
    subject: str,
    src_dir: str,
    out_dir: str,
    policy: str = "global_max",
    target: float = 0.4,
    remove_baseline: bool = True,
    trials: Optional[Union[Dict[int, int], Sequence[Tuple[int, int]]]] = None,
    manifest_name: str = "manifest.csv",
    overwrite: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Scale experimental force traces and write them with a manifest.

    Parameters
    ----------
    subject
        Subject label, e.g. ``"S1"``.  Recorded in the manifest.
    src_dir
        Directory of experimental ``.pkl`` files.
    out_dir
        Where the scaled ``.npy`` files and the manifest are written.
    policy
        One of:

        ``"global_max"``
            One scale factor for the whole set, chosen so that the largest
            peak across all selected trials equals ``target``.  Preserves every
            relative amplitude difference (between days *and* between trials).
            Use this when force amplitude is meant to carry information.

        ``"per_trial_max"``
            Each trial scaled so its own peak equals ``target``.  Guarantees the
            same maximum recruitment on every trial, but discards all
            amplitude differences.

        ``"per_trial_mean"``
            Each trial scaled so its own mean equals ``target``.  Used by the GLM
            analyses.  Note it does *not* control the peak: on S1 it produces
            peaks of 0.44-0.81, i.e. well past full recruitment.
    target
        Target value for the quantity named by ``policy``.
    remove_baseline
        If True, subtract each trace's minimum before scaling, so the scaled
        trace starts at zero drive.  Recorded in the manifest either way.
    trials
        ``None``    -> every trial in ``src_dir``.
        ``dict``    -> ``{day: trial}``, one representative trial per day.
        ``sequence``-> explicit ``[(day, trial), ...]`` pairs.
    overwrite
        If False, existing ``.npy`` files are left alone (the manifest row is
        still written).

    Returns
    -------
    pandas.DataFrame
        The manifest, also written to ``out_dir/manifest_name``.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")

    os.makedirs(out_dir, exist_ok=True)
    available = list_trials(src_dir, subject)
    if not available:
        raise FileNotFoundError(f"No parseable .pkl files in {src_dir}")

    # --- select which trials to scale -------------------------------------
    if trials is None:
        selected = available
    else:
        if isinstance(trials, dict):
            wanted = {(int(d), int(t)) for d, t in trials.items()}
        else:
            wanted = {(int(d), int(t)) for d, t in trials}
        selected = [x for x in available if (x[0], x[1]) in wanted]
        missing = wanted - {(d, t) for d, t, _ in selected}
        if missing:
            raise ValueError(f"Requested trials not found in {src_dir}: {sorted(missing)}")

    # --- pass 1: load and measure -----------------------------------------
    loaded = []
    for day, trial, path in selected:
        tr = load_trial(path, subject=subject)
        raw = tr.force
        baseline = float(raw.min()) if remove_baseline else 0.0
        corrected = raw - baseline
        corrected = np.nan_to_num(corrected, nan=0.0, posinf=0.0, neginf=0.0)
        loaded.append((tr, baseline, corrected))

    # --- compute scale factors --------------------------------------------
    if policy == "global_max":
        global_max = max(float(c.max()) for _, _, c in loaded)
        if global_max <= 0:
            raise ValueError("Global max is non-positive; cannot scale.")
        factors = [target / global_max] * len(loaded)
        if verbose:
            print(f"[{subject}] policy=global_max  global peak={global_max:.5f}  "
                  f"-> one factor {factors[0]:.5f} for all {len(loaded)} trials")
    elif policy == "per_trial_max":
        factors = [target / float(c.max()) if c.max() > 0 else 1.0 for _, _, c in loaded]
    else:  # per_trial_mean
        factors = [target / float(c.mean()) if c.mean() > 0 else 1.0 for _, _, c in loaded]

    # --- pass 2: write ------------------------------------------------------
    rows = []
    for (tr, baseline, corrected), k in zip(loaded, factors):
        scaled = corrected * k
        out_name = f"{subject}_Day{tr.day:02d}_trial{tr.trial:02d}.npy"
        out_path = os.path.join(out_dir, out_name)

        if overwrite or not os.path.exists(out_path):
            np.save(out_path, scaled)

        rows.append({
            "subject": subject, "day": tr.day, "trial": tr.trial,
            "src_file": tr.src_file, "out_file": out_path,
            "policy": policy, "target": target, "remove_baseline": bool(remove_baseline),
            "baseline": baseline, "scale_factor": k, "extra_gain": 1.0,
            "raw_min": float(tr.force.min()), "raw_mean": float(tr.force.mean()),
            "raw_max": float(tr.force.max()),
            "out_min": float(scaled.min()), "out_mean": float(scaled.mean()),
            "out_max": float(scaled.max()),
            "n_samples": int(scaled.size), "fs": tr.fs,
            "duration_s": tr.duration_s,
        })

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS).sort_values(["day", "trial"])
    manifest_path = os.path.join(out_dir, manifest_name)
    manifest.to_csv(manifest_path, index=False)

    if verbose:
        print(f"[{subject}] wrote {len(manifest)} scaled traces + manifest -> {manifest_path}")
        print(f"[{subject}] resulting peak drive: min={manifest.out_max.min():.3f} "
              f"median={manifest.out_max.median():.3f} max={manifest.out_max.max():.3f}")

    return manifest


def load_manifest(path: str) -> pd.DataFrame:
    """Load a manifest CSV written by :func:`scale_forces`."""
    if os.path.isdir(path):
        path = os.path.join(path, "manifest.csv")
    return pd.read_csv(path)


def verify_manifest(manifest: Union[str, pd.DataFrame], tol: float = 1e-9) -> pd.DataFrame:
    """Re-read every ``.npy`` and check it still matches its manifest row.

    Confirms that every trace named in the manifest exists on disk with the
    recorded length and peak, so a sweep is not started against stale or
    partially written files.  Returns the rows that fail.
    """
    man = load_manifest(manifest) if isinstance(manifest, str) else manifest
    bad = []
    for _, r in man.iterrows():
        if not os.path.exists(r.out_file):
            bad.append({**r.to_dict(), "problem": "missing file"})
            continue
        x = np.load(r.out_file)
        if x.size != r.n_samples:
            bad.append({**r.to_dict(), "problem": f"n_samples {x.size} != {r.n_samples}"})
        elif not np.isclose(x.max(), r.out_max, atol=max(tol, 1e-6)):
            bad.append({**r.to_dict(), "problem": f"max {x.max():.6f} != {r.out_max:.6f}"})
    return pd.DataFrame(bad)


# ---------------------------------------------------------------------------
# Firing-rate estimation  (ONE estimator, used for exp and sim alike)
# ---------------------------------------------------------------------------

def firing_rates(
    spikes: np.ndarray,
    fs: float,
    max_isi: Optional[float] = 0.2,
    min_spikes: int = 5,
    min_duty: float = 0.0,
    fr_cap: Optional[float] = 80.0,
    unit_ids: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Average firing rate per unit from a binary spike raster.

    Applied identically to experimental MU rasters and simulated spike arrays.

    Parameters
    ----------
    spikes
        ``(n_units, n_samples)`` binary raster (non-zero = spike).
    fs
        Sampling rate of ``spikes`` in Hz.  For simulations this is
        :data:`SIM_FS_HZ` (20 kHz), **not** 1000 Hz.
    max_isi
        Discard inter-spike intervals longer than this (seconds) before
        averaging, giving the rate while a unit was active.  With
        ``max_isi=None`` every ISI is averaged, so de-recruitment gaps count as
        slow firing.  Experimental units de-recruit and simulated units driven
        above threshold do not, so averaging all ISIs biases the experimental
        distribution down and the simulated one up.  On S1 the difference moves
        the experimental median from 10.2 to 11.5 Hz and the 10th percentile
        from 6.2 to 9.6 Hz.  ``0.2`` s (i.e. 5 Hz) is a common choice for tonic
        MU firing.
    min_spikes
        Units with fewer spikes than this are marked not-included.
    min_duty
        Units whose active span (first to last spike) covers less than this
        fraction of the recording are marked not-included.  Use e.g. 0.5 to
        require a unit to be present for at least half the trial.
    fr_cap
        Rates above this (Hz) are marked not-included.  ``None`` disables.

    Returns
    -------
    pandas.DataFrame
        One row per unit -- **including excluded ones**, flagged via the
        ``included`` column with a ``reason``, so nothing is dropped silently.
        Columns: ``unit, n_spikes, n_isi_all, n_isi_used, isi_dropped_frac,
        duty, active_s, record_s, rate_hz, rate_hz_all_isi, included, reason``.
        ``rate_hz_all_isi`` is the same rate with every ISI averaged, kept for
        comparison against ``rate_hz``.
    """
    spikes = np.atleast_2d(np.asarray(spikes))
    n_units, n_samples = spikes.shape
    record_s = n_samples / fs
    ids = list(unit_ids) if unit_ids is not None else list(range(n_units))

    rows = []
    for i in range(n_units):
        idx = np.flatnonzero(spikes[i] != 0)
        n_spikes = int(idx.size)

        row = {
            "unit": ids[i], "n_spikes": n_spikes,
            "n_isi_all": max(n_spikes - 1, 0), "n_isi_used": 0,
            "isi_dropped_frac": np.nan, "duty": np.nan,
            "active_s": np.nan, "record_s": record_s,
            "rate_hz": np.nan, "rate_hz_all_isi": np.nan,
            "included": False, "reason": "",
        }

        if n_spikes < 2:
            row["reason"] = "fewer than 2 spikes"
            rows.append(row)
            continue

        isi_all = np.diff(idx) / fs
        active_s = (idx[-1] - idx[0]) / fs
        row["active_s"] = active_s
        row["duty"] = active_s / record_s if record_s > 0 else np.nan
        row["rate_hz_all_isi"] = 1.0 / isi_all.mean() if isi_all.mean() > 0 else np.nan

        isi_used = isi_all if max_isi is None else isi_all[isi_all <= max_isi]
        row["n_isi_used"] = int(isi_used.size)
        row["isi_dropped_frac"] = 1.0 - isi_used.size / isi_all.size

        if isi_used.size == 0 or isi_used.mean() <= 0:
            row["reason"] = "no ISI below max_isi"
            rows.append(row)
            continue

        rate = 1.0 / isi_used.mean()
        row["rate_hz"] = rate

        if n_spikes < min_spikes:
            row["reason"] = f"n_spikes < {min_spikes}"
        elif row["duty"] < min_duty:
            row["reason"] = f"duty < {min_duty}"
        elif fr_cap is not None and rate > fr_cap:
            row["reason"] = f"rate > {fr_cap} Hz"
        else:
            row["included"] = True

        rows.append(row)

    return pd.DataFrame(rows)


def included_rates(df: pd.DataFrame) -> np.ndarray:
    """Convenience: the ``rate_hz`` values of the included units, as an array."""
    return df.loc[df.included, "rate_hz"].to_numpy()


# ---------------------------------------------------------------------------
# Simulated spike loading
# ---------------------------------------------------------------------------

def load_sim_spikes(path: str, fs_fallback: float = SIM_FS_HZ,
                    verbose: bool = True) -> Tuple[np.ndarray, float]:
    """Load ``spike_array`` (and its sampling rate) from a simulation .jld2.

    Prefers an ``fs`` entry stored alongside the spikes by the Julia side; if
    the file predates that, falls back to :data:`SIM_FS_HZ` with a warning.

    Returns ``(spikes, fs)`` with ``spikes`` shaped ``(n_units, n_samples)``.
    """
    import h5py

    with h5py.File(path, "r") as h:
        arr = _read_jld2(h, "spike_array")
        if arr is None:
            raise KeyError(f"'spike_array' not found in {path}")
        fs_val = _read_jld2(h, "fs")

    # JLD2 stores Julia column-major arrays transposed relative to numpy.
    spikes = np.asarray(arr)
    if spikes.ndim != 2:
        raise ValueError(f"spike_array in {path} is not 2-D (shape {spikes.shape})")
    if spikes.shape[0] > spikes.shape[1]:
        spikes = spikes.T   # want (n_units, n_samples); units are the short axis

    if fs_val is None:
        fs = fs_fallback
        if verbose:
            print(f"  ! {os.path.basename(path)}: no 'fs' stored, assuming "
                  f"{fs_fallback:g} Hz (0.05 ms integration step)")
    else:
        fs = float(np.ravel(fs_val)[0])

    return spikes, fs


def load_precomputed_rates(path: str, key: str = "avg_rates_per_neuron") -> np.ndarray:
    """Read a firing-rate vector from a ``*_firing_rates.jld2`` file.

    These files hold rates computed at simulation time, over all ISIs and with
    no record of the gain used.  Prefer recomputing from ``spike_array`` via
    :func:`load_sim_spikes` + :func:`firing_rates`, which applies the same
    estimator as the experimental side.
    """
    import h5py

    with h5py.File(path, "r") as h:
        arr = _read_jld2(h, key)
    if arr is None:
        raise KeyError(f"'{key}' not found in {path}")
    return np.asarray(arr, float).ravel()


def _read_jld2(h, key: str):
    """Read a numeric array from a JLD2/HDF5 handle, trying JLD2's layouts."""
    import h5py

    def _val(d):
        # JLD2 writes Julia scalars as scalar dataspaces, which reject [:]
        return d[()] if d.shape == () else d[:]

    if key in h:
        node = h[key]
        if isinstance(node, h5py.Dataset) and node.dtype.kind in "biuf":
            return _val(node)
        if isinstance(node, h5py.Group) and "value" in node:
            v = node["value"]
            if isinstance(v, h5py.Dataset) and v.dtype.kind in "biuf":
                return _val(v)
    vk = f"{key}/value"
    if vk in h and isinstance(h[vk], h5py.Dataset) and h[vk].dtype.kind in "biuf":
        return _val(h[vk])
    return None


# ---------------------------------------------------------------------------
# Time-domain comparison
# ---------------------------------------------------------------------------

def smoothed_rate(spikes: np.ndarray, fs: float, window_s: float = 0.4,
                  target_fs: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """Instantaneous firing rate over time, by Hanning-window smoothing.

    Uses the convention followed elsewhere in the repository -- a normalised
    Hanning window applied to the binary spike train (``conv_spikes_rates.jl``
    uses 400 ms, ``firing_rates.compute_firing_rates`` uses 300 ms).

    Spikes are first binned at ``target_fs`` and smoothed there rather than at
    the native rate.  For a window of several hundred milliseconds the result is
    equivalent, and it keeps a 181-unit, 20 kHz, 20-second raster from expanding
    into hundreds of megabytes of float.

    Parameters
    ----------
    spikes
        ``(n_units, n_samples)`` binary raster.
    fs
        Sampling rate of ``spikes`` in Hz.
    window_s
        Hanning window width in seconds.
    target_fs
        Output sampling rate in Hz.

    Returns
    -------
    (t, rates)
        ``t`` of length ``n_out`` in seconds; ``rates`` shaped
        ``(n_units, n_out)`` in Hz.
    """
    spikes = np.atleast_2d(np.asarray(spikes))
    n_units, n_samples = spikes.shape
    duration_s = n_samples / fs
    n_out = max(int(round(duration_s * target_fs)), 2)

    edges = np.linspace(0, n_samples, n_out + 1)
    binned = np.zeros((n_units, n_out), float)
    for i in range(n_units):
        idx = np.flatnonzero(spikes[i] != 0)
        if idx.size:
            binned[i], _ = np.histogram(idx, bins=edges)

    win = np.hanning(max(int(round(window_s * target_fs)), 3))
    win = win / win.sum()
    rates = np.vstack([np.convolve(b, win, mode="same") * target_fs for b in binned])

    t = (np.arange(n_out) + 0.5) / target_fs
    return t, rates


def recruitment_window(spikes: np.ndarray, fs: float,
                       min_spikes: int = 5) -> pd.DataFrame:
    """First and last spike time per unit.

    Returns a DataFrame with ``unit, n_spikes, first_s, last_s, active_s``.
    Units with fewer than ``min_spikes`` spikes get NaN times, so they can be
    excluded from matching without being silently renumbered.
    """
    spikes = np.atleast_2d(np.asarray(spikes))
    rows = []
    for i in range(spikes.shape[0]):
        idx = np.flatnonzero(spikes[i] != 0)
        if idx.size < min_spikes:
            rows.append({"unit": i, "n_spikes": int(idx.size), "first_s": np.nan,
                         "last_s": np.nan, "active_s": np.nan})
        else:
            first, last = idx[0] / fs, idx[-1] / fs
            rows.append({"unit": i, "n_spikes": int(idx.size), "first_s": first,
                         "last_s": last, "active_s": last - first})
    return pd.DataFrame(rows)


def match_units_by_recruitment(exp_spikes: np.ndarray, exp_fs: float,
                               sim_spikes: np.ndarray, sim_fs: float,
                               min_spikes: int = 5) -> pd.DataFrame:
    """Pair each experimental unit with a simulated unit recruited at a similar time.

    The pool holds far more motoneurons than decomposition recovers (181 against
    11-17 here), so a like-for-like time-domain comparison needs a subset of the
    pool rather than all of it.  Units are paired greedily by recruitment time --
    closest pair first, each simulated unit used at most once.

    Recruitment time is the first spike, so this makes recruitment *order* the
    basis of the comparison and leaves rate modulation and de-recruitment free to
    agree or disagree.

    Returns
    -------
    pandas.DataFrame
        ``exp_unit, sim_unit, exp_first_s, sim_first_s, d_first_s,
        exp_last_s, sim_last_s, d_last_s``, ordered by experimental recruitment.
    """
    e = recruitment_window(exp_spikes, exp_fs, min_spikes).dropna(subset=["first_s"])
    s = recruitment_window(sim_spikes, sim_fs, min_spikes).dropna(subset=["first_s"])
    if e.empty or s.empty:
        return pd.DataFrame(columns=["exp_unit", "sim_unit", "exp_first_s",
                                     "sim_first_s", "d_first_s", "exp_last_s",
                                     "sim_last_s", "d_last_s"])

    cost = np.abs(e.first_s.to_numpy()[:, None] - s.first_s.to_numpy()[None, :])
    e_idx, s_idx = e.index.to_numpy(), s.index.to_numpy()

    pairs, used_e, used_s = [], set(), set()
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for ei, si in order:
        if ei in used_e or si in used_s:
            continue
        used_e.add(ei); used_s.add(si)
        er, sr = e.loc[e_idx[ei]], s.loc[s_idx[si]]
        pairs.append({
            "exp_unit": int(er.unit), "sim_unit": int(sr.unit),
            "exp_first_s": er.first_s, "sim_first_s": sr.first_s,
            "d_first_s": sr.first_s - er.first_s,
            "exp_last_s": er.last_s, "sim_last_s": sr.last_s,
            "d_last_s": sr.last_s - er.last_s,
        })
        if len(used_e) == len(e) or len(used_s) == len(s):
            break

    return (pd.DataFrame(pairs)
            .sort_values("exp_first_s")
            .reset_index(drop=True))


def match_units_by_similarity(exp_spikes: np.ndarray, exp_fs: float,
                              sim_spikes: np.ndarray, sim_fs: float,
                              window_s: float = 0.4, target_fs: float = 100.0,
                              criterion: str = "rms",
                              min_spikes: int = 5) -> pd.DataFrame:
    """Pair each experimental unit with the simulated unit that most resembles it.

    Answers a different question from :func:`match_units_by_recruitment`.  That
    one asks whether unit *i* has a counterpart recruited at the same moment;
    this one asks whether the pool *contains* units that behave like the ones
    that were recorded -- the relevant question when decomposition recovers a
    dozen units out of a pool of ~180 and is not a uniform sample of it.

    **This is a best-of-N selection, biased in the model's favour by construction.**
    With 181 candidates per experimental unit, the closest one is close whatever
    the model does. Always report the selected score against the score of an
    arbitrary pairing (see the shuffled control in the notebook); the selection is
    only meaningful to the extent it beats that null.

    Parameters
    ----------
    criterion
        ``"rms"``  -- minimise the RMS difference between the smoothed rate
        traces, in Hz.  Penalises both shape and level, so a match looks alike
        rather than merely covarying.
        ``"corr"`` -- maximise the correlation of the traces.  Ignores level, so
        a unit firing at 20 Hz can match one at 10 Hz.
    min_spikes
        Units below this are excluded from both sides.

    Returns
    -------
    pandas.DataFrame
        ``exp_unit, sim_unit, rms_hz, corr, exp_rate_hz, sim_rate_hz,
        exp_first_s, sim_first_s``, ordered best match first.  Assignment is
        one-to-one and optimal over the whole cost matrix.
    """
    from scipy.optimize import linear_sum_assignment

    if criterion not in ("rms", "corr"):
        raise ValueError(f"criterion must be 'rms' or 'corr', got {criterion!r}")

    e_win = recruitment_window(exp_spikes, exp_fs, min_spikes).dropna(subset=["first_s"])
    s_win = recruitment_window(sim_spikes, sim_fs, min_spikes).dropna(subset=["first_s"])
    cols = ["exp_unit", "sim_unit", "rms_hz", "corr", "exp_rate_hz", "sim_rate_hz",
            "exp_first_s", "sim_first_s"]
    if e_win.empty or s_win.empty:
        return pd.DataFrame(columns=cols)

    t_e, r_e = smoothed_rate(exp_spikes, exp_fs, window_s, target_fs)
    t_s, r_s = smoothed_rate(sim_spikes, sim_fs, window_s, target_fs)
    n = min(len(t_e), len(t_s))
    E = r_e[e_win.unit.to_numpy()][:, :n]
    S = r_s[s_win.unit.to_numpy()][:, :n]

    # rms[i, j] = RMS difference between experimental i and simulated j
    rms = np.sqrt(((E[:, None, :] - S[None, :, :]) ** 2).mean(axis=2))

    Ez = E - E.mean(1, keepdims=True)
    Sz = S - S.mean(1, keepdims=True)
    denom = np.outer(np.sqrt((Ez ** 2).sum(1)), np.sqrt((Sz ** 2).sum(1)))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0, (Ez @ Sz.T) / denom, np.nan)

    cost = rms if criterion == "rms" else -np.nan_to_num(corr, nan=-1.0)
    ri, ci = linear_sum_assignment(cost)

    eu, su = e_win.unit.to_numpy(), s_win.unit.to_numpy()
    out = pd.DataFrame({
        "exp_unit": eu[ri], "sim_unit": su[ci],
        "rms_hz": rms[ri, ci], "corr": corr[ri, ci],
        "exp_rate_hz": E[ri].mean(1), "sim_rate_hz": S[ci].mean(1),
        "exp_first_s": e_win.set_index("unit").loc[eu[ri], "first_s"].to_numpy(),
        "sim_first_s": s_win.set_index("unit").loc[su[ci], "first_s"].to_numpy(),
    })[cols]
    return out.sort_values("rms_hz" if criterion == "rms" else "corr",
                           ascending=(criterion == "rms")).reset_index(drop=True)
