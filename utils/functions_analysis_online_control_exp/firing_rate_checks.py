"""Helpers for `notebooks/check_firing_rates_online_control.ipynb`.

Everything the notebook needs in order to compare simulated motoneuron pools
against the experimentally decomposed motor units: the plot style, locating the
recordings, firing rates per day, the force-scaling dry run, pairing recorded
units with simulated ones, and the two time-domain figures.

Nothing here reads a global. Every value the notebook configures -- subject,
directories, smoothing window, inclusion criteria -- is passed in, so the same
call made twice with the same arguments gives the same answer.

The heavy lifting (loading trials, estimating rates, smoothing, matching units)
lives in `unified_pipeline`; these functions arrange it.
"""

import os
import re
import glob
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

try:                                   # imported as part of the package
    from . import unified_pipeline as up
except ImportError:                    # imported from inside the folder
    import unified_pipeline as up


# --------------------------------------------------------------------------
# Plot style
# --------------------------------------------------------------------------

# Categorical colours, assigned by entity.
C_EXP = "#2a78d6"   # slot 1, blue   -- experimental
C_SIM = "#eb6834"   # slot 2, orange -- simulated
C_ALT = "#1baf7a"   # slot 3, aqua   -- secondary series
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"


def apply_plot_style():
    """Set the figure defaults used throughout the notebook."""
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": INK2, "ytick.color": INK2,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "legend.frameon": False, "figure.dpi": 110, "savefig.bbox": "tight",
    })


def tidy(ax, ygrid_only=True):
    """Recessive grid: horizontal rules only, no vertical clutter."""
    ax.xaxis.grid(not ygrid_only)
    ax.yaxis.grid(True)
    return ax


# --------------------------------------------------------------------------
# Locating the recordings
# --------------------------------------------------------------------------

def find_data_root(subject, repo_root):
    """Locate the directory holding the per-subject .pkl folders.

    Checked in order: $MN_DATA_ROOT, <repo>/data, ~/Desktop/DATA.
    """
    candidates = [os.environ.get("MN_DATA_ROOT"), Path(repo_root) / "data",
                  Path.home() / "Desktop" / "DATA"]
    tried = []
    for c in candidates:
        if c is None:
            continue
        c = Path(c).expanduser()
        tried.append(str(c))
        if (c / subject).is_dir():
            return c
    raise FileNotFoundError(
        f"Could not find a directory containing {subject}/.\n"
        f"Set MN_DATA_ROOT and restart the kernel, e.g.\n"
        f"    export MN_DATA_ROOT=/path/to/DATA\n"
        f"Tried: " + ", ".join(tried))


# --------------------------------------------------------------------------
# Firing rates
# --------------------------------------------------------------------------

def exp_rates_per_day(exp_dir, max_isi, **kw):
    """Firing rates for every experimental unit, grouped by day.

    Returns a tidy DataFrame: one row per (day, trial, unit).
    """
    frames = []
    for day, trial, path in up.list_trials(str(exp_dir)):
        tr = up.load_trial(path)
        df = up.firing_rates(tr.mu, fs=tr.fs, max_isi=max_isi, **kw)
        df.insert(0, "day", day); df.insert(1, "trial", trial)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_sim_rates(sim_root, subject, max_isi, **kw):
    """Firing rates from simulated spike arrays, grouped by day.

    Reads `fs` from each file rather than assuming it.
    """
    files = sorted(glob.glob(os.path.join(str(sim_root), "Day*", "*.jld2")))
    if not files:
        return None
    frames = []
    for path in files:
        spikes, fs = up.load_sim_spikes(path)
        df = up.firing_rates(spikes, fs=fs, max_isi=max_isi, **kw)
        day = up.extract_day(os.path.basename(path))
        trial = up.extract_trial(os.path.basename(path))
        df.insert(0, "day", day); df.insert(1, "trial", trial); df["fs"] = fs
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_precomputed_rates(data_dir, subject, pattern="force*_firing_rates_rep_.jld2"):
    """Pre-computed rate vectors from earlier runs, keyed by day."""
    out = {}
    for path in glob.glob(str(Path(data_dir) / "firings" / subject / pattern)):
        m = re.search(r"force(\d+)_", os.path.basename(path))
        if m:
            out[int(m.group(1))] = up.load_precomputed_rates(path)
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------
# Force scaling
# --------------------------------------------------------------------------

def dry_run_policies(subject, exp_dir, trials, policies):
    """Compute what each scaling policy would produce, writing nothing."""
    out = {}
    for policy, target in policies:
        with tempfile.TemporaryDirectory() as tmp:
            m = up.scale_forces(subject, str(exp_dir), tmp, policy=policy,
                                target=target, trials=trials, verbose=False)
        out[f"{policy} -> {target}"] = m.set_index("day")["out_max"]
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Pairing recorded units with simulated ones
# --------------------------------------------------------------------------

def load_trial_pair(day, trial, exp_dir, sim_root, subject, min_spikes):
    """Experimental and simulated rasters for one trial, with units paired.

    Returns (trial, sim_spikes, sim_fs, pairs); the last three are None when the
    simulation for that trial has not been run yet.
    """
    hits = [p for d, t, p in up.list_trials(str(exp_dir)) if d == day and t == trial]
    if not hits:
        raise FileNotFoundError(f"no experimental Day{day:02d} trial{trial:02d}")
    tr = up.load_trial(hits[0], subject=subject)

    sim_path = Path(sim_root) / f"Day{day:02d}" / f"{subject}_Day{day:02d}_trial{trial:02d}.jld2"
    if not sim_path.exists():
        return tr, None, None, None

    sim_spikes, sim_fs = up.load_sim_spikes(str(sim_path), verbose=False)
    pairs = up.match_units_by_recruitment(tr.mu, tr.fs, sim_spikes, sim_fs,
                                          min_spikes=min_spikes)
    return tr, sim_spikes, sim_fs, pairs


def paired_rate_traces(tr, sim_spikes, sim_fs, pairs, smooth_s, rate_fs):
    """Smoothed rate traces for the paired units, on a common time base."""
    t_e, r_e = up.smoothed_rate(tr.mu, tr.fs, smooth_s, rate_fs)
    t_s, r_s = up.smoothed_rate(sim_spikes, sim_fs, smooth_s, rate_fs)
    n = min(len(t_e), len(t_s))
    return (t_e[:n],
            r_e[pairs.exp_unit.to_numpy()][:, :n],
            r_s[pairs.sim_unit.to_numpy()][:, :n])


def median_rate_corr(tr, sim_spikes, sim_fs, exp_units, sim_units, smooth_s, rate_fs):
    """Median correlation between the smoothed rate traces of paired units."""
    t_e, r_e = up.smoothed_rate(tr.mu, tr.fs, smooth_s, rate_fs)
    t_s, r_s = up.smoothed_rate(sim_spikes, sim_fs, smooth_s, rate_fs)
    n = min(len(t_e), len(t_s))
    out = []
    for a, b in zip(exp_units, sim_units):
        x, y = r_e[a][:n], r_s[b][:n]
        out.append(np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else np.nan)
    return np.nanmedian(out)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def plot_time_domain(day, trial, manifest, exp_dir, sim_root, subject,
                     smooth_s, rate_fs, min_spikes, max_units=None):
    """Force, spike rasters and smoothed rate traces, experimental vs simulated."""
    tr, sim_spikes, sim_fs, pairs = load_trial_pair(day, trial, exp_dir, sim_root,
                                                    subject, min_spikes)
    if pairs is None or pairs.empty:
        print(f"Day{day:02d} trial{trial:02d}: no simulation available")
        return None
    if max_units:
        pairs = pairs.head(max_units)

    t, r_e, r_s = paired_rate_traces(tr, sim_spikes, sim_fs, pairs, smooth_s, rate_fs)
    n = len(pairs)
    ramps = [plt.cm.Blues(np.linspace(0.38, 0.88, n)),
             plt.cm.Oranges(np.linspace(0.38, 0.88, n))]

    # The drive is drawn above each column rather than spanning both, so that
    # events line up vertically within a column.
    row = manifest[(manifest.day == day) & (manifest.trial == trial)]
    force = np.load(row.iloc[0].out_file) if len(row) else None
    t_force = np.arange(force.size) / row.iloc[0].fs if force is not None else None
    f_lim = (min(0.0, force.min()) - 0.01, force.max() * 1.15) if force is not None else None

    fig = plt.figure(figsize=(12.5, 8.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.5, 1.0, 1.0], hspace=0.16, wspace=0.07)

    sources = [("Experimental", C_EXP, tr.mu, pairs.exp_unit.to_numpy(), tr.fs, r_e),
               ("Simulated", C_SIM, sim_spikes, pairs.sim_unit.to_numpy(), sim_fs, r_s)]
    top = max(np.nanmax(r_e), np.nanmax(r_s)) * 1.08

    for col, (name, colr, spikes, units, fs, rates) in enumerate(sources):
        ax_f = fig.add_subplot(gs[0, col])
        ax_r = fig.add_subplot(gs[1, col], sharex=ax_f)
        ax_c = fig.add_subplot(gs[2, col], sharex=ax_f)

        if force is not None:
            ax_f.plot(t_force, force, color=INK2, linewidth=1.5)
            ax_f.set_ylim(*f_lim)
        ax_f.set_title(name, fontsize=11.5, color=colr, fontweight="semibold", pad=6)
        ax_f.tick_params(labelbottom=False)
        tidy(ax_f)

        for k, u in enumerate(units):
            st = np.flatnonzero(spikes[u] != 0) / fs
            ax_r.vlines(st, k + 0.58, k + 1.42, color=ramps[col][k], linewidth=0.6)
        ax_r.set_ylim(0.4, n + 0.6)
        ax_r.set_yticks([])
        ax_r.tick_params(labelbottom=False)
        tidy(ax_r)
        ax_r.yaxis.grid(False)

        for k in range(n):
            ax_c.plot(t, rates[k], color=ramps[col][k], linewidth=1.3)
        ax_c.set_ylim(0, top)
        ax_c.set_xlabel("Time (s)")
        tidy(ax_c)

        ax_f.set_xlim(t[0], t[-1])
        if col == 0:
            ax_f.set_ylabel("Drive (a.u.)")
            ax_r.set_ylabel("Unit (by recruitment)")
            ax_c.set_ylabel(f"Firing rate (Hz)\n{int(smooth_s * 1000)} ms Hanning")
        else:
            ax_f.tick_params(labelleft=False)
            ax_c.tick_params(labelleft=False)

    fig.suptitle(f"{subject} — Day {day:02d}, trial {trial:02d}   "
                 f"({n} paired units from a pool of {sim_spikes.shape[0]})",
                 fontsize=12.5, fontweight="semibold", color=INK, y=0.965)
    plt.show()
    return fig


def plot_best_matches(day, trial, best_by_day, subject, smooth_s, rate_fs,
                      n_show=8, ncol=4):
    """Recorded units and their closest simulated counterparts, best first."""
    if (day, trial) not in best_by_day:
        print(f"Day{day:02d} trial{trial:02d}: no simulation available")
        return None
    tr, sim_spikes, sim_fs, best = best_by_day[(day, trial)]
    best = best.head(n_show)

    t_e, r_e = up.smoothed_rate(tr.mu, tr.fs, smooth_s, rate_fs)
    t_s, r_s = up.smoothed_rate(sim_spikes, sim_fs, smooth_s, rate_fs)
    n = min(len(t_e), len(t_s))
    t = t_e[:n]

    nrow = int(np.ceil(len(best) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.5 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    top = max(r_e[best.exp_unit.to_numpy()][:, :n].max(),
              r_s[best.sim_unit.to_numpy()][:, :n].max()) * 1.12

    for ax, (_, row) in zip(axes, best.iterrows()):
        ax.plot(t, r_e[int(row.exp_unit)][:n], color=C_EXP, linewidth=1.8,
                label="Experimental")
        ax.plot(t, r_s[int(row.sim_unit)][:n], color=C_SIM, linewidth=1.8,
                label="Simulated", alpha=0.9)
        ax.set_title(f"exp {int(row.exp_unit)} / sim {int(row.sim_unit)}   "
                     f"RMS {row.rms_hz:.1f} Hz,  r = {row['corr']:.2f}", fontsize=9.5)
        ax.set_ylim(0, top)
        tidy(ax)
    for ax in axes[len(best):]:
        fig.delaxes(ax)
    for i in range(len(best)):
        if i % ncol == 0:
            axes[i].set_ylabel("Firing rate (Hz)")
        if i >= len(best) - ncol:
            axes[i].set_xlabel("Time (s)")
            axes[i].tick_params(labelbottom=True)
    axes[0].legend(loc="upper right", fontsize=8.5)

    fig.suptitle(f"{subject} — Day {day:02d}, trial {trial:02d}: "
                 f"closest simulated match for each recorded unit",
                 fontsize=12, fontweight="semibold", y=1.005, color=INK)
    plt.tight_layout(); plt.show()
    return fig
