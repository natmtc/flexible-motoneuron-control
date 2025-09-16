import os, re, csv, pickle
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

# --- helpers ---
def extract_day(filename: str) -> Optional[str]:
    m = re.search(r"(Day\d{2})", filename)
    return m.group(1) if m else None

def extract_trial(filename: str) -> int:
    m = re.search(r'_(\d+)[^0-9]', filename)
    return int(m.group(1)) if m else 0

def _safe_force_from_pickle(data) -> np.ndarray:
    try:
        x = np.array(data['Force']).squeeze()
        if x.ndim > 1:
            x = np.array(data['Force'].tolist())[0].squeeze()
    except Exception:
        x = np.array(data['Force'].tolist())[0].squeeze()
    return np.asarray(x, float)

def scale_and_save_all_trials_mean(
    directory_path: str,
    save_dir: str,
    target_mean: float = 0.3,
    plot_examples: bool = True,
    n_examples: int = 6,
    overwrite: bool = True,
    verbose: bool = True,
    manifest_name: str = "manifest_mean.csv"
):
    """
    For every *.pkl in directory_path:
      1) load Force
      2) baseline-correct per trial (subtract min so min==0)
      3) scale whole trace so mean == target_mean
      4) save as .npy to save_dir: <Day>_trial<NN>_06.npy

    Writes a manifest CSV with:
      (day, trial, src_file, out_file, mean_before, target_mean, scale_factor)
    """
    os.makedirs(save_dir, exist_ok=True)

    pkl_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {directory_path}")

    # group by day (order nicely)
    files_by_day = {}
    for f in pkl_files:
        day = extract_day(f)
        if day is not None:
            files_by_day.setdefault(day, []).append(f)

    session_days = sorted(files_by_day.keys())
    for day in session_days:
        files_by_day[day].sort(key=extract_trial)

    manifest_rows = []
    ex_plots = []

    for day in session_days:
        for fname in files_by_day[day]:
            trial = extract_trial(fname)
            src_path = os.path.join(directory_path, fname)

            # load force
            with open(src_path, "rb") as f:
                data = pickle.load(f)
            force = _safe_force_from_pickle(data)

            # baseline-correct and handle NaNs
            f0 = np.asarray(force, float)
            if np.any(~np.isfinite(f0)):
                finite = np.isfinite(f0)
                repl = np.nanmedian(f0[finite]) if np.any(finite) else 0.0
                f0 = np.nan_to_num(f0, nan=repl)
            f0 = f0 - np.min(f0)  # min==0

            mean_before = float(np.mean(f0)) if f0.size else 0.0

            if mean_before <= 0.0:
                scaled = np.zeros_like(f0)
                scale_factor = 0.0
            else:
                scale_factor = float(target_mean) / mean_before
                scaled = f0 * scale_factor  # do NOT clip; we want mean to be exact

            out_name = f"{day}_trial{trial:02d}.npy"
            out_path = os.path.join(save_dir, out_name)

            if overwrite or (not os.path.exists(out_path)):
                np.save(out_path, scaled)

            if verbose:
                mean_after = float(np.mean(scaled)) if scaled.size else 0.0
                print(f"{day} trial {trial:02d}: mean_before={mean_before:.4f} "
                      f"→ target_mean={target_mean:.4f} scale={scale_factor:.6f} "
                      f"(mean_after={mean_after:.4f})  -> {out_name}")

            manifest_rows.append([
                day, trial, fname, out_name,
                f"{mean_before:.8f}",
                f"{target_mean:.8f}",
                f"{scale_factor:.8f}",
            ])

            # store a few examples to plot
            if plot_examples and len(ex_plots) < n_examples:
                ex_plots.append((day, trial, f0, scaled))

    # write manifest CSV
    manifest_path = os.path.join(save_dir, manifest_name)
    with open(manifest_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["day","trial","src_file","out_file","mean_before","target_mean","scale_factor"])
        w.writerows(manifest_rows)

    if verbose:
        print(f"\nWrote manifest: {manifest_path}")

    # example plots
    if plot_examples and ex_plots:
        cols = 3
        rows = int(np.ceil(len(ex_plots) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=False, sharey=False)
        axes = np.atleast_1d(axes).ravel()
        for i, (d, tr, raw, scaled) in enumerate(ex_plots):
            ax = axes[i]
            ax.plot(raw, label="baseline-corrected")
            ax.plot(scaled, label=f"scaled to mean {target_mean}")
            ax.set_title(f"{d} trial {tr:02d}")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Force (a.u.)")
            ax.legend(fontsize=8)
            ax.grid(True)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle(f"Examples (scaled to mean={target_mean})", y=1.02)
        plt.tight_layout()
        plt.show()