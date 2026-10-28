import os, re, math, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from scipy.signal import convolve, windows

def plot_fr_trajectories_block(directory_path, output_path=None, fsamp_spikes_key='fsamp', force_fs=2048, long_trial_thresh_s=19.0, clip_thresh=35,
    smooth_window_s=0.30, xlim=(0, 35), ylim=(0, 35), tick_locs=(0, 15, 30), font_family="Arial", tick_size=22, line_width=2.0, rect_edge_lw=4):
    """
Plot 2D firing-rate trajectories for MU pairs in a 2×2 day block layout.

Layout:
    - Left block (cols 0–1): earliest 2 days (top = Day1, bottom = Day2)
    - Right block (cols 2–3): last 2 days (top = Day(-2), bottom = Day(-1))
    Each day occupies two columns; trials fill down the first column, then the second.

Args:
    directory_path (str): Folder containing .pkl trials with 'MU', 'Idx_MU_Pair', and 'Force'.
    output_path (str or None): If provided, save the figure to this path.
    fsamp_spikes_key (str): Key in the pickle dict for spike sampling frequency.
    force_fs (float): Sampling frequency of the force signal (Hz), used for duration.
    long_trial_thresh_s (float): Duration threshold (s) to mark “long” trials with a gray background.
    clip_thresh (float or None): Max firing rate (Hz) to clip trajectories; if None, no clipping.
    smooth_window_s (float): Hanning window width (s) for spike-to-FR smoothing.
    xlim (tuple): x-axis limits for firing rate (Hz) of neuron 1.
    ylim (tuple): y-axis limits for firing rate (Hz) of neuron 2.
    tick_locs (tuple): Tick locations for both axes.
    font_family (str): Matplotlib font family.
    tick_size (int): Font size for tick labels.
    line_width (float): Line width of FR trajectories.
    rect_edge_lw (float): Line width of day-colored bounding rectangles.

Process:
    - Collects valid trials per day and picks the first two and last two days.
    - Converts spike trains of the MU pair into smoothed firing rates.
    - Clips firing rates, if requested.
    - Plots FR trajectories (neuron1 vs neuron2) with day-colored frames and
      gray fill for long trials.
    - Uses consistent axis limits, ticks, and minimalist styling.

Returns:
    None (displays the figure and optionally saves it).
"""


    # ---- style ----
    plt.rcParams["font.family"]    = font_family
    plt.rcParams["axes.labelsize"] = 0
    plt.rcParams["axes.titlesize"] = 0
    plt.rcParams["xtick.labelsize"] = tick_size
    plt.rcParams["ytick.labelsize"] = tick_size

    def nice_fmt(x, _):
        return f"{int(round(x))}" if abs(x-round(x)) < 1e-9 else f"{x:.2f}".rstrip("0").rstrip(".")
    fmt = FuncFormatter(nice_fmt)

    # ---- helpers ----
    def extract_day(filename):
        m = re.search(r"(Day\d{2})", filename)
        return m.group(1) if m else None

    def extract_trial(filename):
        m = re.search(r'_(\d+)[^0-9]', filename)
        return int(m.group(1)) if m else 0

    def conv_spikes(spike_mat, pos1, pos2, fs, window_s=smooth_window_s):
        wN = max(1, int(round(window_s * fs)))
        if wN % 2 == 0: wN += 1
        w = windows.hann(wN); w /= np.sum(w)
        s1 = convolve(spike_mat[pos1, :], w, mode='same') * fs
        s2 = convolve(spike_mat[pos2, :], w, mode='same') * fs
        return s1, s2

    def load_force_forceonly(d):
        try:
            x = np.asarray(d["Force"]).squeeze()
            if x.ndim > 1:
                x = np.asarray(d["Force"].tolist())[0].squeeze()
        except Exception:
            x = np.asarray(d["Force"].tolist())[0].squeeze()
        return np.asarray(x, float)

    # ---- collect files ----
    pkl_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]
    files_by_day = {}
    for fname in pkl_files:
        try:
            with open(os.path.join(directory_path, fname), 'rb') as f:
                d = pickle.load(f)
            if 'MU' in d and 'Idx_MU_Pair' in d and 'Force' in d:
                day = extract_day(fname)
                if day:
                    files_by_day.setdefault(day, []).append(fname)
        except Exception:
            continue

    session_days = sorted(files_by_day.keys())
    if len(session_days) < 4:
        print("Not enough valid days with data.")
        return

    for dkey in session_days:
        files_by_day[dkey].sort(key=extract_trial)

    left_days  = [session_days[0], session_days[1]]
    right_days = [session_days[-2], session_days[-1]]

    # colors per day
    if len(session_days) <= 20:
        cmap = cm.get_cmap("tab20")
        day_color = {day: cmap(i % 20) for i, day in enumerate(session_days)}
    else:
        day_color = {day: cm.viridis(i / max(1, len(session_days)-1)) for i, day in enumerate(session_days)}

    # ---- layout ----
    def rows_for_day(day):
        return int(math.ceil(len(files_by_day[day]) / 2.0)) or 1

    l_top   = rows_for_day(left_days[0])
    l_bot   = rows_for_day(left_days[1])
    r_top   = rows_for_day(right_days[0])
    r_bot   = rows_for_day(right_days[1])

    ROWS = max(l_top + l_bot, r_top + r_bot)
    COLS = 4
    fig, axes = plt.subplots(ROWS, COLS, figsize=(16, ROWS * 3), squeeze=False)

    # ---- per-day plotting ----
    def plot_day(day, start_row, col0, col1):
        flist  = files_by_day[day]
        rcount = rows_for_day(day)
        dcolor = day_color.get(day, "black")

        for k, fname in enumerate(flist):
            r_local = k % rcount
            c_local = k // rcount
            if c_local > 1:
                c_local = 1
                r_local = min(r_local + rcount, rcount-1)

            r = start_row + r_local
            c = col0 if c_local == 0 else col1
            if r >= ROWS: continue

            ax = axes[r, c]
            with open(os.path.join(directory_path, fname), 'rb') as f:
                d = pickle.load(f)

            # "not good" via force duration
            force = load_force_forceonly(d)
            duration_s = (len(force) / float(force_fs)) if force.size else 0.0
            is_long = duration_s > long_trial_thresh_s

            # spikes -> FR
            MU = np.asarray(d['MU']).astype(float)
            if MU.ndim == 1:
                MU = np.asarray(d['MU'].tolist()).astype(float)
            n1, n2 = d['Idx_MU_Pair']
            fs_spk = d.get(fsamp_spikes_key, force_fs)
            s1, s2 = conv_spikes(MU, n1, n2, fs=fs_spk)
            if clip_thresh is not None:
                s1 = np.clip(s1, 0, clip_thresh)
                s2 = np.clip(s2, 0, clip_thresh)

            # background rectangle; gray if not-good
            ax.add_patch(Rectangle(
                (xlim[0], ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0],
                fill=is_long, facecolor=(0.94, 0.94, 0.94) if is_long else "none",
                ec=dcolor, lw=rect_edge_lw, zorder=-10
            ))

            # trajectory
            ax.plot(s1, s2, color="black", linewidth=line_width)

            # limits
            ax.set_xlim(xlim); ax.set_ylim(ylim)

            # ticks on ALL axes (marks), but labels only bottom row & first col
            ax.set_xticks(list(tick_locs))
            ax.set_yticks(list(tick_locs))

            is_bottom    = (r == ROWS - 1)
            is_first_col = (c == 0)

            # show tick marks everywhere
            ax.tick_params(axis='x', length=6)
            ax.tick_params(axis='y', length=6)

            # x labels: only bottom row
            if is_bottom:
                ax.xaxis.set_major_formatter(fmt)
                ax.tick_params(axis='x', labelsize=tick_size, labelbottom=True)
            else:
                # keep marks, hide numbers
                ax.tick_params(axis='x', labelbottom=False)

            # y labels: only first column
            if is_first_col:
                ax.yaxis.set_major_formatter(fmt)
                ax.tick_params(axis='y', labelsize=tick_size, labelleft=True)
            else:
                ax.tick_params(axis='y', labelleft=False)

            # clean spines & no axis titles
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("")

    # fill figure
    plot_day(left_days[0], start_row=0,    col0=0, col1=1)
    plot_day(left_days[1], start_row=l_top,col0=0, col1=1)
    plot_day(right_days[0],start_row=0,    col0=2, col1=3)
    plot_day(right_days[1],start_row=r_top,col0=2, col1=3)

    # hide empties
    for r in range(ROWS):
        for c in range(COLS):
            ax = axes[r, c]
            if not ax.has_data():
                ax.set_visible(False)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()



