import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os, re, pickle
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

def extract_day(filename):
    """
Extract the day label (e.g. 'Day03') from a filename.

Args:
    filename (str): Filename containing a substring like 'DayXX'.

Returns:
    str or None: The matched day string, or None if no day pattern is found.
"""

    match = re.search(r"(Day\d{2})", filename)
    return match.group(1) if match else None  




def extract_trial(filename):
    """
Extract the trial number from a filename.

Args:
    filename (str): Filename containing a pattern like '_<trialNumber>'.

Returns:
    int: Extracted trial number, or 0 if no match is found.
"""

    match = re.search(r'_(\d+)[^0-9]', filename)  
    return int(match.group(1)) if match else 0


  

def create_force_plots_by_day(directory_path, fsamp, highlight_samples, extract_day, extract_trial):
    """
Create a grid of force traces grouped and color-coded by session day.

Args:
    directory_path (str): Path to the directory containing .pkl files.
    fsamp (float): Sampling frequency of the force signal (Hz).
    highlight_samples (int): Number of samples to highlight at the start (green)
        and end (red) of each trial.
    extract_day (callable): Function mapping filename -> day label (e.g. 'Day01').
    extract_trial (callable): Function mapping filename -> trial number.

Process:
    - Groups .pkl files by day and sorts trials within each day.
    - Assigns a unique background color to each day.
    - For each trial:
        * Loads the force trace.
        * Plots the full trace in blue.
        * Shades the subplot background with the day color.
        * Highlights the first `highlight_samples` in green and last in red.
    - Arranges all trials in a multi-panel matplotlib figure.
    - Removes unused subplot axes.

Returns:
    matplotlib.figure.Figure: The generated (but not shown/saved) figure.
"""

    # Group files by day
    pkl_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]
    files_by_day = {}
    for file in pkl_files:
        day = extract_day(file)
        if day is not None:
            files_by_day.setdefault(day, []).append(file)

    # Sort days and files within each day
    session_days = sorted(files_by_day.keys())
    for day in session_days:
        files_by_day[day].sort(key=extract_trial)
    
    # Flatten the sorted list
    sorted_pkl_files = [file for day in session_days for file in files_by_day[day]]

    # Colour map per day
    num_days = len(session_days)
    if num_days <= 20:
        cmap = cm.get_cmap("tab20")
        color_map = {day: cmap(i) for i, day in enumerate(session_days)}
    else:
        cmap = cm.get_cmap("viridis")
        color_map = {day: cmap(i / num_days) for i, day in enumerate(session_days)}

    # Layout
    num_files = len(sorted_pkl_files)
    cols = 6
    rows = int(np.ceil(num_files / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3), sharex=False, sharey=False)
    axes = axes.flatten()

    for i, file_name in enumerate(sorted_pkl_files):
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        data_FORCE = np.array(data['Force'].tolist())[0]
        time_axis = np.arange(len(data_FORCE)) / fsamp
        start_region = np.arange(highlight_samples)
        end_region = np.arange(len(data_FORCE) - highlight_samples, len(data_FORCE))

        day_info = extract_day(file_name)
        day_color = color_map.get(day_info, "white")

        ax = axes[i]
        ax.plot(time_axis, data_FORCE, color="blue", alpha=0.8)
        ax.axvspan(time_axis[0], time_axis[-1], color=day_color, alpha=0.3, zorder=-10)
        ax.plot(time_axis[start_region], data_FORCE[start_region], color="green", alpha=1)
        ax.plot(time_axis[end_region], data_FORCE[end_region], color="red", alpha=1)

        ax.set_title(file_name.split('.')[0], fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force")

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig




def plot_force_trials_block(directory_path, fsamp=2048, highlight_seconds=0, long_trial_thresh_s=19.0, y_max=0.2, font_family="Arial", tick_size=30, line_width=1.2, rect_edge_lw=3, save_path=None):
    """
Plot force trajectories in a 2×2 block layout for early vs late training days.

Layout:
    - Left block (cols 0–1): Day1 (top), Day2 (bottom)
    - Right block (cols 2–3): Day(-2) (top), Day(-1) (bottom)
    Each day occupies two columns; trials fill down the first column, then the second.

Args:
    directory_path (str): Folder containing .pkl trials with 'Force' data.
    fsamp (float): Sampling frequency in Hz.
    highlight_seconds (float): Duration (s) at start/end of each trial to slightly
        thicken the trace.
    long_trial_thresh_s (float): Duration threshold to mark “long” trials with a
        filled background.
    y_max (float): Upper limit of the y-axis (force).
    font_family (str): Matplotlib font family.
    tick_size (int): Tick label font size.
    line_width (float): Line width for force traces.
    rect_edge_lw (float): Line width for trial bounding rectangles.
    save_path (str or None): If provided, save the figure to this path.

Process:
    - Detects and groups trials by day, sorting by trial index.
    - Selects first two and last two days to fill the 2×2 day blocks.
    - For each trial:
        * Plots force vs time in black.
        * Draws a colored rectangle per day; fills it for long trials.
        * Optionally thickens the first/last `highlight_seconds` of the trace.
    - Styles axes (no titles/labels, formatted ticks) and hides empty subplots.

Returns:
    None (displays the generated figure; optionally saves it).
"""

    # ----------------------------
    # Style
    # ----------------------------
    plt.rcParams["font.family"] = font_family
    plt.rcParams["axes.labelsize"] = 0     # no axis labels
    plt.rcParams["axes.titlesize"] = 0     # no titles
    plt.rcParams["xtick.labelsize"] = tick_size
    plt.rcParams["ytick.labelsize"] = tick_size

    def _nice_fmt(x, pos):
        # drop .0 but keep other decimals
        if abs(x - round(x)) < 1e-9:
            return f"{int(round(x))}"
        s = f"{x:.2f}".rstrip("0").rstrip(".")
        return s
    fmt = FuncFormatter(_nice_fmt)

    # ----------------------------
    # Helpers
    # ----------------------------
    def extract_day(filename):
        m = re.search(r"(Day\d{2})", filename)
        return m.group(1) if m else None

    def extract_trial(filename):
        m = re.search(r'_(\d+)[^0-9]', filename)
        return int(m.group(1)) if m else 0

    def load_force(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        try:
            x = np.asarray(data["Force"]).squeeze()
            if x.ndim > 1:
                x = np.asarray(data["Force"].tolist())[0].squeeze()
        except Exception:
            x = np.asarray(data["Force"].tolist())[0].squeeze()
        return np.asarray(x, float)

    # ----------------------------
    # Collect files and group by day
    # ----------------------------
    pkl_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]
    files_by_day = {}
    for file in pkl_files:
        day = extract_day(file)
        if day:
            files_by_day.setdefault(day, []).append(file)

    session_days = sorted(files_by_day.keys())
    for d in session_days:
        files_by_day[d].sort(key=extract_trial)

    if len(session_days) < 4:
        raise RuntimeError("Need at least 4 distinct days in the folder.")

    left_days  = [session_days[0], session_days[1]]
    right_days = [session_days[-2], session_days[-1]]

    # Day colors
    num_days = len(session_days)
    if num_days <= 20:
        cmap = cm.get_cmap("tab20")
        day_color = {day: cmap(i % 20) for i, day in enumerate(session_days)}
    else:
        day_color = {day: cm.viridis(i / max(1, num_days - 1)) for i, day in enumerate(session_days)}

    # ----------------------------
    # Axis limits (Y fixed to 0..y_max)
    # ----------------------------
    y_min = 0.0

    # ----------------------------
    # Layout math
    # ----------------------------
    def rows_for_day(day):
        return int(np.ceil(len(files_by_day[day]) / 2)) or 1

    left_rows_top     = rows_for_day(left_days[0])
    left_rows_bottom  = rows_for_day(left_days[1])
    right_rows_top    = rows_for_day(right_days[0])
    right_rows_bottom = rows_for_day(right_days[1])

    left_total_rows  = left_rows_top + left_rows_bottom
    right_total_rows = right_rows_top + right_rows_bottom
    ROWS = max(left_total_rows, right_total_rows)
    COLS = 4

    fig, axes = plt.subplots(ROWS, COLS, figsize=(15, ROWS * 3), squeeze=False)

    # ----------------------------
    # Plotting helper
    # ----------------------------
    highlight_samples = int(highlight_seconds * fsamp)

    def plot_day(day, start_row, block_col0, block_col1):
        flist  = files_by_day[day]
        rcount = rows_for_day(day)
        dcolor = day_color.get(day, "black")

        for k, fname in enumerate(flist):
            r_local = k % rcount
            c_local = k // rcount
            if c_local > 1:
                c_local = 1
                r_local = min(r_local + rows_for_day(day), rcount - 1)

            r = start_row + r_local
            c = block_col0 if c_local == 0 else block_col1
            if r >= ROWS:
                continue

            ax = axes[r, c]
            force = load_force(os.path.join(directory_path, fname))
            t = np.arange(len(force)) / fsamp
            duration_s = t[-1] if len(t) else 0.0

            # ----- Draw background rectangle in data coords: x [0, t[-1]], y [0, y_max]
            rect_width = t[-1] if len(t) else 0.0
            is_long = duration_s > long_trial_thresh_s
            ax.add_patch(Rectangle(
                (0.0, y_min), rect_width, (y_max - y_min),
                fill=is_long,
                facecolor=(0.94, 0.94, 0.94) if is_long else "none",
                ec=dcolor, lw=rect_edge_lw, zorder=-5
            ))

            # Trace
            ax.plot(t, force, color="black", alpha=0.9, linewidth=line_width)

            # Start/end highlight
            if highlight_samples > 0 and len(force) > 0:
                hs = min(highlight_samples, len(force))
                ax.plot(t[:hs], force[:hs], color="black", linewidth=line_width + 0.15)
                ax.plot(t[-hs:], force[-hs:], color="black", linewidth=line_width + 0.15)

            # Fixed axis limits and formatting
            ax.set_ylim(y_min, y_max)
            # (Optional) unify x-limits, comment out next line if you want natural width
            # ax.set_xlim(0.0, max_x)

            for spine in ax.spines.values():
                spine.set_visible(False)

            # No labels/titles; ticks formatted
            ax.set_title(""); ax.set_xlabel(""); ax.set_ylabel("")
            ax.xaxis.set_major_formatter(fmt)
            ax.tick_params(axis='x', which='both', length=6, labelsize=tick_size)

            if c == 0:
                ax.set_yticks([0.0, 0.1, y_max])
                ax.yaxis.set_major_formatter(fmt)
                ax.tick_params(axis='y', which='both', length=6, labelsize=tick_size)
            else:
                ax.set_yticks([])

    # ----------------------------
    # Fill figure
    # ----------------------------
    plot_day(left_days[0],  start_row=0,                block_col0=0, block_col1=1)
    plot_day(left_days[1],  start_row=left_rows_top,    block_col0=0, block_col1=1)
    plot_day(right_days[0], start_row=0,                block_col0=2, block_col1=3)
    plot_day(right_days[1], start_row=right_rows_top,   block_col0=2, block_col1=3)

    # Hide empties
    for r in range(ROWS):
        for c in range(COLS):
            ax = axes[r, c]
            if not ax.has_data():
                ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()