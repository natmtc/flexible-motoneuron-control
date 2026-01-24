"""
Firing Rate Distribution Comparison: Experimental vs Simulated Data

This module provides functions to compare firing rate distributions between
experimental recordings and simulation results.

Two comparison modes:
1. By TRIAL: Compare individual trials for a specific day
2. By DAY: Aggregate all trials per day across multiple days

Author: Your Name
Date: January 2026
"""

import os
import re
import pickle
import numpy as np
import h5py
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


# =============================================================================
# SHARED UTILITY FUNCTIONS
# =============================================================================

def avg_fr_from_spike_times(spike_indices: np.ndarray, fs: float) -> Optional[float]:
    """
    Compute average firing rate from spike time indices.
    
    Args:
        spike_indices (np.ndarray): Indices where spikes occur (sample numbers)
        fs (float): Sampling frequency in Hz
    
    Returns:
        Optional[float]: Average firing rate in Hz, or None if <2 spikes
    
    Method:
        Computes mean ISI and returns 1/ISI
    """
    if spike_indices.size < 2:
        return None
    isi = np.diff(spike_indices) / fs
    mean_isi = isi.mean()
    return None if mean_isi <= 0 else 1.0 / mean_isi


def parse_exp_day_trial(filename: str) -> Optional[Tuple[int, int]]:
    """
    Extract day and trial number from experimental filename.
    
    Args:
        filename (str): Filename like 'E3_MB01_Day03High_18.pkl'
    
    Returns:
        Optional[Tuple[int, int]]: (day_number, trial_number) or None if parsing fails
    
    Examples:
        'E3_MB01_Day03High_18.pkl' → (3, 18)
        'E3_MB02_Day01_trial12.pkl' → (1, 12)
    """
    mday = re.search(r"Day0*(\d+)", filename)
    mtr = re.search(r"(?:_trial|_)(\d+)(?!\d)", filename)
    
    if not mday or not mtr:
        return None
    
    return int(mday.group(1)), int(mtr.group(1))


def extract_day(filename: str) -> Optional[int]:
    """
    Extract day number from filename.
    
    Args:
        filename (str): Filename containing 'DayXX' pattern
    
    Returns:
        Optional[int]: Day number (e.g., 'Day03' → 3) or None
    """
    match = re.search(r"Day(\d{2})", filename)
    return int(match.group(1)) if match else None


def extract_trial(filename: str) -> int:
    """
    Extract trial number from filename.
    
    Args:
        filename (str): Filename with trial number pattern
    
    Returns:
        int: Trial number, or 0 if not found
    """
    match = re.search(r'_(\d+)[^0-9]', filename)
    return int(match.group(1)) if match else 0


def get_MU_array(data: dict) -> np.ndarray:
    """
    Safely extract MU spike array from pickle data.
    
    Args:
        data (dict): Loaded pickle dictionary containing 'MU' key
    
    Returns:
        np.ndarray: MU array with shape (neurons, samples)
    
    Notes:
        Handles both direct numpy arrays and object arrays that need conversion
    """
    try:
        MU = np.array(data["MU"])
        if MU.dtype == object:
            MU = np.array(data["MU"].tolist())
        return MU
    except Exception:
        return np.array(data["MU"].tolist())


def read_jld2_array(h: h5py.File, key: str) -> Optional[np.ndarray]:
    """
    Read numeric arrays from HDF5/JLD2 files with multiple fallback strategies.
    
    Args:
        h (h5py.File): Open HDF5 file handle
        key (str): Dataset key to read (e.g., 'avg_rates_per_neuron')
    
    Returns:
        Optional[np.ndarray]: Numeric array or None if not found
    
    Notes:
        Tries multiple access patterns:
        1. Direct dataset access
        2. Dataset with '/value' suffix
        3. Group with nested 'value' dataset
    """
    # Try direct dataset access
    if key in h and isinstance(h[key], h5py.Dataset) and h[key].dtype.kind in "biuf":
        return h[key][:]
    
    # Try {key}/value pattern
    value_key = f"{key}/value"
    if value_key in h and isinstance(h[value_key], h5py.Dataset):
        d = h[value_key]
        if d.dtype.kind in "biuf":
            return d[:]
    
    # Try group with value member
    if key in h and isinstance(h[key], h5py.Group):
        g = h[key]
        if "value" in g and isinstance(g["value"], h5py.Dataset):
            if g["value"].dtype.kind in "biuf":
                return g["value"][:]
    
    return None


# =============================================================================
# TRIAL-BY-TRIAL COMPARISON (Single Day)
# =============================================================================

def collect_exp_by_trial(exp_dir: str, target_day: int, fs: float = 2048, 
                         fr_cap: float = 80.0) -> Dict[int, List[float]]:
    """
    Collect experimental firing rates grouped by TRIAL for a specific day.
    
    Args:
        exp_dir (str): Directory containing experimental .pkl files
        target_day (int): Which day to extract (e.g., 1, 2, 3)
        fs (float): Sampling frequency in Hz
        fr_cap (float): Maximum firing rate to include (Hz)
    
    Returns:
        Dict[int, List[float]]: {trial_number: [firing_rates...]}
    
    Process:
        - Scans directory for .pkl files
        - Filters for specified day
        - Computes firing rate for each neuron
        - Groups by trial number
    """
    by_trial = {}
    
    for filename in os.listdir(exp_dir):
        if not filename.endswith(".pkl"):
            continue
        
        parsed = parse_exp_day_trial(filename)
        if parsed is None:
            continue
        
        day, trial = parsed
        
        # Filter for target day only
        if day != target_day:
            continue
        
        # Load data
        filepath = os.path.join(exp_dir, filename)
        with open(filepath, "rb") as fh:
            data = pickle.load(fh)
        
        MU = get_MU_array(data)
        
        # Compute firing rates for all neurons
        firing_rates = []
        for neuron_idx in range(MU.shape[0]):
            spike_idx = np.flatnonzero(MU[neuron_idx] == 1)
            fr = avg_fr_from_spike_times(spike_idx, fs)
            
            if fr is not None and fr <= fr_cap:
                firing_rates.append(fr)
        
        if firing_rates:
            by_trial.setdefault(trial, []).extend(firing_rates)
    
    return by_trial


def collect_sim_by_trial(sim_dir: str, fr_cap: float = 80.0) -> Dict[int, np.ndarray]:
    """
    Collect simulated firing rates grouped by trial.
    
    Args:
        sim_dir (str): Directory containing simulation .jld2 files
        fr_cap (float): Maximum firing rate to include (Hz)
    
    Returns:
        Dict[int, np.ndarray]: {trial_number: firing_rates_array}
    
    Expected filename format:
        force{trial}_firing_rates.jld2
        Example: force1_firing_rates.jld2, force12_firing_rates.jld2
    
    Dataset names tried:
        - 'avg_rates_per_neuron'
        - 'firing_rates'
    """
    by_trial = {}
    
    if not os.path.isdir(sim_dir):
        return by_trial
    
    for filename in os.listdir(sim_dir):
        if not (filename.endswith(".jld2") or filename.endswith(".h5")):
            continue
        
        # Parse trial number from filename
        match = re.fullmatch(r"force(\d+)_firing_rates\.(?:jld2|h5)", filename)
        if not match:
            continue
        
        trial = int(match.group(1))
        filepath = os.path.join(sim_dir, filename)
        
        try:
            with h5py.File(filepath, "r") as h:
                # Try multiple possible dataset names
                rates = read_jld2_array(h, "avg_rates_per_neuron")
                if rates is None:
                    rates = read_jld2_array(h, "firing_rates")
                
                if rates is None:
                    continue
                
                # Filter and validate
                rates = np.asarray(rates, float)
                rates = rates[np.isfinite(rates)]
                rates = rates[rates <= fr_cap]
                
                if rates.size > 0:
                    by_trial[trial] = rates
                    
        except Exception as e:
            print(f"Warning: Could not open {filepath} → {e}")
    
    return by_trial


def plot_trial_by_trial_comparison(exp_data: Dict[int, List[float]], 
                                   sim_data: Dict[int, np.ndarray],
                                   bins: np.ndarray = np.linspace(0, 40, 21),
                                   day_label: str = "Day 01",
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Create trial-by-trial comparison plots.
    
    Args:
        exp_data (dict): {trial: [firing_rates]} from experiment
        sim_data (dict): {trial: firing_rates_array} from simulation
        bins (np.ndarray): Histogram bin edges
        day_label (str): Label for the day being plotted
        save_path (str, optional): Path to save figure
    
    Returns:
        matplotlib.figure.Figure: Created figure
    
    Layout:
        - One subplot per trial
        - 5 columns per row
        - Blue = experimental, Red = simulated
    """
    trials = sorted(set(exp_data.keys()) | set(sim_data.keys()))
    
    if not trials:
        raise ValueError("No trials found in experimental or simulation data!")
    
    # Create subplot grid
    cols = 5
    rows = int(np.ceil(len(trials) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = np.atleast_1d(axes).ravel()
    
    # Plot each trial
    for i, trial in enumerate(trials):
        ax = axes[i]
        
        # Plot experimental data (blue)
        if trial in exp_data:
            ax.hist(exp_data[trial], bins=bins, density=True, alpha=0.6,
                    edgecolor="black", linewidth=1.2, label="Experimental", 
                    color='blue')
        
        # Plot simulated data (red)
        if trial in sim_data:
            ax.hist(sim_data[trial], bins=bins, density=True, alpha=0.45,
                    edgecolor="black", linewidth=1.0, label="Simulated", 
                    color='red')
        
        ax.set_title(f"{day_label} — Trial {trial:02d}")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel("Avg firing rate (Hz)")
        ax.set_ylabel("Density")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=10)
    
    # Remove empty axes
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    # Add figure-level legend
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", fontsize=12)
            break
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


# =============================================================================
# DAY-BY-DAY COMPARISON (Multiple Days, Aggregate Trials)
# =============================================================================

def collect_exp_by_day(exp_dir: str, fs: float = 2048, 
                       fr_cap: float = 80.0) -> Dict[int, List[float]]:
    """
    Collect experimental firing rates grouped by DAY (all trials pooled).
    
    Args:
        exp_dir (str): Directory containing experimental .pkl files
        fs (float): Sampling frequency in Hz
        fr_cap (float): Maximum firing rate to include (Hz)
    
    Returns:
        Dict[int, List[float]]: {day_number: [all_firing_rates_that_day]}
    
    Process:
        - Scans all .pkl files
        - Groups by day number
        - Pools all trials for each day
    """
    by_day = {}
    
    pkl_files = sorted(
        [f for f in os.listdir(exp_dir) if f.endswith('.pkl')],
        key=lambda x: (extract_day(x), extract_trial(x))
    )
    
    for filename in pkl_files:
        day = extract_day(filename)
        if day is None:
            continue
        
        filepath = os.path.join(exp_dir, filename)
        with open(filepath, 'rb') as fh:
            data = pickle.load(fh)
        
        MU = get_MU_array(data)
        
        # Compute firing rates for all neurons
        for neuron_idx in range(MU.shape[0]):
            spike_idx = np.where(MU[neuron_idx] == 1)[0]
            fr = avg_fr_from_spike_times(spike_idx, fs)
            
            if fr is not None and fr <= fr_cap:
                by_day.setdefault(day, []).append(fr)
    
    return by_day


def load_sim_by_day(jld2_file_paths: List[str]) -> Dict[int, np.ndarray]:
    """
    Load simulated firing rates from JLD2 files, organized by day.
    
    Args:
        jld2_file_paths (list): List of paths to .jld2 files
    
    Returns:
        Dict[int, np.ndarray]: {day_number: firing_rates_array}
    
    Expected filename pattern:
        force{day}_firing_rates.jld2
        Example: force1_firing_rates.jld2, force14_firing_rates.jld2
    
    Dataset name:
        'avg_rates_per_neuron'
    """
    sim_by_day = {}
    
    for filepath in jld2_file_paths:
        # Extract day from filename
        filename = os.path.basename(filepath)
        match = re.search(r'force(\d+)_firing_rates\.jld2', filename)
        
        if not match:
            print(f"Warning: Filename pattern not recognized: {filename}")
            continue
        
        day = int(match.group(1))
        
        try:
            with h5py.File(filepath, 'r') as f:
                if 'avg_rates_per_neuron' in f:
                    rates = f['avg_rates_per_neuron'][:]
                    sim_by_day[day] = rates
                    print(f"  Loaded Day {day}: {len(rates)} neurons")
                else:
                    print(f"  Warning: 'avg_rates_per_neuron' not found in {filepath}")
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
    
    return sim_by_day


def plot_day_by_day_comparison(exp_dir: str, 
                               jld2_file_paths: List[str],
                               fs: float = 2048,
                               fr_cap: float = 80.0,
                               bins: np.ndarray = np.linspace(0, 40, 21),
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create day-by-day comparison plots (all trials pooled per day).
    
    Args:
        exp_dir (str): Directory with experimental .pkl files
        jld2_file_paths (list): Paths to simulation .jld2 files
        fs (float): Sampling frequency for experimental data (Hz)
        fr_cap (float): Maximum firing rate to include (Hz)
        bins (np.ndarray): Histogram bin edges
        save_path (str, optional): Path to save figure
    
    Returns:
        matplotlib.figure.Figure: Created figure
    
    Layout:
        - One subplot per day
        - 4 columns per row
        - Blue = experimental (all trials pooled)
        - Red = simulated
    """
    # Load data
    print("Loading experimental data...")
    exp_data = collect_exp_by_day(exp_dir, fs=fs, fr_cap=fr_cap)
    
    print("Loading simulated data...")
    sim_data = load_sim_by_day(jld2_file_paths)
    
    # Create subplot grid
    n_days = len(exp_data)
    n_cols = 4
    n_rows = int(np.ceil(n_days / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    
    # Plot each day
    for i, (day, exp_rates) in enumerate(sorted(exp_data.items())):
        ax = axes[i]
        
        # Plot experimental (blue)
        ax.hist(exp_rates, bins=bins, color='blue', alpha=0.4,
                label='Experimental', density=True, 
                edgecolor='black', linewidth=1.2)
        
        # Plot simulated (red)
        if day in sim_data:
            ax.hist(sim_data[day], bins=bins, color='red', alpha=0.4,
                    label='Simulated', density=True, 
                    edgecolor='black', linewidth=1.2)
        
        ax.set_title(f"Day {day:02d}")
        ax.set_xlabel('Avg firing rate (Hz)', fontsize=16)
        ax.set_ylabel('Density', fontsize=16)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


# =============================================================================
# CONVENIENCE WRAPPERS
# =============================================================================

def compare_by_trial(exp_folder: str, sim_folder: str, target_day: int = 2,
                    fs: float = 2048, fr_cap: float = 80.0,
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Convenience function: Complete trial-by-trial comparison workflow.
    
    Args:
        exp_folder (str): Experimental data directory
        sim_folder (str): Simulation data directory
        target_day (int): Which day to analyze
        fs (float): Sampling frequency (Hz)
        fr_cap (float): Max firing rate (Hz)
        save_path (str, optional): Where to save figure
    
    Returns:
        matplotlib.figure.Figure: Comparison figure
    """
    print(f"=== Trial-by-Trial Comparison (Day {target_day}) ===\n")
    
    print("Collecting experimental data...")
    exp_data = collect_exp_by_trial(exp_folder, target_day, fs=fs, fr_cap=fr_cap)
    print(f"  Found {len(exp_data)} experimental trials")
    
    print("Collecting simulation data...")
    sim_data = collect_sim_by_trial(sim_folder, fr_cap=fr_cap)
    print(f"  Found {len(sim_data)} simulation trials")
    
    print("\nCreating plots...")
    bins = np.linspace(0, 40, 21)
    fig = plot_trial_by_trial_comparison(
        exp_data, sim_data, bins=bins, 
        day_label=f"Day {target_day:02d}",
        save_path=save_path
    )
    
    print("✓ Trial-by-trial comparison complete!")
    return fig


def compare_by_day(exp_folder: str, sim_file_paths: List[str],
                  fs: float = 2048, fr_cap: float = 80.0,
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Convenience function: Complete day-by-day comparison workflow.
    
    Args:
        exp_folder (str): Experimental data directory
        sim_file_paths (list): List of simulation .jld2 file paths
        fs (float): Sampling frequency (Hz)
        fr_cap (float): Max firing rate (Hz)
        save_path (str, optional): Where to save figure
    
    Returns:
        matplotlib.figure.Figure: Comparison figure
    """
    print("=== Day-by-Day Comparison ===\n")
    
    fig = plot_day_by_day_comparison(
        exp_folder, sim_file_paths, 
        fs=fs, fr_cap=fr_cap,
        save_path=save_path
    )
    
    print("\n✓ Day-by-day comparison complete!")
    return fig

