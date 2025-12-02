import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, poisson

def load_mu_from_pickle(pkl_path):
    """
Load motoneuron spike matrix (MU) from a pickle file.

Args:
    pkl_path (str): Path to a pickle file containing a dict with key 'MU'.

Process:
    - Opens and loads the pickle dictionary.
    - Converts the stored MU object to a NumPy array.
    - Verifies that MU has shape (neurons, samples).

Returns:
    np.ndarray: MU array of shape (neurons, samples).

Raises:
    ValueError: If MU is not a 2D array.
"""

    """Return MU as (neurons x samples) from your pickle."""
    
    
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    MU = np.array(d['MU'].tolist())   # (neurons x samples)
    if MU.ndim != 2:
        raise ValueError(f"Expected 2D MU, got {MU.shape}")
    return MU



def get_first_file_from_session_map(session_files):
    """
Retrieve the first file entry from a session-to-files mapping.

Args:
    session_files (dict): Mapping {session_name: [file_paths...]}

Returns:
    tuple: (first_session_name, first_file_path)
"""

    first_session, first_files = list(session_files.items())[0]
    return first_session, first_files[0]




def bin_spikes(MU, fsamp, bin_ms):
    """
Bin spike trains into fixed-duration time bins.

Args:
    MU (np.ndarray): Binary spike matrix of shape (neurons, samples).
    fsamp (float): Sampling frequency in Hz.
    bin_ms (float): Bin width in milliseconds.

Process:
    - Converts bin width to samples.
    - Validates bin size.
    - Truncates MU to an integer number of bins.
    - Sums spikes within each bin.

Returns:
    np.ndarray or None:
        Binned spike counts of shape (neurons, n_bins),
        or None if the data are too short for a single bin.

Raises:
    ValueError: If bin_len <= 0.
"""

    bin_len = int(round((bin_ms/1000.0) * fsamp))
    if bin_len <= 0:
        raise ValueError("bin_len <= 0; check fsamp/bin_ms")
    n_bins = MU.shape[1] // bin_len
    if n_bins == 0:
        return None
    MUc = MU[:, :n_bins*bin_len]
    return MUc.reshape(MU.shape[0], n_bins, bin_len).sum(axis=2)



def fano_factor(counts):
    """
Compute the Fano factor of a count distribution.

Args:
    counts (array-like): Sequence of spike counts or event counts.

Process:
    - Converts to a NumPy array.
    - Computes mean (λ) and sample variance.
    - Returns variance / mean, or NaN if mean is zero.

Returns:
    float: Fano factor (variance-to-mean ratio).
"""

    counts = np.asarray(counts)
    lam = counts.mean()
    return np.nan if lam == 0 else counts.var(ddof=1) / lam



def chisq_poisson_gof(counts, min_exp=5, tail_sd=10):
    """
Perform a chi-square goodness-of-fit test for whether count data follow
a Poisson distribution with λ estimated as the sample mean.

The function:
- Estimates λ = mean(counts).
- Builds observed and expected frequency tables up to either max(count)
  or lam + tail_sd * sqrt(lam).
- Pools right-tail bins so that all expected counts ≥ min_exp.
- Computes the chi-square statistic with 1 degree of freedom (λ estimated).

Returns:
    chi2 (float): Chi-square test statistic.
    p (float): p-value for Poisson GOF.
    details (dict): Dictionary with λ̂, pooled observed counts, and expected counts.
"""
    
    x = np.asarray(counts).ravel().astype(int)
    if x.size == 0:
        raise ValueError("counts is empty")
    n   = x.size
    lam = x.mean()

    K = max(x.max(), int(np.ceil(lam + tail_sd*np.sqrt(max(lam,1e-12)))))
    obs = np.bincount(x, minlength=K+1).astype(float)
    exp = n * poisson.pmf(np.arange(K+1), lam)
    exp[-1] += (n - exp.sum())   # fix rounding

    # pool from right until all expected >= min_exp
    while (exp < min_exp).any() and len(exp) > 1:
        i = np.where(exp < min_exp)[0][-1]
        exp[i-1] += exp[i]; obs[i-1] += obs[i]
        exp = np.delete(exp, i); obs = np.delete(obs, i)

    chi2, p = chisquare(f_obs=obs, f_exp=exp, ddof=1)  # ddof=1 (λ estimated)
    return chi2, p, {"lambda_hat": lam, "obs": obs, "exp": exp}




def plot_hist_vs_poisson(spike_counts, bin_ms, session_label="Session", neuron_idx=0):
    """
Plot observed spike-count histogram with a Poisson(μ) overlay.

Args:
    spike_counts (array-like): Spike counts per bin for a single neuron.
    bin_ms (float): Bin width in milliseconds (for labeling only).
    session_label (str): Session name to display in the plot title.
    neuron_idx (int): Neuron index to display in the plot title.

Process:
    - Computes observed histogram of spike counts.
    - Fits μ = mean(spike_counts) and computes expected Poisson frequencies.
    - Plots observed vs. expected counts side-by-side.

Returns:
    float: Estimated Poisson mean μ.
"""

    spike_counts = np.asarray(spike_counts).astype(int)
    mu = spike_counts.mean()
    kmax = int(spike_counts.max())
    k = np.arange(0, kmax + 1)
    obs, _ = np.histogram(spike_counts, bins=np.arange(0, kmax+2))
    exp = poisson.pmf(k, mu) * len(spike_counts)

    plt.figure(figsize=(6,4))
    plt.bar(k-0.2, obs, width=0.4, label="Observed")
    plt.bar(k+0.2, exp, width=0.4, label=f"Poisson(μ={mu:.2f})")
    plt.xlabel("Spikes per bin"); plt.ylabel("Frequency")
    plt.title(f"{session_label} • Neuron {neuron_idx} • bin={bin_ms} ms")
    plt.legend(); plt.tight_layout(); plt.show()
    return mu




def evaluate_one_neuron(MU, fsamp, bin_ms, neuron_idx=0, plot=True, session_label="Session"):
    """
Evaluate spiking variability and Poisson goodness-of-fit for a single neuron.

Args:
    MU (np.ndarray): Spike matrix of shape (neurons, samples).
    fsamp (float): Sampling frequency in Hz.
    bin_ms (float): Bin width in milliseconds.
    neuron_idx (int): Index of the neuron to analyze.
    plot (bool): If True, plot histogram with Poisson overlay.
    session_label (str): Label used in the plot title.

Process:
    - Bins the spike trains.
    - Extracts spike counts for the selected neuron.
    - Computes Fano factor.
    - Performs χ² goodness-of-fit test for Poisson(λ̂).
    - Optionally plots observed vs Poisson histogram.

Returns:
    dict: {
        "fano": Fano factor,
        "chi2": chi-square statistic,
        "p": p-value,
        "lambda_hat": estimated Poisson mean,
        "counts": spike-count vector
    }

Raises:
    ValueError: If MU is too short to form at least one bin.
"""

    counts_mat = bin_spikes(MU, fsamp, bin_ms)
    if counts_mat is None:
        raise ValueError("Not enough samples to form even one bin.")
    spike_counts = counts_mat[neuron_idx]
    ff = fano_factor(spike_counts)
    chi2, p, info = chisq_poisson_gof(spike_counts, min_exp=5)
    mu = None
    if plot:
        mu = plot_hist_vs_poisson(spike_counts, bin_ms, session_label, neuron_idx)
    return {"fano": ff, "chi2": chi2, "p": p, "lambda_hat": info["lambda_hat"], "counts": spike_counts}




def summarize_poisson_by_bins(MU, fsamp, bin_list_ms):
    """
Summarize Poisson statistics across neurons for multiple bin widths.

For each bin size in `bin_list_ms`, the function:
    - Bins MU spike trains.
    - Computes per-neuron Fano factor.
    - Performs χ² Poisson GOF test.
    - Extracts λ̂ (mean spike count per bin).
    - Aggregates across neurons:
        * median Fano factor
        * fraction of neurons with p > 0.05 (fail to reject Poisson)
        * median λ̂

Args:
    MU (np.ndarray): Spike matrix of shape (neurons, samples).
    fsamp (float): Sampling frequency in Hz.
    bin_list_ms (list): List of bin widths in milliseconds.

Returns:
    list of tuples:
        (bin_ms, median_fano, frac_pass, median_lambda)

Notes:
    Skips bin sizes where not enough data exist to form bins.
"""

    summary = []
    for bin_ms in bin_list_ms:
        counts_mat = bin_spikes(MU, fsamp, bin_ms)
        if counts_mat is None:
            print(f"bin={bin_ms} ms: too short to form even 1 bin; skipping.")
            continue
        fanos, pvals, lambdas = [], [], []
        for i in range(counts_mat.shape[0]):
            c = counts_mat[i]
            if c.size < 5:
                continue
            lam = c.mean()
            fanos.append(fano_factor(c))
            _, p, info = chisq_poisson_gof(c, min_exp=5)
            pvals.append(p)
            lambdas.append(info["lambda_hat"])
        if not pvals:
            print(f"bin={bin_ms} ms: not enough bins or neurons; skipping.")
            continue
        med_fano = np.nanmedian(fanos)
        frac_pass = float(np.mean(np.array(pvals) > 0.05))
        med_lambda = np.nanmedian(lambdas)
        summary.append((bin_ms, med_fano, frac_pass, med_lambda))
    return summary




def print_poisson_summary(summary):
    print("\n=== Poisson-ness by bin size ===")
    print("bin_ms | median Fano | % neurons p>0.05 | median λ")
    for b, fano, frac, lam in summary:
        print(f"{b:6d} | {fano:11.3f} | {100*frac:14.1f}% | {lam:8.3f}")
