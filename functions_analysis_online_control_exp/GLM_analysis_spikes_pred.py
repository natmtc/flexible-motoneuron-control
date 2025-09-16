import os
import pickle
import numpy as np
from scipy.special import gammaln
import re
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_preprocess(file, folder_path, bin_size_ms, sampling_freq=2048):
    """
    Load one trial (pickle) and bin spikes for GLM input.

    Parameters
    ----------
    file : str
        Filename of the trial (e.g., 'S1_Session01_trial03.pkl').
    folder_path : str
        Directory containing the trial file.
    bin_size_ms : int or float
        Bin width in milliseconds used to aggregate spikes.
    sampling_freq : int
        Sampling frequency (Hz) of the raw signals in the file.

    Returns
    -------
    binned_spikes_T : np.ndarray, shape (bins, neurons)
        Spike counts per bin (transposed so time/bins is first dimension).
    data : dict
        The full dictionary loaded from the pickle file (unaltered).
    mnpair : list
        Value of 'Idx_MU_Pair' from the file (converted to list).

    """
    file_path = os.path.join(folder_path, file)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    data_MU = np.array(data['MU'].tolist())  # (neurons, samples)

    bin_size_samples = int(bin_size_ms * sampling_freq / 1000)
    binned_spikes = np.add.reduceat(
        data_MU,
        np.arange(0, data_MU.shape[1], bin_size_samples),
        axis=1
    )

    mnpair = data['Idx_MU_Pair'].tolist()
    return binned_spikes.T, data, mnpair  # (bins, neurons), dict, list


def create_glm_data(spike_data, target_idx):
    """
    Build GLM design for spike->spike (instantaneous population; no history).

    Parameters
    ----------
    spike_data : np.ndarray, shape (bins, neurons)
        Binned spike counts for all neurons (time along rows).
    target_idx : int
        Index of the target neuron to predict.

    Returns
    -------
    X : np.ndarray or None, shape (bins, predictors)
        Predictor matrix built from all *other* active neurons (excludes target).
    Y : np.ndarray or None, shape (bins,)
        Target spike counts for the selected neuron.

    Selection rules
    ---------------
    - "Active" neurons are those with >10 spikes across all bins.
    - If the target is not active, returns (None, None).
    - If the target's total spikes < 20, returns (None, None).

    """
    num_timebins, num_neurons = spike_data.shape

    # identify active neurons (>10 spikes total)
    active_neurons = np.where(spike_data.sum(axis=0) > 10)[0]
    if target_idx not in active_neurons:
        return None, None

    # exclude target + inactive neurons from predictors
    mask = np.logical_and(
        np.arange(num_neurons) != target_idx,
        np.isin(np.arange(num_neurons), active_neurons)
    )

    X = spike_data[:, mask]
    Y = spike_data[:, target_idx]


    return X, Y


def compute_pR2(y_true, y_pred, eps=1e-12):
    """
    Poisson deviance-based pseudo-R² (Perich et al.-style).

    Parameters
    ----------
    y_true : array-like
        Observed spike counts per bin.
    y_pred : array-like
        Predicted mean counts per bin (non-negative).
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    pR2 : float
        1 - (Deviance_model / Deviance_null). Higher is better.
        NaN if denominator is zero or not finite.

    Notes
    -----
    - Null model predicts a constant equal to the mean of `y_true`.
    - Uses the saturated model as reference for the deviance computation.
    """
    y = np.asarray(y_true, dtype=float)
    lam = np.clip(np.asarray(y_pred, dtype=float), eps, None)

    def logL(y, lam):
        return np.sum(y * np.log(lam) - lam - gammaln(y + 1))

    nz = y > 0
    ll_sat  = np.sum(y[nz] * np.log(y[nz])) - np.sum(y) - np.sum(gammaln(y + 1))
    ll_model = logL(y, lam)
    ll_null  = logL(y, np.full_like(y, y.mean()))
    denom = ll_sat - ll_null
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return 1.0 - (ll_sat - ll_model) / denom


def plot_pr2_violin(session_results,
                    title="Decoding Accuracy (pR²) by Session",
                    out_path=None,
                    palette_name="plasma",
                    point_jitter=0.2,
                    point_alpha=0.6,
                    ylim=(-1, 1.1),
                    figsize=(12, 6),
                    font_family="Arial"):
    """
    Plot per-session pR² distributions as violins with overlaid dots that are the individual neuron pR² values)
    """
    if not session_results:
        print("plot_pr2_violin: nothing to plot (empty session_results).")
        return None, None

    plt.figure(figsize=figsize)
    plt.rcParams.update({'font.family': font_family})

    # flatten data + simplify labels (keep numeric id if present)
    all_pR2s, all_labels, simplified = [], [], {}
    for s in session_results.keys():
        m = re.search(r'(\d+)', s)
        simplified[s] = str(int(m.group(1))) if m else s

    for s, vals in session_results.items():
        all_pR2s.extend(vals)
        all_labels.extend([simplified[s]] * len(vals))

    # ensure numeric order for sessions like "01,02,..."
    unique_sessions = sorted(set(all_labels),
                             key=lambda x: int(x) if x.isdigit() else x)

    # build palette
    import matplotlib as mpl
    cmap = mpl.cm.get_cmap(palette_name)
    N = max(len(unique_sessions) - 1, 1)
    colors = [cmap(i / N) for i in range(len(unique_sessions))]
    palette_dict = dict(zip(unique_sessions, colors))

    # plot
    sns.violinplot(x=all_labels, y=all_pR2s, order=unique_sessions,
                   inner='box', linewidth=1.2, palette=palette_dict, cut=0, zorder=0)
    sns.stripplot(x=all_labels, y=all_pR2s, order=unique_sessions,
                  color='k', size=3, jitter=point_jitter, alpha=point_alpha, zorder=1)

    ax = plt.gca()
    for sp in ax.spines.values():
        sp.set_linewidth(1.4)
    plt.tick_params(width=1.4, length=6, labelsize=25)
    plt.xlabel("Session", fontsize=25)
    plt.ylabel("Decoding Accuracy (pR²)", fontsize=25)
    if ylim is not None:
        plt.ylim(ylim)
    sns.despine(top=True, right=True)
    if title:
        plt.title(title, fontsize=18)
    plt.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')

    plt.show()
    return plt.gcf(), ax
