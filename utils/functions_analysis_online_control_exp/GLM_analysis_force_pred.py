import os
import pickle
import numpy as np
from scipy.special import gammaln
import re
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_trial_force(file, folder_path, bin_size_ms, fsamp=2048):
    """
    Load a single trial (pickle), bin MU spikes and force to a common bin size.

    Parameters
    ----------
    file : str
        Filename of the trial (e.g., 'S1_session01_trial03.pkl').
    folder_path : str
        Directory containing the trial file.
    bin_size_ms : int or float
        Bin width in milliseconds used to aggregate spikes/force.
    fsamp : int
        Sampling frequency (Hz) of the raw MU/force signals in the file.

    Returns
    -------
    binned_spikes : np.ndarray
        Array of shape (neurons, bins) with spike counts per bin.
    binned_force : np.ndarray
        Array of shape (bins,) with mean force per bin.
    mnpair : list or None
        Optional metadata 'Idx_MU_Pair' if present in the file; otherwise None.

    """
    file_path = os.path.join(folder_path, file)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    force = np.array(data['Force'])
    if force.ndim == 2:
        force = force.squeeze()
    data_MU = np.array(data['MU'].tolist())

    bin_size_samples = int(bin_size_ms * fsamp / 1000)
    num_timepoints   = force.shape[0]
    num_bins         = num_timepoints // bin_size_samples
    if num_bins <= 0:
        return None, None, None

    # spikes: sum within each bin
    binned_spikes = np.add.reduceat(
        data_MU[:, :num_bins * bin_size_samples],
        np.arange(0, num_bins * bin_size_samples, bin_size_samples),
        axis=1
    )
    # force: average within each bin
    binned_force = np.add.reduceat(
        force[:num_bins * bin_size_samples],
        np.arange(0, num_bins * bin_size_samples, bin_size_samples)
    ) / bin_size_samples

    mnpair = data.get('Idx_MU_Pair', None)
    try:
        mnpair = mnpair.tolist() if mnpair is not None else None
    except Exception:
        pass
    return binned_spikes, binned_force, mnpair



def create_force_glm_data(force, spikes):
    """
    Build a design matrix using the force bin to predict spikes.

    Parameters
    ----------
    force : array-like, shape (T,)
        Binned force (one value per time bin).
    spikes : array-like, shape (T,)
        Binned spike counts for one target neuron (one value per time bin).

    Returns
    -------
    X : np.ndarray, shape (T, 1)
        Design matrix with a single predictor: current force per bin.
    Y : np.ndarray, shape (T,)
        Target spike counts.

    """
    if len(force) != spikes.shape[0]:
        return None, None

    X = force.reshape(-1, 1)
    Y = spikes
    return X, Y


def compute_pR2(y_true, y_pred, eps=1e-12):
    """
    Compute Poisson deviance-based pseudo-R² (Perich et al.-style).

    Parameters
    ----------
    y_true : array-like
        Observed spike counts per bin.
    y_pred : array-like
        Model-predicted mean counts (rate * bin width) per bin.
    eps : float
        Small positive constant to avoid log(0).

    Returns
    -------
    pR2 : float
        1 - (Deviance_model / Deviance_null), where the null model predicts the mean count.

    Notes
    -----
    - Uses Poisson log-likelihood with a saturated model reference.
    - Returns NaN if the denominator is zero or not finite.
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


def plot_pr2_violin(session_results, title="Decoding Accuracy (pR²) by Session", out_path=None, palette_name="plasma", point_jitter=0.2, point_alpha=0.6, ylim=(-1, 1.1), figsize=(12,6), font_family="Arial"):
    """
    Plot per-session pR² distributions as violins with overlaid dots that are the individual neuron pR² values)
    """
    if not session_results:
        print("plot_pr2_violin: nothing to plot (empty session_results).")
        return None, None

    plt.figure(figsize=figsize)
    plt.rcParams.update({'font.family': font_family})

    # Flatten data and simplify labels (keep numeric id if present)
    all_pR2s, all_labels, simplified = [], [], {}
    for s in session_results.keys():
        m = re.search(r'(\d+)', s)
        simplified[s] = str(int(m.group(1))) if m else s

    for s, vals in session_results.items():
        all_pR2s.extend(vals)
        all_labels.extend([simplified[s]] * len(vals))

    # Ensure numeric order for labels like "1,2,..."
    unique_sessions = sorted(set(all_labels), key=lambda x: int(x) if x.isdigit() else x)

    # Build color palette
    import matplotlib as mpl
    cmap = mpl.cm.get_cmap(palette_name)
    N = max(len(unique_sessions) - 1, 1)
    colors = [cmap(i / N) for i in range(len(unique_sessions))]
    palette_dict = dict(zip(unique_sessions, colors))

    # Plot: violins + dots
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
