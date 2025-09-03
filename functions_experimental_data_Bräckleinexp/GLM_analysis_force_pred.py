
import os               
import pickle          
import numpy as np
from scipy.special import gammaln



def preprocess_trial_force(file, folder_path, bin_size_ms, fsamp=2048):
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

    binned_spikes = np.add.reduceat(
        data_MU[:, :num_bins * bin_size_samples],
        np.arange(0, num_bins * bin_size_samples, bin_size_samples),
        axis=1
    )
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


def create_force_glm_data_with_history(force, spikes, history_bins):
    if len(force) != spikes.shape[0]:
        return None, None

    X, Y = [], []
    for t in range(history_bins, len(force)):
        history_window = force[t - history_bins:t]
        X.append(history_window)
        Y.append(spikes[t])
    return np.array(X), np.array(Y)

def create_force_glm_data(force, spikes):
    if len(force) != spikes.shape[0]:
        return None, None

    # Use only current force value as predictor
    X = force.reshape(-1, 1)
    Y = spikes

    if np.sum(Y) < 20:
        return None, None

    return X, Y


def compute_pR2(y_true, y_pred, eps=1e-12):
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
