
import os               
import pickle          
import numpy as np      


def preprocess_trial_force(file,folder_path, bin_size_ms):
    file_path = os.path.join(folder_path, file)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    force = np.array(data['Force'])
    if force.ndim == 2:
        force = force.squeeze()

    data_MU = np.array(data['MU'].tolist())

    # Just compute number of bins based on the full length
    num_timepoints = force.shape[0]
    num_bins = num_timepoints // bin_size_ms

    # Bin the spikes and force
    binned_spikes = np.add.reduceat(data_MU, np.arange(0, num_bins * bin_size_ms, bin_size_ms), axis=1)

    binned_force = np.add.reduceat(force[:num_bins * bin_size_ms], np.arange(0, num_bins * bin_size_ms, bin_size_ms)) / bin_size_ms

    mnpair= data['Idx_MU_Pair'].tolist()

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


def compute_pR2(model, X, Y):
    y_pred = model.predict(X)
    y_pred = np.clip(y_pred, 1e-12, None)

    y_mean = np.mean(Y)
    if np.all(Y == 0) or np.all(y_pred == 0):
        return np.nan

    # Use original Y for saturated model, clip only inside log
    ll_model = np.sum(Y * np.log(y_pred) - y_pred)
    ll_null = np.sum(Y * np.log(y_mean) - y_mean)
    ll_data = np.sum(Y * np.log(np.clip(Y, 1e-12, None)) - Y)

    denom = ll_data - ll_null
    if denom <= 0 or np.isnan(denom):
        return np.nan

    r2 = 1 - (ll_model - ll_null) / denom
    return min(1.0, r2)

