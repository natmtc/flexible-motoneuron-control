import os               
import pickle          
import numpy as np      


def load_and_preprocess(file, folder_path, bin_size_ms, sampling_freq=2048):
    file_path = os.path.join(folder_path, file)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    data_MU = np.array(data['MU'].tolist())  # neurons x samples
    
    bin_size_samples = int(bin_size_ms * sampling_freq / 1000)    
    binned_spikes = np.add.reduceat(data_MU, np.arange(0, data_MU.shape[1], bin_size_samples), axis=1)
    mnpair = data['Idx_MU_Pair'].tolist()
    
    return binned_spikes.T, data, mnpair


# def load_and_preprocess(file,folder_path, bin_size_ms):
#     """Load a single trial and preprocess"""
#     file_path = os.path.join(folder_path, file)
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
    
#     data_MU = np.array(data['MU'].tolist())  
#     binned_spikes = np.add.reduceat(data_MU, np.arange(0, data_MU.shape[1], bin_size_ms), axis=1)
#     mnpair= data['Idx_MU_Pair'].tolist()
#     return binned_spikes.T, data, mnpair


def create_glm_data_with_history(spike_data, history_bins, target_idx):
    """
    Create X (predictors) and Y (target) for GLM using past (history) population activity, excluding the target neuron from the predictors.
    """    
    num_timebins, num_neurons = spike_data.shape
    
    # Identify neurons that spike more than 10 times
    active_neurons = np.where(spike_data.sum(axis=0) > 10)[0]
    
    if target_idx not in active_neurons:
        return None, None  # If the target neuron is inactive, return None to skip trial
    
    X_list, Y_list = [], []
    
    for t in range(history_bins, num_timebins):
        # Exclude neurons that never spike + target neuron itself from predictors
        X_t = np.hstack([
            spike_data[t - h, np.logical_and(
                np.arange(num_neurons) != target_idx,  
                np.isin(np.arange(num_neurons), active_neurons)
            )]
            for h in range(1, history_bins + 1)
        ])
        Y_t = spike_data[t, target_idx]  
        X_list.append(X_t)
        Y_list.append(Y_t)
    
    return np.array(X_list), np.array(Y_list)


def create_glm_data(spike_data, target_idx):
    """
    Create X (predictors) and Y (target) for GLM using only instantaneous (same-bin) population activity, excluding the target neuron from the predictors.
    """
    num_timebins, num_neurons = spike_data.shape

    # Identify neurons that spike more than 10 times
    active_neurons = np.where(spike_data.sum(axis=0) > 10)[0]
    
    if target_idx not in active_neurons:
        return None, None  # Skip if target neuron is inactive

    # Mask to exclude target neuron and inactive ones
    mask = np.logical_and(np.arange(num_neurons) != target_idx,np.isin(np.arange(num_neurons), active_neurons))
    # Build X and Y from same time bins
    X = spike_data[:, mask]
    Y = spike_data[:, target_idx]

    if np.sum(Y) < 20:
        return None, None

    return X, Y

def compute_pR2_mcfadden(model, X, Y):
    """Compute McFadden's pseudo-R² with improved numerical stability."""
    y_pred = model.predict(X)

    # Prevent log(0) errors by ensuring positive values
    y_pred = np.clip(y_pred, 1e-6, None)  
    Y = np.clip(Y, 1e-6, None)  

    # when all spikes are 0
    if np.all(Y == 0) or np.all(y_pred == 0):
        return np.nan  # Skip computation if prediction is possible
    
    deviance_model = 2 * np.sum(Y * np.log(Y / y_pred) - (Y - y_pred))  # how far tge model is from a perfect prediction
    deviance_null = 2 * np.sum(Y * np.log(Y / np.mean(Y)) - (Y - np.mean(Y)))

    # Handle division by zero
    if deviance_null == 0 or np.isnan(deviance_null):
        return np.nan  # Skip invalid computations

    return 1 - (deviance_model / deviance_null)



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
