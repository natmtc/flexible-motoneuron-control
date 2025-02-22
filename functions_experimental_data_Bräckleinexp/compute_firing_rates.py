def compute_firing_rates(spike_array, fs, window_duration=0.3):
    window_samples = int(round(window_duration * fs))  # Convert duration to samples
    hanning_window = windows.hann(window_samples)  # Create Hanning window
    hanning_window /= np.sum(hanning_window)  # Normalize window
    
    smoothed_firing_rates = np.zeros_like(spike_array, dtype=float)
    
    for i in range(spike_array.shape[0]):  
        smoothed_firing_rates[i, :] = convolve(spike_array[i, :], hanning_window, mode='same') * fs  
    
    return smoothed_firing_rates
