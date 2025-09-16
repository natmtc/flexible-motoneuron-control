def conv_spikes(array, pos1, pos2, fs, window_duration=0.3):
    window_samples = int(round(window_duration * fs))
    hanning_window = windows.hann(window_samples)
    hanning_window /= np.sum(hanning_window)  
    smoothed_spike_array1 = convolve(array[pos1, :], hanning_window, mode='same') * fs
    smoothed_spike_array2 = convolve(array[pos2, :], hanning_window, mode='same') * fs
    return smoothed_spike_array1, smoothed_spike_array2

