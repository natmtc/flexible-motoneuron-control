function conv_spikes(array, pos1,pos2)

        window_samples = round(Int, 400 * 10^-3 * 20000) ; # 400 ms * 20 kHz  
        hanning_window = DSP.Windows.hanning(window_samples);
        hanning_window ./= sum(hanning_window);# Normalize the window so it sums to 1 (to maintain the same scale after convolution)

        smoothed_spike_array1 = conv(array[pos1,:], hanning_window);# Convolve the spike arrays with the Gaussian window
        smoothed_spike_array2 = conv(array[pos2,:], hanning_window);

        # Because convolution extends the length of the array, 
        # let's truncate it to the original length
        len = length(array[1,:]);
        smoothed_spike_array1 = smoothed_spike_array1[1:len];
        smoothed_spike_array2 = smoothed_spike_array2[1:len];

    return smoothed_spike_array1, smoothed_spike_array2
end