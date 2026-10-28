"""
    calculate_rates_with_timebase(spike_pos, sampling_rate)

Compute instantaneous firing rates from spike sample positions and return the corresponding timebase.

# Inputs
- `spike_pos`: vector of spike indices (1-based, in samples).
- `sampling_rate`: sampling frequency in Hz.

# Outputs
- `timebase`: vector of times (s) for each rate (aligned to spikes 2..N).
- `rates`: vector of instantaneous firing rates (Hz).
"""
function calculate_rates_with_timebase(spike_pos, sampling_rate)
    if isempty(spike_pos)
        return [], []
    end
    ISI = diff(spike_pos)
    rates = 1 ./ (ISI / sampling_rate) # Convert ISI to seconds and calculate rates

    timebase = (spike_pos[2:end] .- 1) / sampling_rate
    return timebase, rates
end
