"""
Calculate_rates_with_timebase
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
