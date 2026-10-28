"""
    filter_input(f, inputs)

Apply a 3rd-order Butterworth low-pass filter with cutoff `f` Hz (Fs = 1000 Hz) to inputs.

# Inputs
- `f`: cutoff frequency in Hz.
- `inputs`: signal vector to filter.

# Outputs
- Filtered signal vector (same length as `inputs`).
"""

function filter_input(f,inputs)
    responsetype = Lowpass(f,fs = 1000)
    designmethod = Butterworth(3)
    inputs = filt(digitalfilter(responsetype, designmethod), inputs)
    return inputs
end