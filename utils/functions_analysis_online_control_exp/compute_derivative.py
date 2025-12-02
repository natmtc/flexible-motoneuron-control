import numpy as np

def compute_derivative(signal, fs):
    """
Compute the time derivative of a signal using finite differences.

Args:
    signal (array-like): Input 1D signal.
    fs (float): Sampling frequency in Hz.

Process:
    - Computes dt = 1/fs.
    - Uses np.gradient to approximate d(signal)/dt.

Returns:
    np.ndarray: Derivative of the signal with respect to time.
"""

    dt = 1 / fs  # Time step
    derivative = np.gradient(signal, dt)
    return derivative