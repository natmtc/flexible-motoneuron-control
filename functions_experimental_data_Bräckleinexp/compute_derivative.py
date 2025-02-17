def compute_derivative(signal, fs):
    dt = 1 / fs  # Time step
    derivative = np.gradient(signal, dt)
    return derivative