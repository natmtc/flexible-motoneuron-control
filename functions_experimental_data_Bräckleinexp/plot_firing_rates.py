def plot_firing_rates(folder_path, pkl_files, fsamp=2048, window_duration=0.3):
    num_files = len(pkl_files)
    cols = 3  # Number of columns for subplots
    rows = math.ceil(num_files / cols)  # Compute number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, file in enumerate(pkl_files):
        file_path = os.path.join(folder_path, file)

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        data_MU = np.array(data['MU'].tolist())  
        smoothed_firing_rates = compute_firing_rates(data_MU, fsamp, window_duration)

        ax = axes[idx]
        time_axis = np.arange(data_MU.shape[1]) / fsamp  # Convert sample index to time in seconds

        for neuron_idx in range(data_MU.shape[0]):
            ax.plot(time_axis, smoothed_firing_rates[neuron_idx, :], label=f'Neuron {neuron_idx}', alpha=0.7)

        num_neurons = data_MU.shape[0]  # Count the number of motoneurons
        ax.set_title(f'Trial: {file} (n={num_neurons})', fontsize=10)  # Update title

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing Rate (Hz)')

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])  

    plt.tight_layout()
    plt.ylim([0, 40])
    # plt.savefig('/Users/nmt121/Desktop/RESULTS_ DATA EXP/S7firings.pdf', dpi=300, bbox_inches='tight')
    plt.show()