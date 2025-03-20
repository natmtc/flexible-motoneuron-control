def apply_pca_and_plot_sessions(folder_path, pkl_files, fsamp=2048, n_components=5):
    # Organize files by session (day)
    sessions = {}
    for file in pkl_files:
        day = extract_day(file)
        if day not in sessions:
            sessions[day] = []
        sessions[day].append(file)
    
    # Process each session separately
    for day, trial_files in sessions.items():
        all_trials = []
        trial_lengths = []
        min_neurons = float('inf')

        # First : determine minimum neuron count across trials
        for file in trial_files:
            file_path = os.path.join(folder_path, file)

            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            data_MU = np.array(data['MU'].tolist())  
            smoothed_firing_rates = compute_firing_rates(data_MU, fsamp)  

            min_neurons = min(min_neurons, smoothed_firing_rates.shape[0])

        # Second : Process and store trials
        for file in trial_files:
            file_path = os.path.join(folder_path, file)

            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            data_MU = np.array(data['MU'].tolist())  
            smoothed_firing_rates = compute_firing_rates(data_MU, fsamp)  

            # Resample neurons to match min_neurons
            if smoothed_firing_rates.shape[0] > min_neurons:
                smoothed_firing_rates = resample(smoothed_firing_rates, min_neurons, axis=0)

            all_trials.append(smoothed_firing_rates.T)  
            trial_lengths.append(smoothed_firing_rates.shape[1])

        # Concatenate trials along time axis
        all_trials_matrix = np.vstack(all_trials)  

        # Apply PCA to the full session
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(all_trials_matrix)  
        print(transformed_data.shape)

        # Compute trial start indices
        trial_starts = np.cumsum([0] + trial_lengths[:-1])  

        # Determine axis limits for consistent visualization
        x_min, x_max = np.min(transformed_data[:, 0]), np.max(transformed_data[:, 0])
        y_min, y_max = np.min(transformed_data[:, 1]), np.max(transformed_data[:, 1])
        z_min, z_max = np.min(transformed_data[:, 2]), np.max(transformed_data[:, 2])

        # Create a single figure for this session
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each trial separately in the same embedding
        colormap = cm.viridis
        colors = [colormap(i / len(trial_files)) for i in range(len(trial_files))]

        for idx in range(len(trial_starts) - 1):
            start, end = trial_starts[idx], trial_starts[idx + 1]
            ax.plot(transformed_data[start:end, 0], transformed_data[start:end, 1], transformed_data[start:end, 2], 
                    color=colors[idx], alpha=0.8, label=f'Trial {idx+1}')

        # Set same limits for all plots
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        # Labels and title
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"Session {day}: All Trials in PCA Space")

        ax.legend(loc='best', fontsize=8)
        plt.show()