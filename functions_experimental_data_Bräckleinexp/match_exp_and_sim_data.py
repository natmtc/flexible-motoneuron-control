from scipy.optimize import linear_sum_assignment
from functions_experimental_data_Bräckleinexp.firing_rates import *
import h5py, numpy as np, re

def match_experimental_to_simulated(session_files, folder_path, session_results, session_max_firing_rates, fsamp, cost_threshold=5):
    matched_results = {}
    for session_file in session_files:
        session_name = session_file.split('.')[0]
        match = re.search(r'force(\d+)', session_file)
        day_number = int(match.group(1)) if match else None
        day_label = f"Day{day_number:02d}"
        if session_name not in session_results:
            continue
        print(f"Matching for {session_name}")
        file_path = folder_path + session_file
        with h5py.File(file_path, 'r') as f:
            spike_data = f['spike_array'][:].T
        firing_rates = compute_firing_rates(spike_data, fsamp)
        max_rates_sim = firing_rates.max(axis=1)
        exp_neurons = session_max_firing_rates.get(day_label, [])
        max_rates_exp = np.array([rate for (_, rate) in exp_neurons])
        if len(max_rates_exp) == 0:
            continue
        cost_matrix = np.abs(max_rates_exp[:, None] - max_rates_sim[None, :])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        all_pr2 = session_results[session_name]
        matched_pr2 = []
        for exp_i, sim_j in zip(row_ind, col_ind):
            exp_rate = max_rates_exp[exp_i]
            sim_rate = max_rates_sim[sim_j]
            cost = abs(exp_rate - sim_rate)
            if cost <= cost_threshold:
                if sim_j < len(all_pr2):
                    matched_pr2.append(all_pr2[sim_j])
                print(f"✓ exp[{exp_i}] (rate={exp_rate:.2f}) ↔ sim[{sim_j}] (rate={sim_rate:.2f}) → cost = {cost:.2f}")
            else:
                print(f"✗ exp[{exp_i}] (rate={exp_rate:.2f}) ↔ sim[{sim_j}] (rate={sim_rate:.2f}) → cost = {cost:.2f} → skipped")
        matched_results[session_name] = matched_pr2
    return matched_results
