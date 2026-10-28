import os
import re
import pickle
import numpy as np

def extract_day(filename):
    """ 
    Extract the day string (e.g., 'Day01', 'Day14') from a given filename. 
    """
  
    match = re.search(r"(Day\d{2})", filename)
    return match.group(1) if match else None  

def trial_success_rate_per_day(directory_path, fsamp=2048, success_thresh=19.0):

    """
    Compute trial success rates per day from .pkl files containing force signals.
    """

    pkl_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]
    
    # group files by day
    files_by_day = {}
    for f in pkl_files:
        day = extract_day(f)
        if day:
            files_by_day.setdefault(day, []).append(f)
    
    results = {}
    for day, files in sorted(files_by_day.items()):
        n_total, n_success = 0, 0
        for file_name in files:
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            force_signal = np.array(data['Force'].tolist())[0]
            duration_sec = len(force_signal) / fsamp

            n_total += 1
            if duration_sec < success_thresh:
                n_success += 1

        success_rate = 100.0 * n_success / n_total if n_total > 0 else np.nan
        results[day] = {"success_rate": success_rate,"n_success": n_success,"n_total": n_total}
    
    return results
