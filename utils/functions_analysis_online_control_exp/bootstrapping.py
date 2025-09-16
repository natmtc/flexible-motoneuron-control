import numpy as np 

def bootstrap_pr2_analysis(session_pR2, n_bootstrap=1000):
    bootstrap_results = {}
    session_bootstrap_means = {}
    for session, pr2_values in session_pR2.items():
        pr2_values = np.array(pr2_values)
        pr2_values = pr2_values[~np.isnan(pr2_values)]
        if len(pr2_values) == 0:
            bootstrap_results[session] = {"Error": "No valid pR² values found."}
            continue
        bootstrap_means = []
        for _ in range(n_bootstrap):
            resample = np.random.choice(pr2_values, size=len(pr2_values), replace=True)
            bootstrap_means.append(np.mean(resample))
        ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])
        session_bootstrap_means[session] = np.mean(bootstrap_means)
        bootstrap_results[session] = {
            "Mean pR²": np.mean(pr2_values),
            "Bootstrap mean": np.mean(bootstrap_means),
            "95% CI lower bound": ci_lower,
            "95% CI upper bound": ci_upper,
            "Number of neurons": len(pr2_values),
            "Significant": ci_lower > 0
        }
    return bootstrap_results, session_bootstrap_means