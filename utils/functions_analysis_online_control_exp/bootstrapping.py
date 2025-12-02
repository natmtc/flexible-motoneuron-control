import numpy as np 

def bootstrap_pr2_analysis(session_pR2, n_bootstrap=1000):
    """
Compute bootstrapped statistics for session-wise pR² values.

For each session, this function:
- Removes NaNs from the pR² array.
- Performs bootstrap resampling (default: 1000 iterations).
- Computes the bootstrap mean and 95% confidence interval.
- Flags significance when the CI lower bound is > 0.

Returns:
    bootstrap_results (dict): Per-session statistics including mean pR²,
        bootstrap mean, CI bounds, neuron count, and significance flag.
    session_bootstrap_means (dict): Mean bootstrap pR² per session.
"""
    
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