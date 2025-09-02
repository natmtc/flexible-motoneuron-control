import numpy as np
import os, pickle, numpy as np, matplotlib.pyplot as plt
from scipy.stats import poisson, chisquare


def chisq_poisson_gof(counts, min_exp=5, tail_sd=10):
    """
    Chi-square GOF test for Poisson with λ = mean(counts).
    Pools bins so all expected >= min_exp.
    """
    x = np.asarray(counts).ravel().astype(int)
    if x.size == 0:
        raise ValueError("counts is empty")
    n   = x.size
    lam = x.mean()

    # Support up to max obs or lam+tail_sd*sqrt(lam)
    K = max(x.max(), int(np.ceil(lam + tail_sd*np.sqrt(max(lam,1e-12)))))
    obs = np.bincount(x, minlength=K+1).astype(float)
    exp = n * poisson.pmf(np.arange(K+1), lam)
    exp[-1] += (n - exp.sum())   # fix rounding

    # Pool from right until all >= min_exp
    while (exp < min_exp).any() and len(exp) > 1:
        i = np.where(exp < min_exp)[0][-1]  # last small bin
        exp[i-1] += exp[i]; obs[i-1] += obs[i]
        exp, obs = np.delete(exp, i), np.delete(obs, i)

    chi2, p = chisquare(f_obs=obs, f_exp=exp, ddof=1)
    return chi2, p, {"lambda_hat": lam, "obs": obs, "exp": exp}


def bin_spikes(MU, fsamp, bin_ms):
    bin_len = int(round((bin_ms/1000.0) * fsamp))
    n = MU.shape[1] // bin_len
    MU = MU[:, :n*bin_len]
    return MU.reshape(MU.shape[0], n, bin_len).sum(axis=2)
