"""Brownian simulation."""
import numpy as np
import pandas as pd

def simulate_brownian_paths(S0, mu, sigma, n_days, n_paths=100, seed=None):
    if seed is not None:
        np.random.seed(seed)
    paths = np.zeros((n_days+1, n_paths))
    paths[0] = S0
    for t in range(1, n_days+1):
        paths[t] = paths[t-1] * np.exp((mu - 0.5*sigma**2) + sigma*np.random.standard_normal(n_paths))
    return paths

def fit_and_simulate(prices, n_days=252, n_paths=100, seed=None):
    p = np.asarray(prices)
    ret = np.diff(p)/p[:-1]
    ret = ret[~np.isnan(ret)]
    if len(ret) < 2:
        return np.tile(p[-1], (n_days+1, n_paths)), 0.0, 0.0
    mu, sigma = np.mean(ret), np.std(ret)
    return simulate_brownian_paths(float(p[-1]), mu, sigma, n_days, n_paths, seed), mu, sigma
