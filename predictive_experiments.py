"""Predictive backtest."""
import numpy as np
import pandas as pd

def run_predictive_backtest(prices, velocity, acceleration, kinetic, potential, total_energy, horizon_days=5, energy_threshold_percentile=75.0):
    n = len(prices)
    if n < horizon_days + 2:
        return {"n_signals":0,"avg_return":0.0,"hit_rate":0.0,"sample_size":0,"signal_indices":[],"signal_types":[]}
    acc, cross = np.sign(acceleration), np.diff(np.sign(acceleration)) != 0
    thresh = np.percentile(total_energy, energy_threshold_percentile)
    rets, idxs, typs = [], [], []
    for i in range(1, n - horizon_days):
        if not cross[i-1] or total_energy[i] < thresh:
            continue
        typs.append("sell" if acc[i-1]>0 and acc[i]<0 else "buy")
        idxs.append(i)
        rets.append((prices[i+horizon_days]-prices[i])/prices[i])
    if not rets:
        return {"n_signals":0,"avg_return":0.0,"hit_rate":0.0,"sample_size":0,"signal_indices":[],"signal_types":[]}
    rets = np.array(rets)
    return {"n_signals":len(rets),"avg_return":float(np.mean(rets)),"std_return":float(np.std(rets)),"hit_rate":float(np.mean(rets>0)),"sample_size":len(rets),"signal_indices":idxs,"signal_types":typs}

def backtest_multi_ticker(ticker_data, horizon_days=5, energy_percentile=75.0):
    rows = []
    for t, d in ticker_data.items():
        if d is None:
            continue
        r = run_predictive_backtest(d["prices"],d["velocity"],d["acceleration"],d["kinetic"],d["potential"],d["total_energy"],horizon_days,energy_percentile)
        r["ticker"] = t
        rows.append(r)
    return pd.DataFrame(rows)
