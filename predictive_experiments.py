"""Predictive backtest."""
import numpy as np
import pandas as pd

def _bootstrap_ci(values, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval for a scalar statistic from values."""
    if len(values) < 2:
        return 0.0, 0.0
    n = len(values)
    rng = np.random.default_rng(42)
    stats = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        stats.append(np.mean(values[idx]))
    stats = np.array(stats)
    lo = np.percentile(stats, (1 - ci) / 2 * 100)
    hi = np.percentile(stats, (1 + ci) / 2 * 100)
    return float(lo), float(hi)

def compute_buy_and_hold_return(prices, horizon_days=5):
    """Return of holding from first eligible day to last, over horizon_days windows."""
    n = len(prices)
    if n < horizon_days + 1:
        return 0.0
    rets = [(prices[i + horizon_days] - prices[i]) / prices[i] for i in range(n - horizon_days)]
    return float(np.mean(rets))

def run_predictive_backtest(
    prices, velocity, acceleration, kinetic, potential, total_energy,
    horizon_days=5, energy_threshold_percentile=75.0, n_bootstrap=500
):
    n = len(prices)
    buy_hold = compute_buy_and_hold_return(prices, horizon_days)
    if n < horizon_days + 2:
        return {
            "n_signals": 0, "avg_return": 0.0, "hit_rate": 0.0, "sample_size": 0,
            "signal_indices": [], "signal_types": [],
            "buy_hold_return": buy_hold,
            "avg_return_ci": (0.0, 0.0), "hit_rate_ci": (0.0, 0.0),
        }
    acc, cross = np.sign(acceleration), np.diff(np.sign(acceleration)) != 0
    thresh = np.percentile(total_energy, energy_threshold_percentile)
    rets, idxs, typs = [], [], []
    for i in range(1, n - horizon_days):
        if not cross[i - 1] or total_energy[i] < thresh:
            continue
        typs.append("sell" if acc[i - 1] > 0 and acc[i] < 0 else "buy")
        idxs.append(i)
        rets.append((prices[i + horizon_days] - prices[i]) / prices[i])
    if not rets:
        return {
            "n_signals": 0, "avg_return": 0.0, "hit_rate": 0.0, "sample_size": 0,
            "signal_indices": [], "signal_types": [],
            "buy_hold_return": buy_hold,
            "avg_return_ci": (0.0, 0.0), "hit_rate_ci": (0.0, 0.0),
        }
    rets = np.array(rets)
    hits = (rets > 0).astype(float)
    avg_ci = _bootstrap_ci(rets, n_bootstrap=n_bootstrap)
    hit_ci = _bootstrap_ci(hits, n_bootstrap=n_bootstrap)
    return {
        "n_signals": len(rets), "avg_return": float(np.mean(rets)),
        "std_return": float(np.std(rets)), "hit_rate": float(np.mean(hits)),
        "sample_size": len(rets), "signal_indices": idxs, "signal_types": typs,
        "buy_hold_return": buy_hold,
        "avg_return_ci": avg_ci, "hit_rate_ci": hit_ci,
    }

def backtest_multi_ticker(ticker_data, horizon_days=5, energy_percentile=75.0):
    rows = []
    for t, d in ticker_data.items():
        if d is None:
            continue
        r = run_predictive_backtest(d["prices"],d["velocity"],d["acceleration"],d["kinetic"],d["potential"],d["total_energy"],horizon_days,energy_percentile)
        r["ticker"] = t
        rows.append(r)
    return pd.DataFrame(rows)
