"""Turning points and FFT cycles."""
import numpy as np
import pandas as pd
from scipy import fft

def detect_turning_points(dates, prices, acceleration) -> pd.DataFrame:
    acc, s = np.asarray(acceleration), np.sign(acceleration)
    if len(acc) < 3:
        return pd.DataFrame(columns=["index","date","price","type"])
    cross = np.diff(s) != 0
    rows = [(i+1, dates[i+1], float(prices[i+1]), "peak" if s[i]>s[i+1] else "trough") for i in range(len(cross)) if cross[i]]
    return pd.DataFrame(rows, columns=["index","date","price","type"]).sort_values("index") if rows else pd.DataFrame(columns=["index","date","price","type"])

def detect_cycles_fft(prices, dt=1.0, n_dominant=5):
    n = len(prices)
    if n < 4:
        return np.array([]), np.array([]), []
    x = prices - np.mean(prices)
    freqs = fft.fftfreq(n, dt)[:n//2]
    power = np.abs(fft.fft(x)[:n//2])**2
    mask = freqs > 0
    freqs, power = freqs[mask], power[mask]
    top = np.argsort(power)[::-1][:n_dominant]
    cycles = [{"frequency":float(freqs[i]),"period_days":float(1/freqs[i]),"power":float(power[i])} for i in top]
    return freqs, power, cycles

def get_dominant_cycle_lengths(prices, candidates=None):
    candidates = candidates or [20.0, 60.0, 120.0]
    n = len(prices)
    if n < 4:
        return [{"period":p,"prominence":0.0} for p in candidates]
    x = prices - np.mean(prices)
    freqs = fft.fftfreq(n, 1.0)[:n//2]
    power = np.abs(fft.fft(x)[:n//2])**2
    mask = freqs > 0
    freqs, power = freqs[mask], power[mask]
    total = np.sum(power) or 1.0
    return [{"period":p,"prominence":float(power[np.argmin(np.abs(freqs-1/p))]/total)} for p in candidates]
