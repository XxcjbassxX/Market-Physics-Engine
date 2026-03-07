"""Potential from force integration."""
import numpy as np

def _smooth(arr, size):
    return arr if size <= 1 else np.convolve(arr, np.ones(size) / size, mode="same")

def estimate_potential(prices: np.ndarray, force: np.ndarray, smooth_window: int = 5) -> np.ndarray:
    f = _smooth(np.asarray(force), min(smooth_window, len(force))) if smooth_window > 1 else np.asarray(force)
    dx = np.diff(np.asarray(prices), prepend=prices[0])
    v = np.zeros_like(prices)
    for i in range(1, len(prices)):
        v[i] = v[i - 1] - f[i - 1] * dx[i]
    return _smooth(v, min(smooth_window, len(v))) - v[0]
