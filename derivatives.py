"""Central difference derivatives."""
import numpy as np
import pandas as pd

def compute_derivatives(prices: pd.Series, dt: float = 1.0) -> tuple:
    p = np.asarray(prices, dtype=float)
    n = len(p)
    v, a, j = np.zeros_like(p), np.zeros_like(p), np.zeros_like(p)
    if n >= 3:
        v[1:-1] = (p[2:] - p[:-2]) / (2 * dt)
        v[0], v[-1] = (p[1] - p[0]) / dt, (p[-1] - p[-2]) / dt
        a[1:-1] = (v[2:] - v[:-2]) / (2 * dt)
        a[0], a[-1] = (v[1] - v[0]) / dt, (v[-1] - v[-2]) / dt
        j[1:-1] = (a[2:] - a[:-2]) / (2 * dt)
        j[0], j[-1] = (a[1] - a[0]) / dt, (a[-1] - a[-2]) / dt
    elif n == 2:
        v[:] = (p[1] - p[0]) / dt
    return v, a, j
