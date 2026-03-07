"""Energy and force."""
import numpy as np
import pandas as pd

def compute_market_energy(prices: np.ndarray, velocity: np.ndarray, ma_window: int = 50) -> tuple:
    ma = pd.Series(prices).rolling(min(ma_window, len(prices)), min_periods=1).mean().values
    k = velocity ** 2
    p = (prices - ma) ** 2
    return k, p, k + p

def get_dynamics_summary(velocity: np.ndarray, acceleration: np.ndarray) -> dict:
    f = np.asarray(acceleration)
    return {"max_velocity": float(np.max(np.abs(velocity))), "max_acceleration": float(np.max(np.abs(acceleration))), "average_force_magnitude": float(np.mean(np.abs(f))), "force_direction_changes": int(np.sum(np.diff(np.sign(f)) != 0))}
