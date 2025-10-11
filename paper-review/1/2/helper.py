# helper.py
from __future__ import annotations
from typing import Tuple, Dict
import numpy as np

# -------- Activation + init --------
def tanh_activation():
    def fn(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    def deriv(x: np.ndarray) -> np.ndarray:
        y = np.tanh(x)
        return 1.0 - y*y
    return fn, deriv

def init_weights(m: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fan_in, fan_out = m + n, n
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(n, m + n)).astype(np.float32)

# -------- Tasks / data generators --------
def xor_delayed(T: int, tau: int, m: int, n_visible: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X, D, M) with shapes:
      X: (T, m)         inputs
      D: (T, n_visible) targets for y(t+1) at index t
      M: (T, n_visible) 0/1 mask for which targets are active at t
    """
    assert n_visible >= 1
    X = np.zeros((T, m), dtype=float)
    D = np.zeros((T, n_visible), dtype=float)
    M = np.zeros((T, n_visible), dtype=float)

    xor_inputs  = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    xor_outputs = np.array([0,1,1,0], dtype=float)

    for t in range(T):
        if t + tau < T:
            idx = np.random.randint(4)
            X[t, :2] = xor_inputs[idx]     # assumes first two dims carry XOR bits
            D[t + tau, 0] = float(xor_outputs[idx])
            M[t + tau, 0] = 1.0
    return X, D, M

# -------- Scorers (pluggable metrics) --------
def masked_binary_accuracy(
    Y_vis: np.ndarray,  # (T_total, n_visible)
    D: np.ndarray,      # (T_total, n_visible)
    M: np.ndarray,      # (T_total, n_visible) ∈ {0,1}
    threshold: float = 0.5,
) -> Dict[str, float]:
    preds = (Y_vis > threshold).astype(int)
    masked = (M == 1.0)
    total = int(masked.sum())
    correct = int(((preds == D.astype(int)) & masked).sum())
    acc = (100.0 * correct / total) if total > 0 else 0.0
    return {"name": "acc", "metric": acc, "correct": correct, "total": total}

def masked_mse(
    Y_vis: np.ndarray,
    D: np.ndarray,
    M: np.ndarray,
) -> Dict[str, float]:
    diff = (Y_vis - D)
    se = float(((diff * diff) * M).sum())
    count = int(M.sum())
    mse = (se / count) if count > 0 else 0.0
    return {"name": "mse", "metric": mse, "count": count}
