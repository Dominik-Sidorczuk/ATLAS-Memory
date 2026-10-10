from __future__ import annotations

import numpy as np

from atlas_memory.l0_dynamic.ttt_kernels import njit


@njit(fastmath=True, nogil=True)
def numba_jepa_step(
    s_t: np.ndarray,
    a_t: np.ndarray,
    w_s: np.ndarray,
    w_a: np.ndarray,
    bias: np.ndarray,
    w_val: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Predykcja przejścia stanu JEPA skompilowana w Numba JIT (Fused Linear Projection):
    next_s = tanh(s_t @ w_s + a_t @ w_a + bias)
    """
    s_dim = s_t.shape[1]
    a_dim = a_t.shape[1]
    n_cols = w_s.shape[1]

    next_s = np.empty((1, n_cols), dtype=np.float64)
    sq_sum = 0.0

    for j in range(n_cols):
        raw = bias[0, j]
        for p in range(s_dim):
            raw += s_t[0, p] * w_s[p, j]
        for p in range(a_dim):
            raw += a_t[0, p] * w_a[p, j]

        val = np.tanh(raw)
        next_s[0, j] = val
        sq_sum += val * val

    # Ocena wartości (next_s @ w_val)
    sim_reward = 0.0
    for p in range(n_cols):
        sim_reward += next_s[0, p] * w_val[p, 0]

    uncertainty = 1.0 - (sq_sum / n_cols) if n_cols > 0 else 1.0
    return next_s, float(sim_reward), float(uncertainty)


@njit(fastmath=True, nogil=True)
def numba_jepa_adapt_step(
    s_t: np.ndarray,
    a_t: np.ndarray,
    target_s_next: np.ndarray,
    w_s: np.ndarray,
    w_a: np.ndarray,
    bias: np.ndarray,
    lr: float,
    weight_decay: float,
) -> float:
    """
    Krok adaptacji online wag JEPA (w_s, w_a, bias) gradientem straty rekonstrukcji
    predykowanego stanu względem obserwowanego target_s_next:
    L = 1/D * ||tanh(s_t @ w_s + a_t @ w_a + bias) - target_s_next||^2
    """
    s_dim = s_t.shape[1]
    a_dim = a_t.shape[1]
    n_cols = w_s.shape[1]

    loss = 0.0
    factor = 2.0 / n_cols if n_cols > 0 else 0.0

    grad_z = np.empty((1, n_cols), dtype=np.float64)

    for j in range(n_cols):
        raw = bias[0, j]
        for p in range(s_dim):
            raw += s_t[0, p] * w_s[p, j]
        for p in range(a_dim):
            raw += a_t[0, p] * w_a[p, j]

        pred_val = np.tanh(raw)
        diff = pred_val - target_s_next[0, j]
        loss += diff * diff
        grad_z[0, j] = factor * diff * (1.0 - pred_val * pred_val)

    loss = loss / n_cols if n_cols > 0 else 0.0

    # Aktualizacja in-place z weight decay
    for j in range(n_cols):
        gz = grad_z[0, j]
        bias[0, j] -= lr * gz
        for p in range(s_dim):
            w_s[p, j] -= lr * (s_t[0, p] * gz + weight_decay * w_s[p, j])
        for p in range(a_dim):
            w_a[p, j] -= lr * (a_t[0, p] * gz + weight_decay * w_a[p, j])

    return loss

