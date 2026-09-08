from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


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
def numba_jepa_rollout(
    s_0: np.ndarray,
    actions_mat: np.ndarray,
    w_s: np.ndarray,
    w_a: np.ndarray,
    bias: np.ndarray,
    w_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wielokrokowa symulacja myślowa (System 2 Rollout) w jednej skompilowanej pętli C.
    Zoptymalizowana pod kątem in-place zapisu stanów trajektorii bez alokacji pośrednich.
    """
    n_steps = actions_mat.shape[0]
    state_dim = s_0.shape[1]
    a_dim = actions_mat.shape[1]

    trajectory_states = np.empty((n_steps, state_dim), dtype=np.float64)
    trajectory_rewards = np.empty(n_steps, dtype=np.float64)
    trajectory_uncertainties = np.empty(n_steps, dtype=np.float64)

    prev_s = s_0
    for i in range(n_steps):
        a_i = actions_mat[i : i + 1]
        sq_sum = 0.0
        sim_reward = 0.0
        for j in range(state_dim):
            raw = bias[0, j]
            for p in range(state_dim):
                raw += prev_s[0, p] * w_s[p, j]
            for p in range(a_dim):
                raw += a_i[0, p] * w_a[p, j]

            val = np.tanh(raw)
            trajectory_states[i, j] = val
            sq_sum += val * val
            sim_reward += val * w_val[j, 0]

        trajectory_rewards[i] = sim_reward
        trajectory_uncertainties[i] = 1.0 - (sq_sum / state_dim) if state_dim > 0 else 1.0
        prev_s = trajectory_states[i : i + 1]

    return trajectory_states, trajectory_rewards, trajectory_uncertainties
