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
def _numba_matmul_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Natywne jądro mnożenia macierzy w Numba bez zewnętrznej zależności od Scipy/BLAS."""
    n = a.shape[0]
    k = a.shape[1]
    m = b.shape[1]
    c = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for p in range(k):
            a_ip = a[i, p]
            for j in range(m):
                c[i, j] += a_ip * b[p, j]
    return c


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
    """
    n_steps = actions_mat.shape[0]
    state_dim = s_0.shape[1]

    trajectory_states = np.zeros((n_steps, state_dim), dtype=np.float64)
    trajectory_rewards = np.zeros(n_steps, dtype=np.float64)
    trajectory_uncertainties = np.zeros(n_steps, dtype=np.float64)

    curr_s = s_0
    for i in range(n_steps):
        a_i = actions_mat[i : i + 1]
        next_s, rew, unc = numba_jepa_step(curr_s, a_i, w_s, w_a, bias, w_val)
        for d in range(state_dim):
            trajectory_states[i, d] = next_s[0, d]
        trajectory_rewards[i] = rew
        trajectory_uncertainties[i] = unc
        curr_s = next_s

    return trajectory_states, trajectory_rewards, trajectory_uncertainties
