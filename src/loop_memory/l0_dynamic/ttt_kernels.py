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
def numba_ttt_forward(x: np.ndarray, w_base: np.ndarray, w_ttt: np.ndarray) -> np.ndarray:
    """Skompresowana projekcja forward: h = tanh(x @ (w_base + w_ttt))."""
    w_eff = w_base + w_ttt
    z = _numba_matmul_2d(x, w_eff)
    return np.tanh(z)


@njit(fastmath=True, nogil=True)
def numba_ttt_adapt_step(
    x: np.ndarray,
    w_base: np.ndarray,
    w_ttt: np.ndarray,
    w_recon: np.ndarray,
    lr: float,
    weight_decay: float,
) -> float:
    """
    Krok optymalizacji online L0 TTT skompilowany do instrukcji maszynowych LLVM.
    Fuzja pętli forward + reconstruction + backward gradient + in-place weight update.
    """
    # 1. Forward
    w_eff = w_base + w_ttt
    z = _numba_matmul_2d(x, w_eff)
    h = np.tanh(z)

    # 2. Rekonstrukcja
    x_hat = _numba_matmul_2d(h, w_recon)

    # 3. Obliczenie straty L2
    diff = x_hat - x
    n_rows = diff.shape[0]
    n_cols = diff.shape[1]
    total_elements = n_rows * n_cols

    loss = 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            loss += diff[i, j] * diff[i, j]
    loss /= total_elements

    # 4. Wsteczny gradient dL/dW_ttt
    grad_x_hat = np.zeros((n_rows, n_cols), dtype=np.float64)
    factor = 2.0 / total_elements
    for i in range(n_rows):
        for j in range(n_cols):
            grad_x_hat[i, j] = factor * diff[i, j]

    w_recon_t = np.ascontiguousarray(w_recon.T)
    grad_h = _numba_matmul_2d(grad_x_hat, w_recon_t)

    grad_z = np.zeros((grad_h.shape[0], grad_h.shape[1]), dtype=np.float64)
    for i in range(grad_h.shape[0]):
        for j in range(grad_h.shape[1]):
            grad_z[i, j] = grad_h[i, j] * (1.0 - h[i, j] * h[i, j])

    x_t = np.ascontiguousarray(x.T)
    grad_w = _numba_matmul_2d(x_t, grad_z)

    # 5. Aktualizacja in-place z weight decay
    for r in range(w_ttt.shape[0]):
        for c in range(w_ttt.shape[1]):
            w_ttt[r, c] -= lr * (grad_w[r, c] + weight_decay * w_ttt[r, c])

    return loss
