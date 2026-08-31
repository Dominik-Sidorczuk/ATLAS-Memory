from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from loop_memory.l0_dynamic.ttt_kernels import numba_ttt_adapt_step, numba_ttt_forward


class TTTLayer:
    """
    L0: Warstwa Plastyczności Wag (Numba JIT Accelerated Test-Time Training).
    
    Wykorzystuje jądra kompilowane w LLVM (Numba fastmath/SIMD) do natychmiastowej
    aktualizacji wag online Delta W_t w czasie rzędu mikrosekund (< 0.05 ms).
    Obsługuje wywołania synchroniczne oraz asynchroniczne (asyncio.to_thread / nogil).
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 32,
        learning_rate: float = 0.05,
        weight_decay: float = 0.001,
        seed: Optional[int] = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.step_count = 0
        self.total_energy = 0.0
        self.last_update_timestamp = time.time()

        rng = np.random.default_rng(seed)
        self.w_base = (rng.standard_normal((input_dim, hidden_dim)) * (1.0 / np.sqrt(input_dim))).astype(np.float64)
        self.w_ttt = np.zeros((input_dim, hidden_dim), dtype=np.float64)
        self.w_recon = (rng.standard_normal((hidden_dim, input_dim)) * (1.0 / np.sqrt(hidden_dim))).astype(np.float64)

        # Rozgrzanie JIT (warmup)
        dummy_x = np.zeros((1, input_dim), dtype=np.float64)
        numba_ttt_adapt_step(dummy_x, self.w_base, self.w_ttt, self.w_recon, self.lr, self.weight_decay)
        self.reset_session()

    @property
    def effective_weights(self) -> np.ndarray:
        return self.w_base + self.w_ttt

    def forward(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        """Przejście w przód: h = tanh(x @ W_eff)."""
        x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        return numba_ttt_forward(x_arr, self.w_base, self.w_ttt)

    def adapt_step(self, x: Union[List[float], np.ndarray]) -> Tuple[float, float]:
        """
        Krok optymalizacji online TTT (synchroniczny).
        Zwraca: (loss, czas_wykonania_ms)
        """
        start_t = time.perf_counter()
        x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        loss = numba_ttt_adapt_step(
            x_arr,
            self.w_base,
            self.w_ttt,
            self.w_recon,
            self.lr,
            self.weight_decay,
        )

        self.step_count += 1
        self.total_energy += float(loss)
        self.last_update_timestamp = time.time()
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return float(loss), elapsed_ms

    async def adapt_step_async(self, x: Union[List[float], np.ndarray]) -> Tuple[float, float]:
        """Asynchroniczne wywołanie kroku TTT w puli wątków (odciążenie pętli AsyncIO)."""
        return await asyncio.to_thread(self.adapt_step, x)

    def compress_stream(self, stream_vectors: Union[List[List[float]], np.ndarray]) -> Dict[str, Any]:
        """Kompresuje ciągły strumień wektorów i adaptuje wagi."""
        vectors = np.ascontiguousarray(np.asarray(stream_vectors, dtype=np.float64))
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        losses = []
        total_time_ms = 0.0

        for i in range(vectors.shape[0]):
            row = vectors[i : i + 1]
            loss, ms = self.adapt_step(row)
            losses.append(loss)
            total_time_ms += ms

        final_repr = self.forward(vectors).mean(axis=0).tolist()
        w_norm = float(np.linalg.norm(self.w_ttt))

        return {
            "compressed_latent": final_repr,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "steps_adapted": len(losses),
            "w_ttt_norm": w_norm,
            "total_adaptation_ms": total_time_ms,
        }

    async def compress_stream_async(self, stream_vectors: Union[List[List[float]], np.ndarray]) -> Dict[str, Any]:
        """Asynchroniczna kompresja strumienia w puli wątków."""
        return await asyncio.to_thread(self.compress_stream, stream_vectors)

    def reset_session(self) -> None:
        """Reset wag plastycznych dla nowego epizodu."""
        self.w_ttt.fill(0.0)
        self.step_count = 0
        self.total_energy = 0.0
        self.last_update_timestamp = time.time()
