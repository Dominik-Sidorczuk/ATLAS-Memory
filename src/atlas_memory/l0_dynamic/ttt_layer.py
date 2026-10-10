from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from atlas_memory.l0_dynamic.ttt_kernels import numba_ttt_adapt_step, numba_ttt_forward


@dataclass(frozen=True)
class TTTCheckpoint:
    """Niezmienny zrzut stanu fazowego wag plastycznych L0 TTT dla odwracalności symplektycznej."""
    w_ttt: np.ndarray
    w_recon: np.ndarray
    step_count: int
    total_energy: float
    last_update_timestamp: float


class TTTCompressionResult(TypedDict):
    """Struktura wyniku kompresji strumienia wektorów L0."""
    compressed_latent: list[float]
    mean_loss: float
    steps_adapted: int
    w_ttt_norm: float
    total_adaptation_ms: float


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
        seed: int | None = 42,
    ) -> None:
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
        self.w_recon_base = self.w_recon.copy()
        self._lock = threading.Lock()

        # Rozgrzanie JIT (warmup)
        dummy_x = np.zeros((1, input_dim), dtype=np.float64)
        numba_ttt_adapt_step(dummy_x, self.w_base, self.w_ttt, self.w_recon, self.lr, self.weight_decay)
        self.reset_session()

    @property
    def effective_weights(self) -> np.ndarray:
        with self._lock:
            return self.w_base + self.w_ttt

    def restore_counters(self, step_count: int, total_energy: float = 0.0) -> None:
        """V87 (T3): odtwarza trwaly licznik krokow po restarcie procesu.

        Monotonicznie (``max``) — odtworzenie nigdy nie cofa licznika w dol, wiec
        straznik V57 nie raportuje "braku uczenia" na swiezo wystartowanym daemonie.
        """
        with self._lock:
            self.step_count = max(self.step_count, int(step_count))
            self.total_energy = max(self.total_energy, float(total_energy))

    def forward(self, x: list[float] | np.ndarray) -> np.ndarray:
        """Przejście w przód: h = tanh(x @ W_eff)."""
        x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        elif x_arr.ndim != 2:
            raise ValueError(f"Input must be 1D or 2D array, got {x_arr.ndim}D")
        if x_arr.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, got {x_arr.shape[-1]}"
            )
        with self._lock:
            return numba_ttt_forward(x_arr, self.w_base, self.w_ttt)

    def adapt_step(self, x: list[float] | np.ndarray) -> tuple[float, float]:
        """
        Krok optymalizacji online TTT (synchroniczny).
        Zwraca: (loss, czas_wykonania_ms)
        """
        x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        elif x_arr.ndim != 2:
            raise ValueError(f"Input must be 1D or 2D array, got {x_arr.ndim}D")
        if x_arr.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, got {x_arr.shape[-1]}"
            )

        with self._lock:
            start_t = time.perf_counter()

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

    def compress_stream(
        self,
        vectors: list[list[float]] | list[np.ndarray] | np.ndarray,
    ) -> TTTCompressionResult:
        """
        Kompresuje strumień wektorów poprzez sekwencyjną adaptację online L0 TTT.
        Zwraca skompresowaną reprezentację ukrytą (ostatni stan forward)
        wraz z metrykami adaptacji w strukturze TTTCompressionResult.
        """
        start_t = time.perf_counter()
        if len(vectors) == 0:
            with self._lock:
                w_norm = float(np.linalg.norm(self.w_ttt))
            return TTTCompressionResult(
                compressed_latent=[0.0] * self.hidden_dim,
                mean_loss=0.0,
                steps_adapted=0,
                w_ttt_norm=w_norm,
                total_adaptation_ms=0.0,
            )

        losses = []
        last_latent = None
        for vec in vectors:
            loss, _ = self.adapt_step(vec)
            losses.append(loss)
            last_latent = self.forward(vec)

        total_ms = (time.perf_counter() - start_t) * 1000.0
        latent_list = last_latent[0].tolist() if last_latent is not None else [0.0] * self.hidden_dim
        with self._lock:
            w_norm = float(np.linalg.norm(self.w_ttt))

        return TTTCompressionResult(
            compressed_latent=latent_list,
            mean_loss=float(np.mean(losses)) if losses else 0.0,
            steps_adapted=len(losses),
            w_ttt_norm=w_norm,
            total_adaptation_ms=total_ms,
        )

    def compress(
        self,
        vectors: list[list[float]] | list[np.ndarray] | np.ndarray,
    ) -> TTTCompressionResult:
        """Alias dla compress_stream."""
        return self.compress_stream(vectors)

    def capture_checkpoint(self) -> TTTCheckpoint:
        """Rejestruje punkt fazowy wag plastycznych dla kompensacji symplektycznej."""
        with self._lock:
            return TTTCheckpoint(
                w_ttt=self.w_ttt.copy(),
                w_recon=self.w_recon.copy(),
                step_count=self.step_count,
                total_energy=self.total_energy,
                last_update_timestamp=self.last_update_timestamp,
            )

    def restore_checkpoint(self, cp: TTTCheckpoint) -> None:
        """Cofa krok adaptacji i przywraca stan fazowy wag plastycznych (Phi_undo)."""
        with self._lock:
            np.copyto(self.w_ttt, cp.w_ttt)
            np.copyto(self.w_recon, cp.w_recon)
            self.step_count = cp.step_count
            self.total_energy = cp.total_energy
            self.last_update_timestamp = cp.last_update_timestamp

    def reset_session(self) -> None:
        """Reset wag plastycznych dla nowego epizodu."""
        with self._lock:
            self.w_ttt.fill(0.0)
            np.copyto(self.w_recon, self.w_recon_base)
            self.step_count = 0
            self.total_energy = 0.0
            self.last_update_timestamp = time.time()
