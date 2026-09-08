from __future__ import annotations

from typing import Union

import numpy as np

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _popcnt64_py(x: Union[int, np.uint64]) -> int:
    """Pure-Python popcount fallback."""
    return bin(int(x)).count("1")


def _hamming_distance_py(a: np.ndarray, b: np.ndarray) -> int:
    """Pure-Python / NumPy hamming distance fallback."""
    dist = 0
    for i in range(len(a)):
        dist += _popcnt64_py(int(a[i] ^ b[i]))
    return dist


def _batch_hamming_py(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Pure-Python / NumPy batch hamming fallback."""
    num_candidates = candidates.shape[0]
    dists = np.zeros(num_candidates, dtype=np.int64)
    for i in range(num_candidates):
        dists[i] = _hamming_distance_py(query, candidates[i])
    return dists


if HAS_NUMBA:
    @numba.njit(fastmath=True, inline="always")
    def _popcnt64_jit(x: np.uint64) -> int:
        """Numba-accelerated 64-bit popcount (SWAR bit-twiddling, compiles to native popcnt)."""
        x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
        x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
        x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        return int((x * np.uint64(0x0101010101010101)) >> np.uint64(56))

    @numba.njit(fastmath=True)
    def _hamming_distance_jit(a: np.ndarray, b: np.ndarray) -> int:
        """Oblicza odległość Hamminga między dwoma spakowanymi wektorami uint64."""
        dist = 0
        n = a.shape[0]
        for i in range(n):
            xor_val = a[i] ^ b[i]
            dist += _popcnt64_jit(xor_val)
        return dist

    @numba.njit(parallel=True, fastmath=True)
    def _batch_hamming_jit(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Zrównoleglony obliczanie odległości Hamminga dla całej partii kandydatów."""
        num_candidates = candidates.shape[0]
        n_words = query.shape[0]
        dists = np.zeros(num_candidates, dtype=np.int64)
        for i in numba.prange(num_candidates):
            d = 0
            for j in range(n_words):
                xor_val = query[j] ^ candidates[i, j]
                d += _popcnt64_jit(xor_val)
            dists[i] = d
        return dists


class SIMDHamming:
    """
    SWAR SIMD popcount via Numba JIT (kompiluje się do native POPCNT na CPU z obsługą; NIE są to jawne instrukcje AVX-512).
    Kalkulator odległości Hamminga wykorzystujący Numba JIT z paralelizacją (prange) do skanowania bazy wektorów,
    z graceful fallbackiem do czystego NumPy/Pythona w środowiskach bez Numba.
    """

    @staticmethod
    def popcount(x: Union[int, np.uint64]) -> int:
        """Zwraca liczbę zapalonych bitów w 64-bitowej liczbie całkowitej."""
        if HAS_NUMBA:
            return _popcnt64_jit(np.uint64(x))
        return _popcnt64_py(x)

    @staticmethod
    def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
        """
        Oblicza odległość Hamminga pomiędzy wektorami a i b (dtype=uint64).
        """
        if a.dtype != np.uint64 or b.dtype != np.uint64:
            a = a.astype(np.uint64)
            b = b.astype(np.uint64)
        if HAS_NUMBA:
            return _hamming_distance_jit(a, b)
        return _hamming_distance_py(a, b)

    @staticmethod
    def batch_hamming(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """
        Batch Hamming scan query przeciwko macierzy candidates (batch, num_uint64).
        Zwraca tablicę 1D z odległościami Hamminga.
        """
        if query.dtype != np.uint64:
            query = query.astype(np.uint64)
        if candidates.dtype != np.uint64:
            candidates = candidates.astype(np.uint64)
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
        if HAS_NUMBA:
            return _batch_hamming_jit(query, candidates)
        return _batch_hamming_py(query, candidates)
