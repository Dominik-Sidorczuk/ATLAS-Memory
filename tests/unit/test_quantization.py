"""
Unit Tests for Hardware-Accelerated Vector Quantization (MIB 32x & SIMD Hamming).
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from atlas_memory.quantization import (
    MIBQuantizer,
    QuantizationConfig,
    QuantizedVector,
    SIMDHamming,
)


def test_mib_compression_ratio_384dim() -> None:
    """384 floats (1536B) -> 6 uint64 (48B) = 32x kompresji."""
    config = QuantizationConfig(target_dim=384)
    quantizer = MIBQuantizer(config)
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(384).astype(np.float32)

    qvec = quantizer.quantize(vec)
    assert isinstance(qvec, QuantizedVector)
    assert qvec.data.dtype == np.uint64
    assert qvec.data.shape == (6,)
    assert qvec.original_dim == 384

    raw_bytes_orig = vec.nbytes  # 384 * 4 = 1536
    raw_bytes_quant = qvec.data.nbytes  # 6 * 8 = 48
    ratio = raw_bytes_orig / raw_bytes_quant
    assert ratio == 32.0


def test_mib_compression_ratio_1536dim() -> None:
    """1536 floats (6144B) -> 24 uint64 (192B) = 32x kompresji."""
    config = QuantizationConfig(target_dim=1536)
    quantizer = MIBQuantizer(config)
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(1536).astype(np.float32)

    qvec = quantizer.quantize(vec)
    assert qvec.data.dtype == np.uint64
    assert qvec.data.shape == (24,)
    assert qvec.original_dim == 1536

    raw_bytes_orig = vec.nbytes  # 1536 * 4 = 6144
    raw_bytes_quant = qvec.data.nbytes  # 24 * 8 = 192
    ratio = raw_bytes_orig / raw_bytes_quant
    assert ratio == 32.0


def test_hamming_distance_identical() -> None:
    """Dla identycznych wektorów Hamming distance = 0."""
    quantizer = MIBQuantizer(QuantizationConfig(target_dim=384))
    rng = np.random.default_rng(123)
    vec = rng.standard_normal(384).astype(np.float32)
    qvec = quantizer.quantize(vec)

    dist = SIMDHamming.hamming_distance(qvec.data, qvec.data)
    assert dist == 0


def test_hamming_distance_opposite() -> None:
    """Dla zanegowanych bitów dystans Hamminga = liczba wymiarów."""
    a = np.array([0xFFFFFFFFFFFFFFFF] * 6, dtype=np.uint64)
    b = np.array([0x0000000000000000] * 6, dtype=np.uint64)

    dist = SIMDHamming.hamming_distance(a, b)
    assert dist == 384  # 6 * 64


def test_hamming_distance_symmetry() -> None:
    """Hamming distance jest symetryczny: dist(a, b) == dist(b, a)."""
    rng = np.random.default_rng(456)
    a = rng.integers(0, 0xFFFFFFFFFFFFFFFF, size=6, dtype=np.uint64)
    b = rng.integers(0, 0xFFFFFFFFFFFFFFFF, size=6, dtype=np.uint64)

    dist_ab = SIMDHamming.hamming_distance(a, b)
    dist_ba = SIMDHamming.hamming_distance(b, a)
    assert dist_ab == dist_ba


def test_quantization_ranking_preservation() -> None:
    """Ranking Hamming pre-filter zachowuje co najmniej 90% top-10 podobieństwa."""
    rng = np.random.default_rng(42)
    dim = 384
    num_candidates = 200

    query = rng.standard_normal(dim).astype(np.float32)
    query /= np.linalg.norm(query)

    # 10 semantycznie bliskich wektorów (query + mały szum) i 190 wektorów szumu tła
    near_candidates = np.array([
        (query + rng.standard_normal(dim).astype(np.float32) * 0.08)
        for _ in range(10)
    ])
    random_candidates = rng.standard_normal((num_candidates - 10, dim)).astype(np.float32)
    candidates = np.vstack([near_candidates, random_candidates])

    # Cosine similarity
    c_norm = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
    cosine_sims = c_norm @ query
    exact_top10 = set(np.argsort(-cosine_sims)[:10])

    # Hamming binarization
    quantizer = MIBQuantizer(QuantizationConfig(target_dim=dim))
    q_quant = quantizer.quantize(query)
    c_quant = quantizer.compress(candidates)

    hamming_dists = SIMDHamming.batch_hamming(q_quant.data, c_quant)
    # Wybieramy top 10 po Hammingu
    hamming_top10 = set(np.argsort(hamming_dists)[:10])

    overlap = len(exact_top10.intersection(hamming_top10)) / 10.0
    assert overlap >= 0.90, f"Ranking preservation overlap: {overlap} < 0.90"


def test_throughput_100k_vectors() -> None:
    """Throughput dla 100k 384-dim wektorów powinien być błyskawiczny (JIT/AVX-512/SIMD)."""
    num_u64 = 6
    num_candidates = 100_000


    rng = np.random.default_rng(999)
    query = rng.integers(0, 0xFFFFFFFFFFFFFFFF, size=num_u64, dtype=np.uint64)
    candidates = rng.integers(0, 0xFFFFFFFFFFFFFFFF, size=(num_candidates, num_u64), dtype=np.uint64)

    # Warm-up JIT
    _ = SIMDHamming.batch_hamming(query[:num_u64], candidates[:10])

    # Benchmark
    t0 = time.perf_counter()
    dists = SIMDHamming.batch_hamming(query, candidates)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0
    assert len(dists) == num_candidates
    # Assert execution is extremely fast (under 10ms in any modern hardware, typically < 1ms)
    assert elapsed_ms < 50.0  # Safe upper bound for VM test run, while target is << 10ms


def test_popcount_correctness() -> None:
    """Weryfikacja poprawności obliczania bit_count w SIMDHamming."""
    assert SIMDHamming.popcount(0b0) == 0
    assert SIMDHamming.popcount(0b1) == 1
    assert SIMDHamming.popcount(0b1010) == 2
    assert SIMDHamming.popcount(0b11111111) == 8
    assert SIMDHamming.popcount(0xFFFFFFFFFFFFFFFF) == 64
    assert SIMDHamming.popcount(0x5555555555555555) == 32
    assert SIMDHamming.popcount(0xAAAAAAAAAAAAAAAA) == 32


def test_batch_hamming_parallel() -> None:
    """Test równoległego batch_hamming dla 1000 kandydatów."""
    rng = np.random.default_rng(1001)
    query = rng.integers(0, 0xFFFFFFFFFFFFFFFF, size=6, dtype=np.uint64)
    candidates = rng.integers(0, 0xFFFFFFFFFFFFFFFF, size=(1000, 6), dtype=np.uint64)

    # Obliczenie referencyjne
    ref = np.array([SIMDHamming.hamming_distance(query, candidates[i]) for i in range(1000)])
    result = SIMDHamming.batch_hamming(query, candidates)

    np.testing.assert_array_equal(result, ref)


def test_mib_balanced_partition_skewed() -> None:
    """Test podziału MIB na skośnym rozkładzie lognormal (mean != median)."""
    rng = np.random.default_rng(777)
    # Rozkład silnie skośny: lognormal
    vec = rng.lognormal(mean=0.0, sigma=1.0, size=384).astype(np.float32)
    mean_val = float(np.mean(vec))
    median_val = float(np.median(vec))
    assert abs(mean_val - median_val) > 0.05, "Rozkład musi mieć mean != median"

    quantizer = MIBQuantizer(QuantizationConfig(target_dim=384))
    qvec = quantizer.quantize(vec)

    total_ones = sum(SIMDHamming.popcount(x) for x in qvec.data)
    assert total_ones == 384 // 2, f"Oczekiwano zbalansowanego podziału 192 bitów, otrzymano {total_ones}"


def test_mib_metadata_median_threshold() -> None:
    """Test metadanych median_threshold zapisywanych w QuantizedVector."""
    rng = np.random.default_rng(888)
    vec = rng.lognormal(mean=0.0, sigma=1.0, size=384).astype(np.float32)
    quantizer = MIBQuantizer(QuantizationConfig(target_dim=384))
    qvec = quantizer.quantize(vec)

    assert "median_threshold" in qvec.metadata
    np.testing.assert_allclose(qvec.metadata["median_threshold"], float(np.median(vec)), atol=1e-6)


def test_quantized_vector_wire_roundtrip() -> None:
    """Test serializacji i deserializacji wire format QuantizedVector."""
    rng = np.random.default_rng(999)
    vec = rng.standard_normal(384).astype(np.float32)
    quantizer = MIBQuantizer(QuantizationConfig(target_dim=384))
    qvec = quantizer.quantize(vec)
    qvec.metadata["custom_tag"] = "test_wire"

    blob = qvec.to_bytes()
    assert isinstance(blob, bytes)
    assert blob[:4] == b"QV01"

    restored = QuantizedVector.from_bytes(blob)
    np.testing.assert_array_equal(restored.data, qvec.data)
    assert restored.original_dim == qvec.original_dim
    assert restored.metadata == qvec.metadata


def test_mibquantizer_empty_embedding_raises_valueerror():
    """Empty (dim=0) embedding raises ValueError, not NotImplementedError."""
    q = MIBQuantizer()
    empty = np.array([], dtype=np.float32)
    with pytest.raises(ValueError, match="empty embeddings"):
        q.quantize(empty)
