from __future__ import annotations

import numpy as np

from atlas_memory.quantization.rabitq_engine import RaBitQEngine


def test_rabitq_odd_dimensions_quantization():
    """Verify RaBitQ quantizes and scans odd-dimensional vectors (e.g. d=65, 127)."""
    for dim in [3, 17, 65, 127]:
        n_vectors = 50
        data = np.random.randn(n_vectors, dim).astype(np.float32)

        engine = RaBitQEngine(dim=dim, bits=4, seed=42)
        quantized = engine.quantize(data)

        assert quantized.original_dim == dim
        assert quantized.n_vectors == n_vectors

        query = np.random.randn(dim).astype(np.float32)
        top_k = engine.fast_lut_asymmetric_scan(query, quantized, top_k=5)

        assert len(top_k) == 5
        assert all(0 <= idx < n_vectors for idx in top_k)


def test_rabitq_zero_vectors_and_zero_query():
    """Verify RaBitQ handles zero-magnitude vectors without NaN or division-by-zero crashes."""
    dim = 32
    data = np.zeros((10, dim), dtype=np.float32)
    data[0, 0] = 1.0  # At least one non-zero

    engine = RaBitQEngine(dim=dim, bits=4, seed=42)
    quantized = engine.quantize(data)

    # Zero query
    zero_query = np.zeros(dim, dtype=np.float32)
    top_k = engine.fast_lut_asymmetric_scan(zero_query, quantized, top_k=3)
    assert len(top_k) == 3


def test_rabitq_1bit_and_2bit_modes():
    """Verify RaBitQ runs in 1-bit and 2-bit quantization modes."""
    dim = 64
    n_vecs = 30
    data = np.random.randn(n_vecs, dim).astype(np.float32)

    for bits in [1, 2]:
        engine = RaBitQEngine(dim=dim, bits=bits, seed=10)
        quantized = engine.quantize(data)
        assert quantized.bits == bits

        query = np.random.randn(dim).astype(np.float32)
        top_k = engine.fast_asymmetric_scan(query, quantized, top_k=5)
        assert len(top_k) == 5
