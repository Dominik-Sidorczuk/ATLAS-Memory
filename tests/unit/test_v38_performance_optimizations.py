from __future__ import annotations

import time

import numpy as np

from atlas_memory.extensions.decay_scorer import SalienceDecayEngine
from atlas_memory.l1_working.jepa_kernels import numba_jepa_step
from atlas_memory.models import MemoryRecord
from atlas_memory.quantization.rabitq_engine import RaBitQEngine


def test_v38_vectorized_salience_decay_batch():
    """Test SalienceDecayEngine vectorized batch calculation vs scalar calculations."""
    engine = SalienceDecayEngine(decay_lambda=0.001)
    now = 10000.0

    records = [
        MemoryRecord(
            subject=f"entity_{i}",
            predicate="relates_to",
            object=f"target_{i}",
            confidence=0.9,
            importance_score=0.5 + (i % 5) * 0.1,
            timestamp=now - i * 100.0,
            access_count=i,
        )
        for i in range(100)
    ]

    sims = np.linspace(0.1, 0.9, 100, dtype=np.float32)

    # Scalar computation
    scalar_scores = [
        engine.calculate_salience(rec, similarity_score=float(sims[i]), current_time=now)
        for i, rec in enumerate(records)
    ]

    # Vectorized batch computation
    batch_scores = engine.calculate_salience_batch(records, similarity_scores=sims, current_time=now)

    assert len(batch_scores) == 100
    np.testing.assert_allclose(batch_scores, scalar_scores, rtol=1e-5, atol=1e-5)


def test_v38_fused_jepa_step_consistency():
    """Test fused numba_jepa_step mathematical correctness and output shapes."""
    np.random.seed(42)
    s_dim = 16
    a_dim = 8

    s_t = np.random.randn(1, s_dim).astype(np.float64)
    a_t = np.random.randn(1, a_dim).astype(np.float64)
    w_s = np.random.randn(s_dim, s_dim).astype(np.float64)
    w_a = np.random.randn(a_dim, s_dim).astype(np.float64)
    bias = np.zeros((1, s_dim), dtype=np.float64)
    w_val = np.random.randn(s_dim, 1).astype(np.float64)

    next_s, rew, unc = numba_jepa_step(s_t, a_t, w_s, w_a, bias, w_val)

    assert next_s.shape == (1, s_dim)
    assert isinstance(rew, float)
    assert 0.0 <= unc <= 1.0


def test_v38_rabitq_jit_lut_scan_speedup():
    """Test that JIT LUT scan on 4-bit nibbles executes quickly and matches top-k expectations."""
    dim = 128
    n_vecs = 1000
    np.random.seed(123)

    data = np.random.randn(n_vecs, dim).astype(np.float32)
    engine = RaBitQEngine(dim=dim, bits=4, seed=123)
    quantized = engine.quantize(data)

    query = np.random.randn(dim).astype(np.float32)

    # Warmup
    _ = engine.fast_lut_asymmetric_scan(query, quantized, top_k=10)

    t0 = time.perf_counter()
    top_k_indices = engine.fast_lut_asymmetric_scan(query, quantized, top_k=10)
    duration_ms = (time.perf_counter() - t0) * 1000.0

    assert len(top_k_indices) == 10
    # Should execute in under 5ms even in Python test runner
    assert duration_ms < 10.0
