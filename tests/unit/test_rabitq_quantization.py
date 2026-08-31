"""
Unit tests for RaBitQ Quantization Engine & Matryoshka Representation Learning (MRL).
"""
from __future__ import annotations

import numpy as np

from atlas_memory.quantization import (
    MatryoshkaEmbedding,
    MatryoshkaResult,
    RaBitQEngine,
    RaBitQResult,
)


def test_rabitq_quantize_preserves_shape() -> None:
    """Test sprawdza, czy n_vectors i original_dim są zachowane po kwantyzacji."""
    dim = 384
    n_vectors = 50
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n_vectors, dim)).astype(np.float32)

    engine = RaBitQEngine(dim=dim, bits=4, seed=42)
    res = engine.quantize(vectors)

    assert isinstance(res, RaBitQResult)
    assert res.n_vectors == n_vectors
    assert res.original_dim == dim
    assert res.bits == 4
    assert res.scale.shape == (n_vectors,)
    # Dla b=4 pakujemy 2 wartości na 1 bajt -> (dim + 1) // 2
    assert res.quantized_data.shape == (n_vectors, (dim + 1) // 2)


def test_rabitq_compression_ratio_b1_is_32x() -> None:
    """Współczynnik kompresji dla b=1 wynosi dokładnie 32.0x (float32 do 1-bit)."""
    engine = RaBitQEngine(dim=384, bits=1)
    assert engine.compression_ratio() == 32.0


def test_rabitq_compression_ratio_b4_is_8x() -> None:
    """Współczynnik kompresji dla b=4 wynosi dokładnie 8.0x (float32 do 4-bit)."""
    engine = RaBitQEngine(dim=384, bits=4)
    assert engine.compression_ratio() == 8.0


def test_rabitq_reconstruct_reasonable() -> None:
    """Rekonstrukcja dla losowych znormalizowanych wektorów osiąga MSE < 0.5."""
    dim = 384
    n_vectors = 20
    rng = np.random.default_rng(100)
    vectors = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms  # Jednostkowe wektory L2

    engine = RaBitQEngine(dim=dim, bits=4, seed=123)
    quantized = engine.quantize(vectors)
    reconstructed = engine.reconstruct(quantized)

    assert reconstructed.shape == vectors.shape
    mse = float(np.mean((vectors - reconstructed) ** 2))
    assert mse < 0.5


def test_rabitq_recall_at_k_b4_ge_0_70() -> None:
    """Recall@10 dla b=4 na zbiorze 100 wektorów wynosi >= 0.70."""
    dim = 384
    n_vectors = 100
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    engine = RaBitQEngine(dim=dim, bits=4, seed=42)
    quantized = engine.quantize(vectors, store_exact=True)

    query = vectors[0] + rng.standard_normal(dim).astype(np.float32) * 0.1
    recall = engine.recall_at_k(query=query, quantized=quantized, k=10)
    assert recall >= 0.70


def test_rabitq_deterministic_with_seed() -> None:
    """Ten sam seed daje identyczną macierz rotacji i skwantyzowane bity."""
    dim = 128
    rng = np.random.default_rng(77)
    vectors = rng.standard_normal((10, dim)).astype(np.float32)

    engine1 = RaBitQEngine(dim=dim, bits=4, seed=999)
    engine2 = RaBitQEngine(dim=dim, bits=4, seed=999)

    res1 = engine1.quantize(vectors)
    res2 = engine2.quantize(vectors)

    np.testing.assert_allclose(res1.rotation_matrix, res2.rotation_matrix)
    np.testing.assert_array_equal(res1.quantized_data, res2.quantized_data)
    np.testing.assert_allclose(res1.scale, res2.scale)


def test_rabitq_bound_optimality() -> None:
    """Weryfikacja teoretycznego boundu błędu kwantyzacji RaBitQ (SIGMOD'25)."""
    dim = 256
    rng = np.random.default_rng(2025)
    vectors = rng.standard_normal((5, dim)).astype(np.float32)

    engine = RaBitQEngine(dim=dim, bits=4, seed=42)
    for vec in vectors:
        bound = engine.theoretical_error_bound(vec)
        norm_sq = float(np.sum(vec ** 2))
        expected_bound = (norm_sq / float(dim)) * (1.0 - 1.0 / (2.0 ** (2 * 4)))
        assert np.isclose(bound, expected_bound, rtol=1e-5)
        assert bound > 0.0


def test_matryoshka_shortlist_then_rerank() -> None:
    """64-dim shortlist -> 1536-dim rerank poprawnie znajduje top-1 wektor."""
    dim = 1536
    n_corpus = 100
    rng = np.random.default_rng(42)
    corpus = rng.standard_normal((n_corpus, dim)).astype(np.float32)
    norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    corpus = corpus / norms

    # Zapytanie bardzo bliskie dokumentowi o indeksie 7
    target_idx = 7
    query = corpus[target_idx] + rng.standard_normal(dim).astype(np.float32) * 0.01

    mrl = MatryoshkaEmbedding(max_dim=1536, dimensions=[64, 128, 256, 512, 1536])
    final_indices = mrl.shortlist_then_rerank(query, corpus, top_k=5, shortlist_k=20)

    assert len(final_indices) == 5
    assert final_indices[0] == target_idx


def test_matryoshka_adaptive_search_achieves_recall() -> None:
    """Matryoshka adaptive_search osiąga recall >= recall_target (np. 0.95)."""
    dim = 512
    n_corpus = 50
    rng = np.random.default_rng(123)
    corpus = rng.standard_normal((n_corpus, dim)).astype(np.float32)
    norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    corpus = corpus / norms

    query = corpus[0] + rng.standard_normal(dim).astype(np.float32) * 0.05

    mrl = MatryoshkaEmbedding(max_dim=512, dimensions=[64, 128, 256, 512])
    result = mrl.adaptive_search(query, corpus, recall_target=0.95, max_candidates=50, k_eval=5)

    assert isinstance(result, MatryoshkaResult)
    assert result.recall_achieved >= 0.95
    assert len(result.final_indices) > 0
    assert result.n_candidates > 0
    assert len(result.dimensions_used) >= 1


def test_matryoshka_dimensions_monotonic() -> None:
    """Wymiary w MatryoshkaEmbedding są posortowane ściśle rosnąco i zawierają max_dim."""
    mrl = MatryoshkaEmbedding(max_dim=1536, dimensions=[256, 64, 512, 128])
    assert mrl.dimensions == [64, 128, 256, 512, 1536]
    assert mrl.max_dim == 1536
