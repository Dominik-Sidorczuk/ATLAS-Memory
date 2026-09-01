"""
Unit tests for V24 Optimization Components (Knapsack packing, JIT Asymmetric Scan, Zero-copy tensors).
"""

from __future__ import annotations

import numpy as np

from atlas_memory.arrow_buffer.trajectory_buffer import ArrowTrajectoryBuffer
from atlas_memory.models import ActionPlan, EpistemicSource, LatentState, MemoryRecord, PredictedTransition
from atlas_memory.orchestrator import MemoryOrchestrator
from atlas_memory.quantization.rabitq_engine import RaBitQEngine


def test_epistemic_knapsack_packing_optimality():
    """Weryfikacja że Epistemic Knapsack Packing maksymalizuje gęstość w zadanym budżecie tokenów."""
    orchestrator = MemoryOrchestrator()

    # Zbiór faktów: niektóre długie z małym scorem, niektóre krótkie z dużym scorem
    records = [
        # Krótki i ważny (wysoka gęstość)
        (MemoryRecord(subject="auth", predicate="ip", object="10.0.0.1", source_type=EpistemicSource.USER_EXPLICIT), 1.0),
        # Długi i mało ważny (niska gęstość)
        (MemoryRecord(subject="log", predicate="msg", object="a" * 400, source_type=EpistemicSource.AGENT_INFERENCE), 0.5),
        # Krótki i średni
        (MemoryRecord(subject="db", predicate="port", object="5432", source_type=EpistemicSource.TOOL_OUTPUT), 0.85),
    ]

    res = orchestrator.apply_token_budget(records, max_tokens=50, strategy="knapsack")
    assert res["estimated_tokens"] <= 50
    selected_subjects = [f["subject"] for f in res["selected_facts"]]
    # Długi fakt powinien zostać pominięty na rzecz krótkich o wyższej gęstości
    assert "auth" in selected_subjects
    assert "db" in selected_subjects
    assert "log" not in selected_subjects


def test_rabitq_fast_asymmetric_scan_correctness():
    """Weryfikacja poprawności i szybkości równoległego jądra JIT w RaBitQ."""
    dim = 64
    n_vecs = 500
    engine = RaBitQEngine(dim=dim, bits=4, seed=42)

    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_vecs, dim)).astype(np.float32)
    query = data[10]  # Wektor 10 powinien być w top wynikach

    quantized = engine.quantize(data)
    top_indices = engine.fast_asymmetric_scan(query, quantized, top_k=5)

    assert len(top_indices) == 5
    assert 10 in top_indices[:3]


def test_arrow_trajectory_zero_copy_tensor():
    """Weryfikacja bezkopiowej konwersji trajektorii Arrow do tensora NumPy."""
    buf = ArrowTrajectoryBuffer(state_dim=16)
    for i in range(20):
        vec = [float(i) * 0.1] * 16
        trans = PredictedTransition(
            previous_state=LatentState(step_index=i, timestamp=float(i), vector=vec, dimension=16),
            action=ActionPlan(name=f"act_{i}"),
            predicted_state=LatentState(step_index=i + 1, timestamp=float(i + 1), vector=vec, dimension=16),
            simulated_reward=0.5,
            uncertainty=0.1,
        )


        buf.append_transition(trans)

    tensor = buf.to_zero_copy_tensor()
    assert tensor.shape == (20, 16)
    assert np.allclose(tensor[0], [0.0] * 16)
    assert np.allclose(tensor[5], [0.5] * 16)
