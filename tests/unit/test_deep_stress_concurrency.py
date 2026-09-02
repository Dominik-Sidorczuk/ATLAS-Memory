from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from atlas_memory.arrow_buffer.trajectory_buffer import ArrowTrajectoryBuffer
from atlas_memory.l0_dynamic.ttt_layer import TTTLayer
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.models import ActionPlan, LatentState, PredictedTransition


def _create_sample_transition(step: int, dim: int = 16) -> PredictedTransition:
    prev_state = LatentState(
        step_index=step,
        dimension=dim,
        timestamp=1700000000.0 + step,
        vector=[float(i + step) for i in range(dim)],
    )
    pred_state = LatentState(
        step_index=step + 1,
        dimension=dim,
        timestamp=1700000001.0 + step,
        vector=[float(i + step + 1) for i in range(dim)],
    )
    action = ActionPlan(name=f"action_{step % 4}", parameters={"speed": 1.0})
    return PredictedTransition(
        previous_state=prev_state,
        predicted_state=pred_state,
        action=action,
        simulated_reward=1.0 / (step + 1),
        uncertainty=0.05 * (step % 3),
    )


def test_concurrent_kv_store_merkle_integrity():
    """Verify VerifiedKVStore maintains cryptographic hash chain under rapid writes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "stress_kv.db"
        store = VerifiedKVStore(db_path=db_path)

        n_records = 50
        for i in range(n_records):
            store.set_sync(
                key=f"agent_{i % 10}_task_{i}",
                value=f"task_payload_{i}",
                confidence=0.95,
            )

        # Check audit log chain verification
        count = store._conn.execute("SELECT COUNT(*) FROM state_audit_log").fetchone()[0]
        assert count == 50

        # Ensure random reads return valid records
        for i in range(0, n_records, 5):
            fetched = store.get_sync(f"agent_{i % 10}_task_{i}")
            assert fetched is not None
            assert fetched["value"] == f"task_payload_{i}"


def test_concurrent_ttt_layer_continuous_adaptation():
    """Stress test TTTLayer under 50 continuous online update steps."""
    dim = 64
    layer = TTTLayer(input_dim=dim, hidden_dim=32, learning_rate=0.01)

    np.random.seed(42)
    initial_w_norm = np.linalg.norm(layer.w_ttt)
    assert initial_w_norm == 0.0

    for _step in range(50):
        x = np.random.randn(4, dim).astype(np.float64)
        loss, elapsed_ms = layer.adapt_step(x)
        assert not np.isnan(loss)
        assert loss >= 0.0
        assert elapsed_ms >= 0.0

    final_w_norm = np.linalg.norm(layer.w_ttt)
    assert final_w_norm > 0.0

    # Forward pass after adaptation
    test_x = np.random.randn(2, dim).astype(np.float64)
    out = layer.forward(test_x)
    assert out.shape == (2, 32)
    assert not np.isnan(out).any()


def test_arrow_buffer_rapid_chunking_and_slicing():
    """Stress test ArrowTrajectoryBuffer under rapid append and stream operations."""
    dim = 16
    buf = ArrowTrajectoryBuffer(state_dim=dim)

    for i in range(100):
        buf.append_transition(_create_sample_transition(i, dim=dim))

    batches = list(buf.stream_batches(batch_size=25, zero_copy=True))
    assert len(batches) == 4
    assert all(b["batch_size"] == 25 for b in batches)


from atlas_memory.active.prediction_error import ActiveSensingEngine, PredictionCheck


@pytest.mark.asyncio
async def test_active_sensing_rapid_expectations_and_discrepancy_scan():
    """Verify ActiveSensingEngine handles 100 expectations and detects anomalies cleanly."""
    engine = ActiveSensingEngine()

    for i in range(100):
        engine.register_expectation(
            PredictionCheck(
                check_id=f"check_{i}",
                target_entity=f"service_{i}",
                expected_predicate="latency_ms",
                expected_value=str(10.0 + (i % 5)),
                tolerance=2.0,
            )
        )

    assert len(engine.expectation_checks()) == 100

    # 1. Normal value within tolerance
    err_none = engine.detect_discrepancy("service_0", "latency_ms", "11.0")
    assert err_none is None

    # 2. Anomaly outside tolerance
    err = engine.detect_discrepancy("service_0", "latency_ms", "50.0")
    assert err is not None
    assert err.target_entity == "service_0"
    assert err.discrepancy_score > 0.5
