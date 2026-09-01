from pathlib import Path

from atlas_memory.arrow_buffer.trajectory_buffer import ArrowTrajectoryBuffer
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


def test_arrow_trajectory_buffer_stream_batches_zero_copy():
    dim = 16
    buf = ArrowTrajectoryBuffer(state_dim=dim)
    for i in range(50):
        buf.append_transition(_create_sample_transition(i, dim=dim))

    batches = list(buf.stream_batches(batch_size=20, zero_copy=True))
    assert len(batches) == 3
    assert batches[0]["batch_size"] == 20
    assert batches[1]["batch_size"] == 20
    assert batches[2]["batch_size"] == 10

    assert batches[0]["latent_matrix"].shape == (20, dim)
    assert batches[0]["step_ids"].shape == (20,)
    assert len(batches[0]["action_names"]) == 20


def test_arrow_trajectory_buffer_zstd_compression_and_stream(tmp_path: Path):
    dim = 8
    buf = ArrowTrajectoryBuffer(state_dim=dim)
    for i in range(100):
        buf.append_transition(_create_sample_transition(i, dim=dim))

    parquet_file = tmp_path / "trajectory_zstd.parquet"
    ok = buf.dump_to_parquet(parquet_file, compression="zstd", compression_level=3)
    assert ok is True
    assert parquet_file.exists()

    # Stream from parquet
    streamed_batches = list(ArrowTrajectoryBuffer.stream_from_parquet(parquet_file, batch_size=32, state_dim=dim))
    assert len(streamed_batches) == 4
    assert sum(b["batch_size"] for b in streamed_batches) == 100
    assert streamed_batches[0]["latent_matrix"].shape == (32, dim)


def test_arrow_trajectory_buffer_compression_stats():
    dim = 32
    buf = ArrowTrajectoryBuffer(state_dim=dim)
    for i in range(500):
        buf.append_transition(_create_sample_transition(i, dim=dim), metadata={"turn": i, "agent": "hermes"})

    stats = buf.get_compression_stats()
    assert stats["total_records"] == 500
    assert stats["state_dim"] == 32
    assert stats["raw_estimated_ram_bytes"] > 0
    assert stats["arrow_columnar_bytes"] > 0
    assert stats["ram_efficiency_ratio"] > 0.5


def test_arrow_trajectory_buffer_empty_streaming():
    buf = ArrowTrajectoryBuffer(state_dim=16)
    batches = list(buf.stream_batches(batch_size=10))
    assert len(batches) == 0
