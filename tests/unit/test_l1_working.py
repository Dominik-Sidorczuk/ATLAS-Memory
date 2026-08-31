"""
Unit Tests for Layer 1: Working Memory & JEPA World Model.
"""
from __future__ import annotations

from atlas_memory.l1_working.jepa_latent import JEPALatentBuffer
from atlas_memory.models import ActionPlan


def test_jepa_initial_state():
    buffer = JEPALatentBuffer(state_dim=32, action_dim=16)
    state = buffer.current_state
    assert state.dimension == 32
    assert len(state.vector) == 32
    assert state.step_index == 0


def test_jepa_predict_transition():
    buffer = JEPALatentBuffer(state_dim=32, action_dim=16)
    action = ActionPlan(name="query_database", parameters={"table": "users"})

    transition = buffer.predict_transition(buffer.current_state, action)
    assert transition.previous_state.step_index == 0
    assert transition.predicted_state.step_index == 1
    assert len(transition.predicted_state.vector) == 32
    # Stan bufora nie powinien ulec zmianie przed commitem
    assert buffer.current_state.step_index == 0


def test_jepa_rollout_and_selection():
    buffer = JEPALatentBuffer(state_dim=32, action_dim=16)
    seq1 = [
        ActionPlan(name="search_docs", parameters={"topic": "api"}),
        ActionPlan(name="call_tool", parameters={"cmd": "run"}),
    ]
    seq2 = [
        ActionPlan(name="fallback_help", parameters={}),
        ActionPlan(name="exit", parameters={}),
    ]

    best_seq, trajectory, best_score = buffer.select_best_action_trajectory([seq1, seq2])
    assert len(best_seq) == 2
    assert len(trajectory) == 2
    assert isinstance(best_score, float)


def test_jepa_l0_latent_injection():
    buffer = JEPALatentBuffer(state_dim=32)
    l0_vec = [0.5] * 32
    new_state = buffer.inject_l0_latent(l0_vec)
    assert new_state.step_index == 1
    assert "l0_injected" in new_state.context_tags
import os
import tempfile

from atlas_memory.arrow_buffer.trajectory_buffer import HAS_PYARROW, ArrowTrajectoryBuffer
from atlas_memory.models import LatentState, PredictedTransition


def test_arrow_trajectory_buffer_zero_copy_and_parquet():
    buf = ArrowTrajectoryBuffer(state_dim=32)

    # Dodaj kilka przejść
    for i in range(5):
        s_prev = LatentState(vector=[0.1 * i] * 32, dimension=32, step_index=i)
        action = ActionPlan(name=f"action_{i}", parameters={"param": i})
        s_next = LatentState(vector=[0.2 * i] * 32, dimension=32, step_index=i + 1)
        trans = PredictedTransition(
            previous_state=s_prev,
            action=action,
            predicted_state=s_next,
            simulated_reward=1.0 + i,
            uncertainty=0.05,
        )
        buf.append_transition(trans, session_id="test_session_1")

    assert len(buf) == 5

    # 1. Bezkopiowa macierz NumPy
    matrix = buf.to_numpy_latent_matrix()
    assert matrix.shape == (5, 32)
    assert matrix[0, 0] == 0.0
    assert matrix[1, 0] == 0.2

    # 2. Apache Arrow Table & Parquet I/O
    if HAS_PYARROW:
        table = buf.to_arrow_table()
        assert table is not None
        assert table.num_rows == 5
        assert "latent_vector" in table.column_names

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            assert buf.dump_to_parquet(tmp_path) is True
            loaded_buf = ArrowTrajectoryBuffer.load_from_parquet(tmp_path, state_dim=32)
            assert len(loaded_buf) == 5
            loaded_mat = loaded_buf.to_numpy_latent_matrix()
            assert loaded_mat.shape == (5, 32)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

