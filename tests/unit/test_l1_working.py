from __future__ import annotations

import numpy as np

from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.l1_working.jepa_kernels import numba_jepa_step
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


def test_jepa_action_encoding_deterministic():
    """Weryfikacja że encode_action jest 100% deterministyczny na tym samym procesie i między restartami."""
    buf1 = JEPALatentBuffer(state_dim=16, action_dim=8, seed=42)
    buf2 = JEPALatentBuffer(state_dim=16, action_dim=8, seed=42)

    act = ActionPlan(name="db_query", parameters={"table": "users", "limit": 10})
    enc1 = buf1.encode_action(act)
    enc2 = buf2.encode_action(act)

    assert np.allclose(enc1, enc2)


def test_jepa_history_rolling_buffer():
    """Weryfikacja że historia stanów w JEPA nie rośnie w sposób nieograniczony (ochrona przed memory leak)."""
    buf = JEPALatentBuffer(state_dim=16, action_dim=8, seed=42, max_history=5)
    for i in range(20):
        buf.commit_state_transition(ActionPlan(name=f"step_{i}"))

    assert len(buf.history) == 5
    assert buf.history[-1].context_tags == ["step_19"]


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


def test_record_mental_transition_populates_arrow_buffer():
    """Weryfikuje zapis przez engine.record_mental_transition, sprawdzając bufor i strukturę trajectories."""
    engine = HybridMemoryEngine.create_default(db_path=":memory:", qdrant_location=":memory:")
    assert len(engine.trajectory_buffer) == 0
    assert engine.trajectory_buffer.count == 0

    action1 = ActionPlan(name="observe:color", parameters={"subject": "sky", "object": "blue"})
    engine.record_mental_transition(action1, session_id="session_alpha")

    assert len(engine.trajectory_buffer) == 1
    assert engine.trajectory_buffer.count == 1
    trajs = engine.trajectory_buffer.trajectories
    assert len(trajs) == 1
    assert trajs[0]["session_id"] == "session_alpha"
    assert trajs[0]["actions"] == ["observe:color"]
    assert len(trajs[0]["steps"]) == 1
    assert trajs[0]["steps"][0]["tool"] == "observe:color"
    assert trajs[0]["steps"][0]["action"] == "observe:color"
    assert trajs[0]["success"] is True

    # Kolejny krok w tej samej sesji oraz krok w innej sesji
    action2 = ActionPlan(name="observe:brightness", parameters={"subject": "sun", "object": "bright"})
    engine.record_mental_transition(action2, session_id="session_alpha")

    action3 = ActionPlan(name="sync_turn", parameters={"facts_count": 2})
    engine.record_mental_transition(action3, session_id="session_beta")

    assert len(engine.trajectory_buffer) == 3
    assert engine.trajectory_buffer.count == 3
    all_trajs = engine.trajectory_buffer.trajectories
    assert len(all_trajs) == 2

    alpha_trajs = engine.trajectory_buffer.get_session_trajectories("session_alpha")
    assert len(alpha_trajs) == 1
    assert alpha_trajs[0]["actions"] == ["observe:color", "observe:brightness"]
    assert len(alpha_trajs[0]["steps"]) == 2

    beta_trajs = engine.trajectory_buffer.get_session_trajectories("session_beta")
    assert len(beta_trajs) == 1
    assert beta_trajs[0]["actions"] == ["sync_turn"]


