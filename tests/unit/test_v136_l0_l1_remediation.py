"""Unit tests verifying Milestone v136.0: L0 Dynamic & L1 Working Memory remediation.

Acceptance criteria:
1. ttt_kernels.py & ttt_layer.py:
   - w_recon adapts along with w_ttt in numba_ttt_adapt_step.
   - Dimension check raises ValueError on dimension mismatch in forward/adapt_step.
   - reset_session resets w_recon to base state.
   - compress_stream / compress returns valid TTTCompressionResult.
2. jepa_latent.py & jepa_kernels.py:
   - Safe parameter sorting prevents TypeError on complex/mixed parameter types.
   - NumPy array encoding check prevents boolean evaluation ValueError.
   - predict_transition updates _current_state and appends to history (capped by max_history).
   - numba_jepa_adapt_step / adapt_transition adapts transition weights online.
3. hot_buffer.py:
   - update/append purges existing key first, preventing duplicates and premature eviction.
   - is_session_accessible treats effective_session=None as shared/accessible.
   - extract_effective_session correctly parses two-part keys (session:12345, session_12345).
   - superseded_history tracks superseded keys.
4. session_tracker.py:
   - PrefixCacheGuard handles non-serializable objects with default=str.
   - _reconcile_l1_to_l2 checks superseded_history and marks facts superseded without rollback_dual_node_record.
5. trajectory_buffer.py:
   - to_zero_copy_tensor and to_numpy_latent_matrix share memory with _latent_array.
   - trajectories success heuristic tolerates single negative rewards when overall trajectory succeeds.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from atlas_memory.core.models import ActionPlan, ConsolidationStats, LatentState
from atlas_memory.l0_dynamic.ttt_kernels import numba_ttt_adapt_step
from atlas_memory.l0_dynamic.ttt_layer import TTTCompressionResult, TTTLayer
from atlas_memory.l1_working.hot_buffer import (
    HotBufferPolicy,
    extract_effective_session,
    is_session_accessible,
)
from atlas_memory.l1_working.jepa_latent import JEPALatentBuffer
from atlas_memory.l1_working.session_tracker import HermesSessionHook, PrefixCacheGuard
from atlas_memory.l1_working.trajectory_buffer import ArrowTrajectoryBuffer

# ==============================================================================
# 1. TTT Kernels & TTTLayer Tests
# ==============================================================================


def test_ttt_w_recon_adaptation():
    """Verify that w_recon adapts during numba_ttt_adapt_step."""
    input_dim = 16
    hidden_dim = 8
    rng = np.random.default_rng(42)

    w_base = rng.standard_normal((input_dim, hidden_dim)) * 0.1
    w_ttt = np.zeros((input_dim, hidden_dim), dtype=np.float64)
    w_recon = rng.standard_normal((hidden_dim, input_dim)) * 0.1
    w_recon_orig = w_recon.copy()

    x = np.ones((1, input_dim), dtype=np.float64)
    loss = numba_ttt_adapt_step(x, w_base, w_ttt, w_recon, lr=0.1, weight_decay=0.0)

    assert loss > 0.0
    # w_recon must have changed from original
    assert not np.allclose(w_recon, w_recon_orig)


def test_ttt_layer_dimension_validation():
    """Verify that TTTLayer.forward and adapt_step raise ValueError on dimension mismatch."""
    layer = TTTLayer(input_dim=16, hidden_dim=8)

    # Wrong dimension 10 instead of 16
    invalid_x = np.zeros((1, 10))
    with pytest.raises(ValueError, match="Input dimension mismatch"):
        layer.forward(invalid_x)

    with pytest.raises(ValueError, match="Input dimension mismatch"):
        layer.adapt_step(invalid_x)


def test_ttt_layer_reset_session_restores_w_recon():
    """Verify that reset_session resets both w_ttt and w_recon."""
    layer = TTTLayer(input_dim=16, hidden_dim=8)
    initial_w_recon = layer.w_recon.copy()

    x = np.ones(16)
    layer.adapt_step(x)
    assert not np.allclose(layer.w_recon, initial_w_recon)
    assert np.linalg.norm(layer.w_ttt) > 0.0

    layer.reset_session()
    assert np.allclose(layer.w_recon, initial_w_recon)
    assert np.allclose(layer.w_ttt, 0.0)


def test_ttt_layer_compress_stream():
    """Verify compress_stream returns TTTCompressionResult with correct schema."""
    layer = TTTLayer(input_dim=16, hidden_dim=8)
    vectors = [np.random.randn(16).tolist() for _ in range(5)]

    res: TTTCompressionResult = layer.compress_stream(vectors)
    assert isinstance(res, dict)
    assert len(res["compressed_latent"]) == 8
    assert res["steps_adapted"] == 5
    assert res["mean_loss"] > 0.0
    assert res["w_ttt_norm"] > 0.0
    assert res["total_adaptation_ms"] >= 0.0


# ==============================================================================
# 2. JEPA Latent & Kernels Tests
# ==============================================================================


def test_jepa_action_parameter_safe_sorting():
    """Verify that action parameters with complex/mixed types do not raise TypeError."""
    buf = JEPALatentBuffer(state_dim=16, action_dim=8)
    # Dictionary and mixed values in parameters
    action = ActionPlan(
        name="complex_tool",
        parameters={"nested": {"b": 2}, "number": 100, "flag": True},
    )
    vec = buf._encode_action_vector(action)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (8,)


def test_jepa_onnx_array_truth_value_safe():
    """Verify that ONNX encoder returning ndarray does not raise ValueError on boolean check."""
    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = np.zeros(16, dtype=np.float32)

    buf = JEPALatentBuffer(state_dim=16, action_dim=8, encoder=mock_encoder)
    action = ActionPlan(name="tool_call", parameters={"arg": "value"})

    vec = buf._encode_action_vector(action)
    assert action.action_encoding == "onnx_embed"
    assert vec.shape == (8,)


def test_jepa_predict_transition_purity_and_step_history():
    """Verify that predict_transition is pure (What-If safe) and step updates _current_state."""
    buf = JEPALatentBuffer(state_dim=16, action_dim=8, max_history=3)
    curr = buf.current_state
    action1 = ActionPlan(name="hypothetical_1", parameters={})
    action2 = ActionPlan(name="hypothetical_2", parameters={})

    # Purity check: counterfactual What-If simulations from same state do not mutate _current_state
    pred1 = buf.predict_transition(curr, action1)
    pred2 = buf.predict_transition(curr, action2)
    assert buf.current_state == curr
    assert len(buf.history) == 1
    assert pred1.predicted_state != pred2.predicted_state

    # Explicit step advances state and history
    step1 = buf.step(action1)
    assert buf.current_state == step1.predicted_state
    assert len(buf.history) == 2

    # Add more steps to verify max_history capping
    buf.step(action1)
    buf.step(action2)
    assert len(buf.history) == 3


def test_jepa_adapt_transition():
    """Verify that numba_jepa_adapt_step and adapt_transition train weights online."""
    buf = JEPALatentBuffer(state_dim=8, action_dim=4)
    curr = buf.current_state
    action = ActionPlan(name="action_adapt", parameters={})
    target = np.ones(8) * 0.5

    loss1 = buf.adapt_transition(curr, action, target, lr=0.1)
    loss2 = buf.adapt_transition(curr, action, target, lr=0.1)
    assert loss1 > 0.0
    assert loss2 <= loss1


# ==============================================================================
# 3. Hot Buffer Tests
# ==============================================================================


def test_hot_buffer_update_purges_old_key():
    """Verify that update removes existing key before appending to prevent queue bloat."""
    buf = HotBufferPolicy(maxlen=3)
    buf.update("k1", "v1")
    buf.update("k2", "v2")
    buf.update("k1", "v1_updated")

    keys = [e.key for e in buf._buffer]
    assert keys == ["k2", "k1"]
    assert len(keys) == 2
    assert buf._buffer[-1].value == "v1_updated"


def test_hot_buffer_session_accessibility():
    """Verify that effective_session=None is accessible to all sessions."""
    assert is_session_accessible(None, "session_123") is True
    assert is_session_accessible(None, "default") is True
    assert is_session_accessible(None, "") is True
    assert is_session_accessible("s1", "s1") is True
    assert is_session_accessible("s1", "s2") is False


def test_hot_buffer_extract_effective_session():
    """Verify extraction of session keys including full Hermes sessions with multiple underscores."""
    assert extract_effective_session("session:12345") == "12345"
    assert extract_effective_session("session_67890") == "67890"
    assert extract_effective_session("session:custom:extra") == "custom"
    assert extract_effective_session("session_20261004_170710_6717e1") == "20261004_170710_6717e1"
    assert extract_effective_session("session_20261004_170710_6717e1:rule:db") == "20261004_170710_6717e1"
    eff_sess = extract_effective_session("session_20261004_170710_6717e1")
    assert is_session_accessible(eff_sess, "session_20261004_170710_6717e1") is True


def test_hot_buffer_superseded_history_tracking():
    """Verify that mark_superseded and reconcile_superseded track superseded_history."""
    buf = HotBufferPolicy(maxlen=5)
    buf.update("k1", "v1")
    buf.update("k2", "v2")

    buf.mark_superseded("k1")
    assert "k1" in buf.superseded_history
    assert len(buf._buffer) == 1

    buf.reconcile_superseded(["k2", "k3"])
    assert "k2" in buf.superseded_history
    assert "k3" in buf.superseded_history


# ==============================================================================
# 4. Session Tracker Tests
# ==============================================================================


def test_prefix_cache_guard_non_serializable():
    """Verify build_immutable_prefix handles non-serializable objects with default=str."""
    class CustomObj:
        def __str__(self):
            return "CustomObjStr"

    verified_state = {
        "key1": {"value": CustomObj()},
        "key2": {"value": {1, 2, 3}},
    }
    prefix = PrefixCacheGuard.build_immutable_prefix(verified_state)
    assert "CustomObjStr" in prefix
    assert "key1" in prefix
    assert "key2" in prefix


@pytest.mark.asyncio
async def test_session_tracker_reconcile_superseded_history():
    """Verify _reconcile_l1_to_l2 uses superseded_history and does not invoke rollback_dual_node_record."""
    mock_engine = MagicMock()
    mock_hot = HotBufferPolicy(maxlen=10)
    mock_hot.update("fact:old", "stale data")
    mock_hot.mark_superseded("fact:old")

    mock_engine.hot_buffer = mock_hot
    mock_engine.kv = MagicMock()
    mock_engine.kv.get_all_sync.return_value = {}
    mock_engine.kv.get_state_sync.return_value = {"key": "fact:old", "value": "stale data", "metadata": {}}
    mock_engine.vector_store = MagicMock()
    mock_engine.graph = MagicMock()

    hook = HermesSessionHook(mock_engine)
    stats = ConsolidationStats()
    await hook._reconcile_l1_to_l2("sess_test", {"fact:old"}, stats)

    # Verify fact:old was updated in KV as is_superseded=True
    assert mock_engine.kv.set_state_sync.called or mock_engine.kv.set_state.called
    # rollback_dual_node_record must NOT be called on graph
    assert not mock_engine.graph.rollback_dual_node_record.called
    assert stats.facts_superseded >= 1


# ==============================================================================
# 5. Trajectory Buffer Tests
# ==============================================================================


def test_trajectory_buffer_true_zero_copy():
    """Verify to_zero_copy_tensor and to_numpy_latent_matrix share memory with internal array."""
    buf = ArrowTrajectoryBuffer(state_dim=8)
    state = LatentState(vector=[1.0] * 8, dimension=8, step_index=1, timestamp=time.time())
    pred = MagicMock()
    pred.predicted_state = state
    pred.action = ActionPlan(name="action_1", parameters={})
    pred.simulated_reward = 1.0
    pred.uncertainty = 0.0

    buf.append_transition(pred)

    tensor_view = buf.to_zero_copy_tensor()
    matrix_view = buf.to_numpy_latent_matrix()

    assert np.shares_memory(tensor_view, buf._latent_array)
    assert np.shares_memory(matrix_view, buf._latent_array)


def test_trajectory_buffer_success_heuristic_resilient():
    """Verify that a single negative reward does not disqualify a successful trajectory."""
    buf = ArrowTrajectoryBuffer(state_dim=4)
    action = ActionPlan(name="tool", parameters={})

    # Steps with rewards: [1.0, 1.0, -0.2, 1.0] -> overall positive and ending positive
    for i, rew in enumerate([1.0, 1.0, -0.2, 1.0]):
        state = LatentState(vector=[0.0] * 4, dimension=4, step_index=i + 1, timestamp=time.time())
        pred = MagicMock()
        pred.predicted_state = state
        pred.action = action
        pred.simulated_reward = rew
        pred.uncertainty = 0.0
        buf.append_transition(pred, session_id="test_sess")

    trajs = buf.trajectories
    assert len(trajs) == 1
    assert trajs[0]["success"] is True
