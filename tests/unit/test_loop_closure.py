"""
Canonical Unit Tests for Closed-Loop Cognitive Wiring:
1. L0-TTT semantic observation vector projection (record_mental_transition -> _real_observation_vector).
2. Sleep consolidation causal annealing phase (_anneal_phase -> CausalAnnealer).
3. Active sensing attention producer -> consumer -> HotBuffer escalation.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from atlas_memory.core.models import ActionPlan
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.server.atlas_daemon import AtlasDaemon


def test_record_mental_transition_feeds_semantic_vector_to_ttt():
    """Verifies record_mental_transition projects ActionPlan into deterministic L2-normalized vector for TTT."""
    engine = HybridMemoryEngine.create_default(db_path=":memory:")
    a1 = ActionPlan(name="read_file", parameters={"path": "/etc/hosts"})
    a2 = ActionPlan(name="delete_db", parameters={"target": "prod"})

    v1 = engine._real_observation_vector(a1)
    v2 = engine._real_observation_vector(a2)
    assert len(v1) == engine.latent.state_dim
    assert v1 == engine._real_observation_vector(a1)
    assert v1 != v2

    steps_before = engine.ttt.step_count
    engine.record_mental_transition(a1, session_id="s1")
    assert engine.ttt.step_count == steps_before + 1


@pytest.mark.asyncio
async def test_daemon_anneal_phase_and_attention_loop():
    """Verifies _run_causal_anneal_phase delegation and active-sensing attention queue escalation into HotBuffer."""
    daemon = AtlasDaemon(engine=MagicMock(graph=None, latent=None, kv=None, causal_engine=None))

    recal = AsyncMock(return_value={
        "status": "ok", "iterations": 7, "updated_edges": 3,
        "converged": True, "duration_ms": 12.5,
    })
    daemon.causal_engine = SimpleNamespace(graph=object(), recalibrate_graph_with_annealer=recal)
    stats = SimpleNamespace()
    anneal_res = await daemon._run_causal_anneal_phase(stats, {"anneal_target": "Root"})
    assert anneal_res["status"] == "ok" and anneal_res["iterations"] == 7
    recal.assert_awaited_once()

    emitted = daemon._emit_attention_event({
        "metadata": {"writeback_key": "probe:DB:last_observed", "severity": "CRITICAL"},
        "prediction_error": {
            "target_entity": "DB",
            "severity": "CRITICAL",
            "discrepancy_score": 0.55,
            "observed": "5433",
            "expected": "5432",
        },
    })
    assert emitted is True
    ev = await daemon._attention_queue.get()
    daemon._apply_attention_event(ev)
    active_keys = [e.key for e in daemon.hot_buffer.get_active()]
    assert "attention:DB" in active_keys

