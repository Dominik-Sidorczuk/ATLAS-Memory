"""
Unit tests for PredictiveCoder (Closed-Loop Active Sensing) & HotBufferPolicy.

Validates:
1. PredictiveCoder:
   - Tolerance suppression (discrepancy <= tolerance -> has_error=False, severity='NONE', anneal=False, writeback=False).
   - Sensory noise gating (LOW severity -> has_error=True, anneal=False, writeback=False).
   - Closed-Loop Writeback: CRITICAL or MODERATE anomaly -> anneal triggered, writes probe:X:last_observed to KV,
     adds graph edge, and sets world_model_updated=True.
2. HotBufferPolicy:
   - Deduplication: updating an existing key updates the value and keeps a single entry.
   - Supersession purging: superseded keys are purged from active retrieval.
   - Session filtering & isolation.
   - Seed restoration from SQLite state_variables.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas_memory.cognitive.hot_buffer import HotBufferPolicy
from atlas_memory.cognitive.predictive_coder import PredictiveCoder
from atlas_memory.engine import HybridMemoryEngine


class MockCausalEngine:
    """Mock causal engine with annealer."""

    def __init__(self) -> None:
        self.recalibrate_called = False
        self.target_entity = None
        self.prediction_error = 0.0

    async def recalibrate_graph_with_annealer(self, *args, target_entity: str = "Root", prediction_error: float = 0.0, **kwargs):
        self.recalibrate_called = True
        self.target_entity = target_entity or kwargs.get("target_entity")
        self.prediction_error = prediction_error or kwargs.get("prediction_error", 0.0)
        return {
            "annealed": True,
            "iterations": 42,
            "energy_delta": -0.15,
            "converged": True,
        }


class MockGraphClient:
    """Mock graph client tracking added relations."""

    def __init__(self) -> None:
        self.relations = []

    def add_relation(self, subject: str, predicate: str, object: str, confidence: float = 1.0, timestamp: float = 0.0):
        self.relations.append({
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "confidence": confidence,
            "timestamp": timestamp,
        })


@pytest.mark.asyncio
async def test_predictive_coder_tolerance_suppression(tmp_path: Path):
    """Verifies that observation within tolerance causes no error, no anneal, and no writeback."""
    db_path = str(tmp_path / "test_pred_tol.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    causal = MockCausalEngine()
    coder = PredictiveCoder(engine=engine, causal_engine=causal)

    # 1480 vs 1500 with tolerance 50
    res = await coder.execute_sensing({
        "target_entity": "agent_speed",
        "observed_value": 1480,
        "expected_value": 1500,
        "tolerance": 50,
    })

    assert res["status"] == "ok"
    assert res["has_error"] is False
    assert res["severity"] == "NONE"
    assert res["discrepancy_score"] == 0.0
    assert res["anneal_triggered"] is False
    assert res["world_model_updated"] is False
    assert causal.recalibrate_called is False


@pytest.mark.asyncio
async def test_predictive_coder_sensory_noise_gating(tmp_path: Path):
    """Verifies that LOW severity discrepancy does NOT trigger auto-anneal or writeback."""
    db_path = str(tmp_path / "test_pred_noise.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    causal = MockCausalEngine()
    coder = PredictiveCoder(engine=engine, causal_engine=causal)

    # 1480 vs 1500 with no tolerance: diff = 20, rel_diff = 20/1500 = 0.0133 -> LOW severity
    res = await coder.execute_sensing({
        "target_entity": "agent_speed",
        "observed_value": 1480,
        "expected_value": 1500,
    })

    assert res["status"] == "ok"
    assert res["has_error"] is True
    assert res["severity"] == "LOW"
    assert res["anneal_triggered"] is False
    assert res["world_model_updated"] is False
    assert causal.recalibrate_called is False


@pytest.mark.asyncio
async def test_predictive_coder_closed_loop_writeback(tmp_path: Path):
    """
    Verifies that CRITICAL anomaly triggers auto-anneal AND performs closed-loop writeback:
    1. Writes probe:agent_speed:last_observed to KV store.
    2. Wires (agent_speed)-[observed_state]->(500) in graph store.
    3. Returns world_model_updated: True!
    """
    db_path = str(tmp_path / "test_pred_anomaly.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    causal = MockCausalEngine()
    graph_mock = MockGraphClient()
    coder = PredictiveCoder(engine=engine, causal_engine=causal)

    # 500 vs 1500: diff = 1000, rel_diff = 0.6667 -> CRITICAL severity
    res = await coder.execute_sensing({
        "target_entity": "agent_speed",
        "observed_value": 500,
        "expected_value": 1500,
    }, graph_client=graph_mock)

    assert res["status"] == "ok"
    assert res["has_error"] is True
    assert res["severity"] == "CRITICAL"
    assert res["anneal_triggered"] is True
    assert res["world_model_updated"] is True
    assert causal.recalibrate_called is True
    assert causal.target_entity == "agent_speed"

    # Verify KV store writeback
    kv_record = engine.kv.get_sync("probe:agent_speed:last_observed")
    assert kv_record is not None
    assert kv_record["value"] == 500
    assert kv_record["metadata"]["severity"] == "CRITICAL"
    assert kv_record["metadata"]["source_type"] == "sensor_observation"

    # Verify Graph relation writeback
    assert len(graph_mock.relations) == 1
    assert graph_mock.relations[0]["subject"] == "agent_speed"
    assert graph_mock.relations[0]["predicate"] == "observed_state"
    assert graph_mock.relations[0]["object"] == "500"


def test_hot_buffer_deduplication_and_supersession():
    """Verifies that HotBufferPolicy deduplicates updates and purges superseded entries."""
    hb = HotBufferPolicy(maxlen=5, ttl_seconds=60.0)

    # 1. Sequential updates to same key
    hb.update("key:counter", 1, metadata={"session_id": "s1"})
    hb.update("key:counter", 2, metadata={"session_id": "s1"})
    active = hb.get_active(session_id="s1")
    assert len(active) == 1
    assert active[0].value == 2

    # 2. Add second key
    hb.update("key:state", "active", metadata={"session_id": "s1"})
    active2 = hb.get_active(session_id="s1")
    assert len(active2) == 2

    # 3. Purge superseded key
    purged = hb.purge("key:counter")
    assert purged >= 1
    active3 = hb.get_active(session_id="s1")
    assert len(active3) == 1
    assert active3[0].key == "key:state"


def test_hot_buffer_seed_from_kv(tmp_path: Path):
    """Verifies that HotBufferPolicy restores recent state variables from SQLite."""
    db_path = str(tmp_path / "test_seed.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )

    # Seed KV store with 3 state variables
    engine.kv.set_sync("config:timeout", 30, metadata={"source_type": "user_explicit"})
    engine.kv.set_sync("rule:git_flow", "PR only", metadata={"source_type": "user_explicit"})
    engine.kv.set_sync("fact:project", "ATLAS Memory", metadata={"source_type": "user_explicit"})

    hb = HotBufferPolicy(maxlen=10)
    seeded_count = hb.seed_from_kv(engine.kv, limit=5)
    assert seeded_count == 3
    assert len(hb) == 3

    active = hb.get_active()
    keys = {item.key for item in active}
    assert "config:timeout" in keys
    assert "rule:git_flow" in keys
    assert "fact:project" in keys


def test_f4_predictive_coder_noise_calibration():
    """
    F4: Verifies numerical tolerance and micro-fluctuation (< 1e-5) calibration
    in PredictiveCoder.compute_discrepancy.
    """
    coder = PredictiveCoder()

    # Micro-fluctuation without tolerance: diff < 1e-5 -> NO ERROR
    res_micro = coder.compute_discrepancy(1500.000001, 1500.0)
    assert res_micro["has_error"] is False
    assert res_micro["severity"] == "NONE"

    # Real discrepancy without tolerance: diff = 20 > 1e-5 -> ERROR (LOW)
    res_noise = coder.compute_discrepancy(1480, 1500)
    assert res_noise["has_error"] is True
    assert res_noise["severity"] == "LOW"

    # Discrepancy within explicit tolerance: diff = 20 <= 50 -> NO ERROR
    res_tol = coder.compute_discrepancy(1480, 1500, tolerance=50.0)
    assert res_tol["has_error"] is False
    assert res_tol["severity"] == "NONE"

