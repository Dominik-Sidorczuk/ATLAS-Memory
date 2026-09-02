from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from atlas_memory.causal.retro_causal_edge import RetroCausalEngine
from atlas_memory.l3_procedural.sleep_baker import SleepBaker, Step
from atlas_memory.quantization.rabitq_engine import RaBitQEngine
from atlas_memory.server.atlas_daemon import AtlasDaemon


@pytest.mark.asyncio
async def test_v37_auto_consolidate_and_bake(tmp_path: Path):
    """Test SleepBaker end-to-end auto consolidation and Hermes skill registration."""
    baker = SleepBaker(min_frequency=2, min_success_rate=0.8)

    trajectories = [
        {
            "trajectory_id": "traj_1",
            "steps": [
                Step(tool_name="terminal", params_pattern={"cmd": "git status"}),
                Step(tool_name="pytest", params_pattern={"args": "-q"}),
            ],
            "success": True,
        },
        {
            "trajectory_id": "traj_2",
            "steps": [
                Step(tool_name="terminal", params_pattern={"cmd": "git status"}),
                Step(tool_name="pytest", params_pattern={"args": "-q"}),
            ],
            "success": True,
        },
    ]

    mock_kv = AsyncMock()
    mock_kv.set_state = AsyncMock(return_value={"status": "ok"})

    skills_dir = tmp_path / "skills"
    results = await baker.auto_consolidate_and_bake(
        trajectories=trajectories,
        kv_store=mock_kv,
        skills_dir=skills_dir,
    )

    assert len(results) == 1
    assert results[0]["success_rate"] == 1.0
    assert results[0]["invocations_count"] == 2
    assert results[0]["skill_registered"] is True

    # Check skill files were written to atlas_procedural root
    sop_id = results[0]["sop_id"]
    skill_dir = skills_dir / "atlas_procedural" / sop_id
    assert skill_dir.exists()
    assert (skill_dir / "SKILL.md").exists()
    assert (skill_dir / "scripts" / "handler.py").exists()


@pytest.mark.asyncio
async def test_v37_recalibrate_graph_with_annealer():
    """Test RetroCausalEngine graph auto-recalibration via CausalAnnealer."""
    class MockGraph:
        def __init__(self):
            self.relations = [
                {"subject": "A", "predicate": "depends_on", "object": "B", "confidence": 0.8},
                {"subject": "B", "predicate": "depends_on", "object": "C", "confidence": 0.9},
            ]
            self.updated = []

        async def get_subgraph_relations(self, entities, max_depth=2):
            return {"relations": self.relations, "active_roots": entities}

        def add_relation(self, src, pred, tgt, confidence=1.0):
            self.updated.append((src, pred, tgt, confidence))

    g = MockGraph()
    engine = RetroCausalEngine(graph_client=g)

    res = await engine.recalibrate_graph_with_annealer(target_entity="A", max_depth=2)

    assert res["status"] == "ok"
    assert res["updated_edges"] == 2
    assert len(g.updated) == 2


@pytest.mark.asyncio
async def test_v37_active_sensing_triggers_auto_anneal():
    """Test AtlasDaemon _handle_active_sensing spawns background causal graph annealing upon discrepancy."""
    mock_active = MagicMock()
    mock_active.register_expectation = MagicMock()

    # Return prediction error
    from atlas_memory.active.prediction_error import PredictionError
    err = PredictionError(
        check_id="chk_srv_01",
        target_entity="srv_01",
        predicate="status",
        expected_value="running",
        observed_value="crashed",
        discrepancy_score=0.9,
        severity="CRITICAL",
    )
    mock_active.detect_discrepancy = MagicMock(return_value=err)

    mock_causal = MagicMock()
    mock_causal.recalibrate_graph_with_annealer = AsyncMock(return_value={"status": "ok", "updated_edges": 1})

    daemon = AtlasDaemon(
        active_sensing=mock_active,
        causal_engine=mock_causal,
    )

    res = await daemon._handle_active_sensing({
        "target_entity": "srv_01",
        "observed_predicate": "status",
        "observed_value": "crashed",
        "expected_value": "running",
    })

    assert res["status"] == "ok"
    assert res["has_error"] is True
    assert res["anneal_triggered"] is True

    # Give event loop a tick to process background task
    await asyncio.sleep(0.01)
    mock_causal.recalibrate_graph_with_annealer.assert_called_once_with(target_entity="srv_01")


@pytest.mark.asyncio
async def test_v37_daemon_trigger_sleep_consolidation_rpc(tmp_path: Path):
    """Test AtlasDaemon _handle_trigger_sleep_consolidation RPC method."""
    mock_engine = MagicMock()
    mock_engine.kv = AsyncMock()
    mock_engine.auditor = MagicMock()
    mock_engine.auditor.run_sleep_cycle_consolidation = AsyncMock(
        return_value=MagicMock(consolidated_records=3)
    )
    mock_engine.trajectory_buffer = MagicMock(trajectories=[])

    daemon = AtlasDaemon(engine=mock_engine)

    res = await daemon._handle_trigger_sleep_consolidation({
        "skills_dir": str(tmp_path / "skills"),
    })

    assert res["status"] == "ok"
    assert res["consolidated_records"] == 3
    assert res["baked_sops_count"] == 0


def test_v37_rabitq_fast_lut_scan():
    """Test RaBitQ fast LUT-based asymmetric batch distance scanning."""
    dim = 64
    n_vecs = 500
    np.random.seed(42)

    data = np.random.randn(n_vecs, dim).astype(np.float32)
    engine = RaBitQEngine(dim=dim, bits=4, seed=42)
    quantized = engine.quantize(data)

    query = np.random.randn(dim).astype(np.float32)

    top_k_standard = engine.fast_asymmetric_scan(query, quantized, top_k=10)
    top_k_lut = engine.fast_lut_asymmetric_scan(query, quantized, top_k=10)

    assert len(top_k_lut) == 10
    # Both methods should return identical or near-identical top ranked indices
    intersection = len(set(top_k_standard.tolist()).intersection(set(top_k_lut.tolist())))
    assert intersection >= 8
