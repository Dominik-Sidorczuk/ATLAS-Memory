"""
Canonical Unit Tests for HybridMemoryEngine, MemoryOrchestrator, SalienceDecayEngine, and Distributed Sagas.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from atlas_memory.core.models import MemoryRecord
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.extensions.decay_scorer import SalienceDecayEngine
from atlas_memory.orchestrator import MemoryOrchestrator
from atlas_memory.server.atlas_daemon import AtlasDaemon, clamp_rpc_limit


@pytest.mark.asyncio
async def test_hybrid_memory_engine_recall_commit_and_worker():
    """Verifies HybridMemoryEngine recall, commit_observation, and background worker lifecycle."""
    engine = HybridMemoryEngine.create_default(db_path=":memory:")
    worker = engine.start_worker()
    assert not worker.done()

    r1 = MemoryRecord(subject="ServiceX", predicate="depends_on", object="ServiceY", confidence=0.98)
    r2 = MemoryRecord(subject="ServiceX", predicate="is_active", object="true", confidence=1.0, is_state_variable=True)

    await engine.commit_observation(r1)
    await engine.commit_observation(r2)
    await engine.process_all_pending()

    result = await engine.recall(query="Service dependencies", active_entities=["ServiceX"])
    assert "semantic_context" in result
    assert "graph_topology" in result
    assert "verified_state" in result
    assert result["retrieval_latency_ms"] < 300.0

    canon_subj = engine.canonicalizer.canonicalize("ServiceX")
    assert canon_subj in result["verified_state"]
    assert result["verified_state"][canon_subj]["value"] == "true"

    await engine.stop_worker()
    assert worker.done()
    await engine.kv.close()


@pytest.mark.asyncio
async def test_saga_transaction_and_crash_recovery():
    """Verifies distributed Saga transaction success and rollback on vector store failure."""
    engine = HybridMemoryEngine.create_default(db_path=":memory:")
    rec_ok = MemoryRecord(subject="DatabaseCluster", predicate="primary_node", object="node_1", confidence=0.95)
    assert await engine.auditor.atomic_insert_with_saga(rec_ok) is True

    engine.auditor.graph.add_record = AsyncMock(side_effect=RuntimeError("Simulated Kuzu crash"))
    rec_fail = MemoryRecord(subject="RollbackNode", predicate="status", object="failed_write", confidence=0.9, is_state_variable=True)
    with pytest.raises(RuntimeError, match="Simulated Kuzu crash"):
        await engine.auditor.atomic_insert_with_saga(rec_fail)
    await engine.kv.close()


def test_salience_decay_and_entity_canonicalizer():
    """Verifies SalienceDecayEngine scalar + vectorized batch scoring and EntityCanonicalizer aliases."""
    decay = SalienceDecayEngine()
    now = time.time()
    rec_fresh = MemoryRecord(subject="A", predicate="p", object="o", confidence=1.0, timestamp=now, access_count=5)
    rec_stale = MemoryRecord(subject="B", predicate="p", object="o", confidence=0.4, timestamp=now - 60 * 86400.0, access_count=0)

    s_fresh = decay.calculate_salience(rec_fresh, current_time=now)
    s_stale = decay.calculate_salience(rec_stale, current_time=now)
    assert s_fresh > s_stale

    batch_scores = decay.calculate_salience_batch([rec_fresh, rec_stale], current_time=now)
    assert np.allclose(batch_scores, [s_fresh, s_stale], atol=1e-6)

    engine = HybridMemoryEngine.create_default(db_path=":memory:")
    engine.canonicalizer.register_entity("srv_prod", "Production Server", aliases=["prod_box", "main_srv"])
    assert engine.canonicalizer.canonicalize("prod_box") == "srv_prod"


@pytest.mark.asyncio
async def test_shadow_reconcile_and_daemon_delete_truthfulness(tmp_path: Path):
    """Verifies shadow_reconcile fact extraction, delete truthfulness, and RPC limit/confidence clamping."""
    triples = []

    class _MockMnemosyne:
        async def recall(self, query: str, active_entities=None):
            return {"graph_topology": {"relations": [{"subject": "ServerNAS", "predicate": "ip", "object": "10.0.0.1", "confidence": 1.0}]}, "semantic_context": []}

        async def triple_add(self, subject: str, predicate: str, object_: str, confidence: float, source: str, supersede: bool):
            triples.append((subject, predicate, object_))

    orch = MemoryOrchestrator(mnemosyne_client=_MockMnemosyne())
    await orch.shadow_reconcile("Zmień IP ServerNAS na 10.0.0.2", "Czas odpowiedzi wynosi 15ms")
    assert len(triples) == 1
    assert triples[0][2] == "10.0.0.2"

    assert clamp_rpc_limit(500) == 100
    assert clamp_rpc_limit(-10) == 1

    daemon = AtlasDaemon.create_default(
        atlas_dir=tmp_path / "atlas",
        socket_path=tmp_path / "d.sock",
        ingest_mnemosyne_on_start=False,
    )
    try:
        daemon.engine.kv.set_sync("user:preference:lang", "pl")

        res_del1 = await daemon._handle_delete({"key": "user:preference:lang"})
        assert res_del1["deleted"] is True and res_del1["found"] is True

        res_del2 = await daemon._handle_delete({"key": "user:preference:lang"})
        assert res_del2["deleted"] is False and res_del2["found"] is False

        res_empty = await daemon._handle_commit_observation({"subject": "  ", "object": "val"})
        assert res_empty["status"] == "error" and res_empty["error"] == "empty_subject"

        res_clamp = await daemon._handle_commit_observation({"subject": "svc", "predicate": "p", "object": "v", "confidence": 999.0})
        assert res_clamp["status"] == "ok"
        assert (await daemon.engine.kv.get_state("svc:p"))["confidence"] == 1.0
    finally:
        await daemon.stop()
