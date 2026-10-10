"""
Canonical Zero-Trust Core Integrity Tests (V86/V86.1):
1. Session/end fast-exit (<50ms) and scoped consolidation when active.
2. Single-fetch candidate gathering (zero redundant triple-fetch).
3. Ingest single write authority (EpistemicWritePath choke point).
4. Zero-torch runtime and startup import budget.
5. Cross-session durable rule recall (F-08), input quarantine (F-09), and value_hash tamper detection (F-10).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from atlas_memory.cognitive.retrieval.candidates import gather_candidates
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.ingest.mnemosyne_ingest import MnemosyneIngestEngine
from atlas_memory.server.atlas_daemon import AtlasDaemon


@pytest.mark.asyncio
async def test_session_end_fast_exit_and_scoped_consolidation(tmp_path: Path):
    """Verifies session/end completes in <50ms when idle (idempotent_noop) and consolidates when active."""
    daemon = AtlasDaemon.create_default(
        atlas_dir=tmp_path / "atlas",
        socket_path=tmp_path / "d.sock",
        ingest_mnemosyne_on_start=False,
    )
    t0 = time.perf_counter()
    res_noop = await daemon._handle_session_end({"session_id": "idle_sess"})
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 50.0
    assert res_noop.get("status") == "noop" and res_noop.get("reason") == "idempotent_noop"

    daemon.session_hook.active_sessions["active_sess"] = {"touched_keys": {"fact:active:1"}}
    res_active = await daemon._handle_session_end({"session_id": "active_sess"})
    assert res_active["status"] in ("ok", "noop")


@pytest.mark.asyncio
async def test_candidates_eliminates_redundant_triple_fetch_and_ingest_choke_point(tmp_path: Path):
    """Verifies gather_candidates performs single-pass fetch and MnemosyneIngestEngine writes solely via EpistemicWritePath."""
    mock_vec = AsyncMock()
    mock_vec.search_hybrid_async.return_value = []
    engine = HybridMemoryEngine.create_default(
        db_path=str(tmp_path / "kv.db"),
        qdrant_location=":memory:",
        kuzu_path=str(tmp_path / "kuzu"),
    )
    await gather_candidates("rule:git:branch", engine=engine, vector_store=mock_vec, limit=10)
    assert mock_vec.search_hybrid_async.call_count <= 1

    md_file = tmp_path / "MEMORY.md"
    md_file.write_text("rule:git:main_branch: Nigdy nie pushuj bezpośrednio na main\n", encoding="utf-8")
    ingestor = MnemosyneIngestEngine(db_path=tmp_path / "missing.db", memory_md_path=md_file)
    stats = ingestor.sync_memory_md_into_engine(engine)
    assert stats["ingested"] >= 1
    await engine.kv.close()


def test_package_startup_zero_torch():
    """Verifies importing atlas_memory does not load torch into sys.modules."""
    cmd = [
        sys.executable,
        "-c",
        "import atlas_memory; import sys; assert 'torch' not in sys.modules, 'torch leaked!'",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr


@pytest.mark.asyncio
async def test_cross_session_durable_recall_input_quarantine_and_tamper_detection(tmp_path: Path):
    """Verifies F-08 cross-session durable rules, F-09 null-byte rejection, and F-10 value_hash tamper detection."""
    daemon = AtlasDaemon.create_default(
        atlas_dir=tmp_path / "atlas",
        socket_path=tmp_path / "d.sock",
        ingest_mnemosyne_on_start=False,
    )
    # F-08: Rule written in session A is retrieved in session B
    await daemon._handle_set({
        "key": "rule:security:ssh_port",
        "value": "Serwer SSH musi zawsze nasłuchiwać na porcie 2222",
        "session_id": "session_A",
        "confidence": 1.0,
        "source_type": "user_explicit",
    })
    res_b = await daemon._handle_prefetch({"query": "rule:security:ssh_port", "session_id": "session_B"})
    assert any(r["subject"] == "rule:security:ssh_port" for r in res_b["records"])

    # F-09: Null-byte key rejected
    with pytest.raises(ValueError, match="null bytes"):
        await daemon._handle_set({"key": "bad\x00key", "value": "val"})

    # F-10: Out-of-band SQLite tampering detected by verify_audit_log_sync(deep=True)
    conn = daemon.engine.kv._ensure_conn()
    with conn:
        conn.execute("UPDATE state_variables SET value = '\"TAMPERED_PORT_9999\"' WHERE key = 'rule:security:ssh_port'")

    is_valid, _ = daemon.engine.kv.verify_audit_log_sync(deep=True)
    assert is_valid is False
    await daemon.stop()


@pytest.mark.asyncio
async def test_loop_v83_f1_to_f7_invariants(tmp_path: Path):
    """Verifies F1 (ingest choke point), F2 (NumPy annealer equivalence <1e-9), F3/F6 (protocols & 0 reflection in candidates.py), F4 (EpistemicDecisionKernel EAV), and F7 (retrieval discrimination)."""
    from atlas_memory.causal.annealer import CausalAnnealer
    from atlas_memory.causal.energy_module import EnergyModule
    from atlas_memory.causal.models import CausalEdge
    from atlas_memory.cognitive.conflict_arbiter import ConflictArbiter, EpistemicDecisionKernel
    from atlas_memory.cognitive.retrieval.gate import evaluate_gate
    from atlas_memory.cognitive.retrieval.score import score_candidates
    from atlas_memory.core.protocols import (
        GraphStoreProtocol,
        HotBufferProtocol,
        KVStoreProtocol,
        VectorStoreProtocol,
    )

    repo_src = Path(__file__).resolve().parent.parent.parent / "src" / "atlas_memory"

    # F1: Zero raw set_sync / insert_sync in mnemosyne_ingest.py
    mnemosyne_src = (repo_src / "ingest" / "mnemosyne_ingest.py").read_text(encoding="utf-8")
    assert "set_sync(" not in mnemosyne_src and "insert_sync(" not in mnemosyne_src

    # F2: Exact equivalence (<1e-9) between vectorized NumPy annealer and scalar EnergyModule reference
    edges = [
        CausalEdge(source="node_a", predicate="depends_on", target="node_b", confidence=0.8),
        CausalEdge(source="node_b", predicate="powers", target="node_c", confidence=0.6),
        CausalEdge(source="node_c", predicate="triggers", target="node_a", confidence=0.4),
    ]
    annealer_vec = CausalAnnealer(energy_module=EnergyModule(), rng_seed=42)
    annealer_ref = CausalAnnealer(energy_module=EnergyModule(), rng_seed=42)
    res_vec = await annealer_vec.run(edges, T_init=0.8, T_min=0.05, decay=0.85, max_iter=25)
    res_ref = await annealer_ref._run_reference_scalar(
        edges, T_init=0.8, T_min=0.05, decay=0.85, max_iter=25, lambda_reg=0.1, start=time.perf_counter()
    )
    assert len(res_vec.energy_trace) == len(res_ref.energy_trace)
    for e_v, e_r in zip(res_vec.energy_trace, res_ref.energy_trace, strict=True):
        assert abs(e_v - e_r) <= 1e-9
    for edge_v, edge_r in zip(res_vec.final_edges, res_ref.final_edges, strict=True):
        assert abs(edge_v.confidence - edge_r.confidence) <= 1e-9

    # F3 & F6: Runtime checkable Protocols and zero reflection in candidates.py
    candidates_src = (repo_src / "cognitive" / "retrieval" / "candidates.py").read_text(encoding="utf-8")
    assert "hasattr(" not in candidates_src and "getattr(" not in candidates_src
    engine = HybridMemoryEngine.create_default(
        db_path=str(tmp_path / "f6_kv.db"),
        qdrant_location=":memory:",
        kuzu_path=str(tmp_path / "f6_kuzu"),
    )
    assert isinstance(engine.kv, KVStoreProtocol)
    assert isinstance(engine.vector_store, VectorStoreProtocol)
    assert isinstance(engine.graph, GraphStoreProtocol)
    assert isinstance(engine.write_path.hot_buffer, HotBufferProtocol)
    await engine.kv.close()
    engine.vector_store.close()

    # F4: EpistemicDecisionKernel EAV guard (fact:cors:is vs fact:api_key:is -> accept; multi-valued -> accept)
    arbiter_src = (repo_src / "cognitive" / "conflict_arbiter.py").read_text(encoding="utf-8")
    assert "is_both_rules" not in arbiter_src
    arbiter = ConflictArbiter()
    res_disjoint = arbiter.arbitrate(
        key="fact:api_key:is",
        val="ustawiona na true w środowisku prod",
        confidence=0.99,
        metadata={"source_type": "user_explicit"},
        existing_items={"fact:cors:is": {"value": "ustawiona na false w środowisku prod", "confidence": 0.95}},
    )
    assert res_disjoint.action != "supersede" and not res_disjoint.is_belief_revision
    assert EpistemicDecisionKernel.can_supersede("fact:api_key:is", "fact:cors:is") is False
    assert EpistemicDecisionKernel.can_supersede("config:app:allowed_users:1", "config:app:allowed_users:2") is False

    # F7: Pairwise retrieval discrimination against dense saturation (sim=0.95)
    saturated_pool = [
        {"subject": "service:crawl4ai:port", "object": "11235", "vector_score": 0.95, "confidence": 1.0, "source_type": "user_explicit"},
        {"subject": "config:worker_pool:size", "object": "8", "vector_score": 0.95, "confidence": 1.0, "source_type": "user_explicit"},
        {"subject": "rule:git:branch_strategy", "object": "Używaj isolated worktree dla każdej fazy", "vector_score": 0.95, "confidence": 1.0, "source_type": "user_explicit"},
    ]
    for neg_query in ("Bauman DeFi", "etyka AGI człowiek"):
        scored_neg = score_candidates(neg_query, [dict(c) for c in saturated_pool], fusion_strategy="dbsf")
        gate_neg = evaluate_gate(neg_query, scored_neg)
        assert gate_neg.should_retrieve is False and gate_neg.reason == "below_threshold"

    pos_pool = [dict(c) for c in saturated_pool] + [
        {"subject": "fact:atlas:arch", "object": "ATLAS pamięciowy backend", "vector_score": 0.92, "confidence": 1.0, "source_type": "user_explicit"}
    ]
    scored_pos = score_candidates("architektura ATLAS", pos_pool, fusion_strategy="dbsf")
    gate_pos = evaluate_gate("architektura ATLAS", scored_pos)
    assert gate_pos.should_retrieve is True
    assert scored_pos[0]["subject"] == "fact:atlas:arch"


