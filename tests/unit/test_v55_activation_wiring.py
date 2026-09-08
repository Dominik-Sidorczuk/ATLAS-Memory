"""
Unit and integration tests for ATLAS V55 Activation Wiring (F1-F5).

Verifies:
1. F1: L0-TTT online adaptation triggered by record_mental_transition and reported in get_stats.
2. F2: Closed-loop task feedback (G8) boosting confidence in commit_observation and task_feedback RPC.
3. F3: Daemon startup auto-healing of broken SHA audit chain (D2).
4. F4: ADR-002 offline/dormant module integrity and status characterization.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.models import ActionPlan
from atlas_memory.server.atlas_daemon import AtlasDaemon
from atlas_memory.server.client import AtlasDaemonClient


def test_f1_ttt_online_adaptation_on_mental_transition():
    """F1: Verifies record_mental_transition executes TTT online adaptation."""
    engine = HybridMemoryEngine.create_default(
        db_path=":memory:",
        qdrant_location=":memory:",
        kuzu_path=":memory:",
    )
    assert engine.ttt is not None
    initial_steps = getattr(engine.ttt, "step_count", 0)
    assert initial_steps == 0

    action = ActionPlan(name="test_observation_step", parameters={"focus": "hot_path_activation"})
    transition = engine.record_mental_transition(action, session_id="test_session")

    assert transition is not None
    assert engine.ttt.step_count == 1
    assert engine.ttt.total_energy >= 0.0


@pytest.mark.asyncio
async def test_f1_live_uds_observation_triggers_ttt(tmp_path: Path):
    """F1: End-to-end UDS test verifying commit_observation triggers TTT adaptation."""
    sock_path = tmp_path / "v55_ttt.sock"
    pid_path = tmp_path / "v55_ttt.pid"
    engine = HybridMemoryEngine.create_default(
        db_path=str(tmp_path / "atlas_ttt.db"),
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    daemon = AtlasDaemon(
        socket_path=sock_path,
        pid_path=pid_path,
        engine=engine,
    )
    await daemon.start()

    client = AtlasDaemonClient(socket_path=sock_path)
    try:
        res = await client.call("commit_observation", {
            "subject": "service:mesh",
            "predicate": "health",
            "object": "nominal",
            "confidence": 0.9,
            "session_id": "test_v55_sess",
        })
        assert res["status"] == "ok"
        assert res["committed"] is True

        stats = await client.call("get_stats")
        assert stats["status"] == "ok"
        assert stats.get("ttt_steps", 0) >= 1
        assert "ttt_energy" in stats
    finally:
        await client.close()
        await daemon.stop()


@pytest.mark.asyncio
async def test_f2_feedback_loop_boosts_confidence(tmp_path: Path):
    """F2: Verifies task success feedback boosts record confidence in KV and logs in audit log."""
    sock_path = tmp_path / "v55_fb.sock"
    pid_path = tmp_path / "v55_fb.pid"
    engine = HybridMemoryEngine.create_default(
        db_path=str(tmp_path / "atlas_fb.db"),
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    daemon = AtlasDaemon(
        socket_path=sock_path,
        pid_path=pid_path,
        engine=engine,
    )
    await daemon.start()

    client = AtlasDaemonClient(socket_path=sock_path)
    try:
        # 1. Commit initial observation with confidence 0.80
        res1 = await client.call("commit_observation", {
            "subject": "developer:preference",
            "predicate": "theme",
            "object": "solarized_dark",
            "confidence": 0.80,
            "session_id": "test_sess",
        })
        assert res1["status"] == "ok"
        assert res1["confidence"] == 0.80

        # 2. Re-commit observation with task_success=True (should boost confidence by +0.05)
        res2 = await client.call("commit_observation", {
            "subject": "developer:preference",
            "predicate": "theme",
            "task_success": True,
            "session_id": "test_sess",
        })
        assert res2["status"] == "ok"
        assert res2["confidence"] == pytest.approx(0.85, abs=1e-3)

        # 3. Dedicated task_feedback RPC call with success
        res3 = await client.call("task_feedback", {
            "key": "developer:preference:theme",
            "task_success": True,
            "delta": 0.05,
        })
        assert res3["status"] == "ok"
        assert res3["confidence"] == pytest.approx(0.90, abs=1e-3)

        # 4. Verify in KV store directly
        state = engine.kv.get_sync("developer:preference:theme")
        assert state is not None
        assert state["confidence"] == pytest.approx(0.90, abs=1e-3)

        # 5. Verify audit log integrity
        audit_res = await client.call("verify_audit_log", {"deep": True, "heal": False})
        assert audit_res["valid"] is True
    finally:
        await client.close()
        await daemon.stop()


@pytest.mark.asyncio
async def test_f3_daemon_startup_auto_heals_broken_chain(tmp_path: Path):
    """F3: Verifies daemon auto-heals corrupted SHA-256 audit log on startup."""
    db_file = tmp_path / "atlas_corrupted.db"
    kv = VerifiedKVStore(db_path=str(db_file))
    kv.set_sync("config:timeout", "30s")
    kv.set_sync("config:retries", "3")
    kv.set_sync("config:backoff", "exponential")

    # Verify initial valid state
    is_valid, _ = kv.verify_audit_log_sync(deep=False)
    assert is_valid is True

    # Tamper with row 2 entry_hash directly in SQLite
    conn = sqlite3.connect(str(db_file))
    conn.execute("UPDATE state_audit_log SET entry_hash = 'tampered_malformed_hash' WHERE seq = 2")
    conn.commit()
    conn.close()

    # Verify chain is broken
    is_valid_broken, broken_seq = kv.verify_audit_log_sync(deep=False)
    assert is_valid_broken is False
    assert broken_seq == 2

    # Launch daemon on top of this corrupted DB
    sock_path = tmp_path / "v55_heal.sock"
    pid_path = tmp_path / "v55_heal.pid"
    engine = HybridMemoryEngine.create_default(
        db_path=str(db_file),
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    daemon = AtlasDaemon(
        socket_path=sock_path,
        pid_path=pid_path,
        engine=engine,
    )

    # Daemon start must trigger D2 auto-heal
    await daemon.start()

    client = AtlasDaemonClient(socket_path=sock_path)
    try:
        # After daemon startup, audit log must be healed and valid
        audit_res = await client.call("verify_audit_log", {"deep": False, "heal": False})
        assert audit_res["valid"] is True

        is_valid_after, _ = engine.kv.verify_audit_log_sync(deep=False)
        assert is_valid_after is True
    finally:
        await client.close()
        await daemon.stop()


def test_f4_adr_002_module_status_characterization():
    """F4: Verifies clean import and baseline instantiation of ADR-002 offline/dormant modules."""
    from atlas_memory.ingest.mnemosyne_ingest import MnemosyneIngestEngine
    from atlas_memory.quantization.matryoshka import MatryoshkaEmbedding
    from atlas_memory.sync.epistemic_bft import BFTLWWSet, EpistemicReputation
    from atlas_memory.sync.threshold_signer import ThresholdSigner

    # 1. Epistemic BFT & Threshold Signer
    signer = ThresholdSigner(node_id="peer_alpha", signing_key=b"12345678901234567890123456789012")
    sig = signer.sign("test_state_payload")
    assert signer.verify(sig, "test_state_payload") is True

    rep = EpistemicReputation()
    rep.record("peer_beta", validated=True)
    assert rep.score("peer_beta") == 1.0
    assert rep.is_trusted("peer_beta") is True

    bft_set = BFTLWWSet(quorum=1)
    assert bft_set.quorum == 1
    assert bft_set.lookup("item_1") is False

    # 2. Matryoshka Embedding
    mrl = MatryoshkaEmbedding(max_dim=64, dimensions=[16, 32, 64])
    assert mrl.max_dim == 64
    assert mrl.dimensions == [16, 32, 64]

    # 3. Mnemosyne Ingestion Engine
    ingester = MnemosyneIngestEngine()
    assert ingester is not None

    # 4. ADR-002 Document Presence and Section Validation
    adr_path = Path("docs/adr/ADR-002-dormant-and-offline-modules-v55.md")
    if not adr_path.exists():
        pytest.skip("docs/adr/ not in public repo (private milestone docs)")
    content = adr_path.read_text(encoding="utf-8")
    assert "ThresholdSigner" in content
    assert "MatryoshkaEmbedding" in content
    assert "semantic_dedup" in content
    assert "Zero rm bez ADR" in content
