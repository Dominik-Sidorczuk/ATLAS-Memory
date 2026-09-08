"""
Contract and regression tests for V60:
1. T19: Delete Guard — Protected prefix rejection (hermes:, fact:, prompt-, mnemosyne:) in daemon and provider.
2. T19: Delete SHA-256 Hash-Chain Integrity — Deleting an allowed key records an audit entry with reason="delete".
3. T05: Confidence Inversion Warning — Same-key same-value write with lower confidence triggers confidence_inversion.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from atlas_memory.cognitive.conflict_arbiter import ConflictArbiter
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.hermes.atlas_provider import AtlasMemoryProvider
from atlas_memory.security.delete_guard import DELETE_PROTECTED_PREFIXES, is_delete_protected
from atlas_memory.server.atlas_daemon import AtlasDaemon


def test_a3_identical_protected_prefixes_tuple():
    """A3: Verifies that AtlasDaemon and delete_guard share the identical tuple."""
    assert AtlasDaemon._DELETE_PROTECTED_PREFIXES == DELETE_PROTECTED_PREFIXES
    assert isinstance(DELETE_PROTECTED_PREFIXES, tuple)
    assert ("hermes:", "fact:", "prompt-", "mnemosyne:") == DELETE_PROTECTED_PREFIXES


@pytest.mark.parametrize("prefix", list(DELETE_PROTECTED_PREFIXES))
@pytest.mark.asyncio
async def test_a3_delete_guard_rejection_per_prefix(tmp_path: Path, prefix: str):
    """A3: Parameterized test verifying rejection for each protected prefix in daemon and provider."""
    pkey = f"{prefix}test_key_{prefix.replace(':', '_').replace('-', '_')}"

    # 1. Direct function check
    is_prot, matched = is_delete_protected(pkey)
    assert is_prot is True
    assert matched == prefix

    # 2. Provider in-process check
    provider = AtlasMemoryProvider()
    res_prov = provider._handle_in_process_tool_call("atlas_delete", {"key": pkey})
    assert res_prov["status"] == "rejected"
    assert res_prov["reason"] == "protected_key"

    # 3. Daemon check
    db_path = str(tmp_path / f"test_a3_{prefix.replace(':', '_').replace('-', '_')}.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    daemon = AtlasDaemon(engine=engine)
    res_daemon = await daemon._handle_delete({"key": pkey})
    assert res_daemon["status"] == "rejected"
    assert res_daemon["reason"] == "protected_key"


@pytest.mark.asyncio
async def test_v60_delete_guard_rejects_protected_prefixes(tmp_path: Path):
    """
    T19: Verifies that AtlasDaemon._handle_delete rejects deletion of keys
    with protected prefixes: 'hermes:', 'fact:', 'prompt-', 'mnemosyne:'.
    """
    db_path = str(tmp_path / "test_atlas_delete_guard.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    daemon = AtlasDaemon(engine=engine)

    protected_keys = [
        "hermes:memory_md:ATLAS (LOOP/Memory)",
        "hermes:system_rule:exclusivity",
        "fact:user:birthday",
        "prompt-delegated_agent:rules",
        "mnemosyne:working_memory:ctx",
    ]

    for pkey in protected_keys:
        # Seed the key into KV
        engine.kv.set_sync(pkey, "important_value", confidence=1.0)
        assert engine.kv.get_sync(pkey) is not None

        # Attempt delete via daemon RPC
        res = await daemon._handle_delete({"key": pkey})
        assert res["status"] == "rejected", f"Expected 'rejected' for protected key {pkey}, got {res}"
        assert res["deleted"] is False
        assert res["reason"] == "protected_key"
        assert "protected" in res["message"].lower()

        # Verify key still exists in KV
        assert engine.kv.get_sync(pkey) is not None, f"Protected key {pkey} was deleted from KV!"


def test_v60_provider_in_process_delete_guard():
    """
    T19: Verifies that AtlasMemoryProvider._handle_in_process_tool_call
    also rejects protected prefixes in offline/fallback mode.
    """
    provider = AtlasMemoryProvider()
    protected_keys = [
        "hermes:memory_md:foo",
        "fact:core:rule",
        "prompt-system:instruction",
        "mnemosyne:triple:x",
    ]
    for pkey in protected_keys:
        res = provider._handle_in_process_tool_call("atlas_delete", {"key": pkey})
        assert res["status"] == "rejected"
        assert res["deleted"] is False
        assert res["reason"] == "protected_key"


@pytest.mark.asyncio
async def test_v60_delete_allowed_key_records_audit_chain(tmp_path: Path):
    """
    T19: Verifies that deleting an unprotected key succeeds and records
    an audit log entry in the SHA-256 hash-chain with reason='delete'.
    """
    db_path = str(tmp_path / "test_atlas_delete_audit.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    daemon = AtlasDaemon(engine=engine)

    allowed_key = "probe:test_latency:ms"
    engine.kv.set_sync(allowed_key, "42", confidence=0.9)

    # Verify audit entries count before delete
    valid_before, _ = engine.kv.verify_audit_log_sync()
    assert valid_before is True

    # Delete via daemon
    res = await daemon._handle_delete({"key": allowed_key})
    assert res["status"] == "ok"
    assert res["deleted"] is True
    assert res["found"] is True

    # Verify key is gone
    assert engine.kv.get_sync(allowed_key) is None

    # Verify audit chain has recorded delete
    valid_after, broken_seq = engine.kv.verify_audit_log_sync()
    assert valid_after is True, f"Hash-chain broken after delete at seq {broken_seq}"

    # Verify last audit entry is the delete event
    conn = engine.kv._ensure_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT key, value_hash, reason FROM state_audit_log ORDER BY seq DESC LIMIT 1")
    row = cursor.fetchone()
    assert row["key"] == allowed_key
    assert row["reason"] == "delete"


def test_v60_t05_confidence_inversion_on_same_value():
    """
    T05: Verifies that when a write arrives for an existing key with the SAME value
    but LOWER confidence (e.g. 0.9 -> 0.6), ConflictArbiter detects confidence_inversion
    and issues an epistemic_conflict warning rather than silently accepting.
    """
    arbiter = ConflictArbiter()
    key = "service:port"
    value = "8080"

    existing_items = {
        key: {
            "value": value,
            "confidence": 0.9,
            "metadata": {"source_type": "user_explicit"},
        }
    }

    # Write arrives with same value but lower confidence (0.6 < 0.9)
    res = arbiter.arbitrate(
        key=key,
        val=value,
        confidence=0.6,
        metadata={"source_type": "agent_inference"},
        existing_items=existing_items,
    )

    assert res.is_conflict is True, "Expected conflict to be True on confidence inversion"
    assert res.reason == "confidence_inversion"
    assert res.warning == "epistemic_conflict"
    assert res.conflicting_key == key
