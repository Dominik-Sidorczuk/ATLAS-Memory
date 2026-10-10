"""Unit tests verifying F-032 verify_audit_log non-vacuous audit verification."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.server.atlas_daemon import AtlasDaemon


@pytest.mark.asyncio
async def test_verify_audit_log_with_entries_counts_real_entries(tmp_path: Path) -> None:
    """F-032: Must report real checked_entries >= 5, not vacuous 0."""
    db_file = tmp_path / "test_atlas.db"
    kv = VerifiedKVStore(db_path=str(db_file))

    # Write 5 entries into KV store
    for i in range(5):
        kv.set(f"task2_key_{i}", f"val_{i}")

    daemon = AtlasDaemon.__new__(AtlasDaemon)
    mock_engine = MagicMock()
    mock_engine.kv = kv
    daemon.engine = mock_engine

    result = await daemon._handle_verify_audit_log({"deep": True})

    # Verification must be valid and report all 5 checked entries
    assert result["status"] == "ok"
    assert result["valid"] is True
    assert result["checked_entries"] == 5
    assert result["tampered"] is False


@pytest.mark.asyncio
async def test_verify_audit_log_tampered_entry_detected(tmp_path: Path) -> None:
    """F-032: Tampered entry must yield valid=False, tampered=True and point to the broken seq."""
    db_file = tmp_path / "test_atlas.db"
    kv = VerifiedKVStore(db_path=str(db_file))

    for i in range(5):
        kv.set(f"tamper_key_{i}", f"val_{i}")

    # Directly tamper with seq 3 in state_audit_log
    audit_db_file = kv.audit_db_path if kv.audit_db_path and kv.audit_db_path != ":memory:" else str(db_file)
    with sqlite3.connect(str(audit_db_file)) as conn:
        conn.execute("UPDATE state_audit_log SET value_hash = 'tampered_hash_value' WHERE seq = 3")
        conn.commit()

    daemon = AtlasDaemon.__new__(AtlasDaemon)
    mock_engine = MagicMock()
    mock_engine.kv = kv
    daemon.engine = mock_engine

    result = await daemon._handle_verify_audit_log({"deep": True})

    assert result["valid"] is False
    assert result["tampered"] is True
    assert result["checked_entries"] == 5
    tampered_seq = result.get("tampered_seq") or result.get("broken_seq")
    assert tampered_seq == 3


@pytest.mark.asyncio
async def test_verify_audit_log_empty_chain_not_vacuous_true(tmp_path: Path) -> None:
    """F-032: Empty chain must report empty_chain with valid=False, never vacuous valid=True."""
    db_file = tmp_path / "test_empty.db"
    kv = VerifiedKVStore(db_path=str(db_file))

    daemon = AtlasDaemon.__new__(AtlasDaemon)
    mock_engine = MagicMock()
    mock_engine.kv = kv
    daemon.engine = mock_engine

    result = await daemon._handle_verify_audit_log({"deep": True})

    # Vacuous true is forbidden: reporting valid=True on 0 checked entries is rejected
    assert result["status"] == "empty_chain"
    assert result["valid"] is False
    assert result["checked_entries"] == 0
    assert result["tampered"] is False
