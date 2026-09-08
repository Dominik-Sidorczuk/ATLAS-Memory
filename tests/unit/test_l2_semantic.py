"""
Unit Tests for Layer 2: Semantic Memory & Dual-Engine Store.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.l2_semantic.qdrant_store import QdrantVectorStore
from atlas_memory.models import MemoryRecord


@pytest.mark.asyncio
async def test_qdrant_vector_store_embedded():
    store = QdrantVectorStore(collection_name="test_collection", location=":memory:", dimension=64)
    r1 = MemoryRecord(subject="auth_service", predicate="uses_protocol", object="oauth2")
    r2 = MemoryRecord(subject="database", predicate="hosted_on", object="postgres_cluster")

    await store.insert(r1)
    await store.insert(r2)
    assert await store.count() == 2

    matches = await store.search("authentication protocol", top_k=2)
    assert len(matches) > 0
    assert matches[0]["record"]["subject"] == "auth_service"


@pytest.mark.asyncio
async def test_kuzu_graph_store_subgraph_and_path():
    graph = KuzuGraphStore(db_path=":memory:")
    await graph.add_record(MemoryRecord(subject="User", predicate="triggers", object="DeployAction"))
    await graph.add_record(MemoryRecord(subject="DeployAction", predicate="updates", object="ProductionService"))
    await graph.add_record(MemoryRecord(subject="ProductionService", predicate="depends_on", object="DatabaseCluster"))

    # Relacje 2-go stopnia
    subgraph = await graph.get_subgraph_relations(["User"], max_depth=2)
    assert "DeployAction" in subgraph["nodes"]
    assert "ProductionService" in subgraph["nodes"]

    # Ścieżka przyczynowo-skutkowa
    path = await graph.find_causal_path("User", "DatabaseCluster")
    assert path is not None
    assert len(path) == 3
    assert path[0]["from"] == "User"
    assert path[-1]["to"] == "DatabaseCluster"

    graph.close()


@pytest.mark.asyncio
async def test_verified_kv_store_acid_and_audit():
    kv = VerifiedKVStore(db_path=":memory:")
    await kv.set_state("active_model", "gpt-4o", confidence=1.0)
    await kv.set_state("active_model", "claude-3-5-sonnet", confidence=0.95, reason="model_switch")

    state = await kv.get_state("active_model")
    assert state is not None
    assert state["value"] == "claude-3-5-sonnet"
    assert state["confidence"] == 0.95

    states = await kv.get_states(["active_model", "non_existent"])
    assert "active_model" in states
    assert "non_existent" not in states

    await kv.close()



import pytest


@pytest.mark.asyncio
async def test_hash_chain_append_and_verify():
    """Test 1: Dołączanie wpisów do SHA-256 hash-chain i weryfikacja integralności."""
    kv = VerifiedKVStore(db_path=":memory:")

    # Dodaj kilka wpisów
    h1 = await kv.append_audit_log({"key": "cfg_01", "value": {"port": 8080}, "timestamp": time.time()})
    assert len(h1) == 64

    h2 = await kv.append_audit_log({"key": "cfg_02", "value": {"host": "0.0.0.0"}, "timestamp": time.time()})
    assert len(h2) == 64
    assert h1 != h2

    # Weryfikacja łańcucha
    is_valid, broken_seq = await kv.verify_chain_integrity()
    assert is_valid is True
    assert broken_seq == 0

    await kv.close()


@pytest.mark.asyncio
async def test_hash_chain_tampering_detection():
    """Test 2: Wykrywanie naruszenia integralności łańcucha (tampering)."""
    kv = VerifiedKVStore(db_path=":memory:")

    await kv.append_audit_log({"key": "var1", "value": "val1"})
    await kv.append_audit_log({"key": "var2", "value": "val2"})
    await kv.append_audit_log({"key": "var3", "value": "val3"})

    is_valid, broken_seq = await kv.verify_chain_integrity()
    assert is_valid is True
    assert broken_seq == 0

    # Symulacja ataku / manipulacji w bazie danych (tampering seq=2)
    assert kv._conn is not None
    with kv._conn:
        kv._conn.execute("UPDATE state_audit_log SET value_hash = 'tampered_hash_1234' WHERE seq = 2")

    # Weryfikacja powinna wykryć błąd na seq=2
    is_valid, broken_seq = await kv.verify_chain_integrity()
    assert is_valid is False
    assert broken_seq == 2

    await kv.close()


@pytest.mark.asyncio
async def test_hash_chain_broken_prev_hash_tampering():
    """Test 3: Wykrywanie naruszenia prev_hash w łańcuchu audytu."""
    kv = VerifiedKVStore(db_path=":memory:")

    await kv.append_audit_log({"key": "user", "value": "alice"})
    await kv.append_audit_log({"key": "role", "value": "admin"})

    is_valid, broken_seq = await kv.verify_chain_integrity()
    assert is_valid is True

    # Manipulacja prev_hash na seq=2
    assert kv._conn is not None
    with kv._conn:
        kv._conn.execute("UPDATE state_audit_log SET prev_hash = '0000000000000000000000000000000000000000000000000000000000000000' WHERE seq = 2")

    is_valid, broken_seq = await kv.verify_chain_integrity()
    assert is_valid is False
    assert broken_seq == 2

    await kv.close()


def test_verified_kv_store_raw_string_and_f6_resilience():
    """Test Finding F6: raw string values and malformed metadata in _row_to_state_dict & get_all_sync."""
    kv = VerifiedKVStore(db_path=":memory:")
    assert kv._conn is not None

    # Wstawienie surowych, niezakodowanych w JSON stringów bezpośrednio do SQLite
    with kv._conn:
        kv._conn.execute("""
            INSERT INTO state_variables (key, value, confidence, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, ("tamper:test", "ORIGINAL", 1.0, 123456.0, "not_valid_json_metadata"))
        kv._conn.execute("""
            INSERT INTO state_variables (key, value, confidence, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, ("json:test", '{"greeting": "hello"}', 0.9, 123457.0, '{"tag": "test"}'))

    # Sprawdzenie get_sync
    res_tamper = kv.get_sync("tamper:test")
    assert res_tamper is not None
    assert res_tamper["value"] == "ORIGINAL"
    assert res_tamper["metadata"] == {}

    res_json = kv.get_sync("json:test")
    assert res_json is not None
    assert res_json["value"] == {"greeting": "hello"}
    assert res_json["metadata"] == {"tag": "test"}

    # Sprawdzenie get_all_sync oraz aliasu get_all
    all_sync = kv.get_all_sync()
    assert len(all_sync) == 2
    assert all_sync["tamper:test"]["value"] == "ORIGINAL"
    assert all_sync["json:test"]["value"] == {"greeting": "hello"}

    all_alias = kv.get_all()
    assert len(all_alias) == 2
    assert all_alias == all_sync


def test_verified_kv_store_audit_log_sync():
    """Test Finding F1: synchronous verify_audit_log_sync and verify_chain_integrity_sync."""
    kv = VerifiedKVStore(db_path=":memory:")

    # Używamy set_sync do dodania wpisów
    kv.set_sync("k1", "v1")
    kv.set_sync("k2", {"nested": "v2"})

    # Weryfikacja synchroniczna bez pętli zdarzeń
    is_valid, broken_seq = kv.verify_audit_log_sync()
    assert is_valid is True
    assert broken_seq == 0

    is_valid_chain, broken_chain_seq = kv.verify_chain_integrity_sync()
    assert is_valid_chain is True
    assert broken_chain_seq == 0

    # Test deep verification
    is_deep_valid, _ = kv.verify_audit_log_sync(deep=True)
    assert is_deep_valid is True

    # Tampering test w state_audit_log
    assert kv._conn is not None
    with kv._conn:
        kv._conn.execute("UPDATE state_audit_log SET value_hash = 'corrupt' WHERE seq = 1")

    is_valid_after_tamper, broken_seq_after = kv.verify_audit_log_sync()
    assert is_valid_after_tamper is False
    assert broken_seq_after == 1


def test_qdrant_ngram_hashing_deterministic():
    """Weryfikacja deterministycznego rzutowania n-gramów w QdrantVectorStore."""
    store = QdrantVectorStore(dimension=32)
    vec1 = store.encoder.encode("PostgreSQL configuration database port 5432")
    vec2 = store.encoder.encode("PostgreSQL configuration database port 5432")

    assert np.allclose(vec1, vec2)
    assert len(vec1) == 32


def test_kuzu_graph_store_close_cleans_resources():
    """Weryfikacja że KuzuGraphStore.close() czyści uchwyty i usuwa katalog tymczasowy."""
    store = KuzuGraphStore(db_path=":memory:")
    assert store.nx_graph is not None
    store.close()
    assert store.conn is None
    assert store.db is None
    assert store._temp_dir is None


def test_get_recent_sync_descending_order(tmp_path: Path):
    """Verifies get_recent_sync fetches freshest records and returns them in ascending chronological order."""
    db_path = str(tmp_path / "test_kv.db")
    kv = VerifiedKVStore(db_path=db_path)

    base_time = time.time() - 300.0  # 5 minutes ago
    conn = kv._ensure_conn()
    with conn:
        for i in range(30):
            conn.execute(
                "INSERT OR REPLACE INTO state_variables (key, value, confidence, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
                (f"state:item_{i:02d}", json.dumps({"index": i, "data": f"payload_{i}"}), 0.9, base_time + (i * 2.0), "{}"),
            )

    recent = kv.get_recent_sync(max_age_seconds=600.0, limit=25)
    assert len(recent) == 25

    returned_keys = [item["key"] for item in recent]
    assert returned_keys[0] == "state:item_05"
    assert returned_keys[-1] == "state:item_29"

    for i in range(len(recent) - 1):
        assert recent[i]["timestamp"] <= recent[i + 1]["timestamp"]


@pytest.mark.asyncio
async def test_salience_gc_pruning_and_user_rule_protection():
    """Weryfikuje usuwanie przestarzałych i niskiej pewności faktów oraz ochronę user_explicit."""
    engine = HybridMemoryEngine.create_default(db_path=":memory:", qdrant_location=":memory:")

    now = time.time()
    await engine.kv.set_state(
        key="temp:weather_guess",
        value="rainy tomorrow",
        confidence=0.20,
        metadata={"source_type": "agent_inference", "importance_score": 0.2},
    )

    await engine.kv.set_state(
        key="temp:chatter",
        value="discussed lunch options",
        confidence=0.70,
        metadata={"source_type": "agent_inference", "importance_score": 0.01},
    )
    async with engine.kv._lock:
        conn = engine.kv._ensure_conn()
        with conn:
            conn.execute("UPDATE state_variables SET timestamp = ? WHERE key = ?", (now - 864000.0, "temp:chatter"))

    await engine.kv.set_state(
        key="rule:preferred_language",
        value="Polish",
        confidence=1.0,
        metadata={"source_type": "user_explicit", "importance_score": 0.95},
    )

    await engine.kv.set_state(
        key="user:full_name",
        value="Dominik",
        confidence=0.95,
        metadata={"source_type": "user_explicit", "importance_score": 0.90},
    )

    all_init = await engine.kv.get_all_states()
    assert len(all_init) == 4

    stats = await engine.auditor.run_sleep_cycle_consolidation()

    assert stats.pruned_stale_facts >= 2
    assert stats.decayed_pruned_records >= 2

    assert await engine.kv.get_state("temp:weather_guess") is None
    assert await engine.kv.get_state("temp:chatter") is None
    rule_state = await engine.kv.get_state("rule:preferred_language")
    assert rule_state is not None
    assert rule_state["value"] == "Polish"
    user_state = await engine.kv.get_state("user:full_name")
    assert user_state is not None
    assert user_state["value"] == "Dominik"


@pytest.mark.asyncio
async def test_h14_sleep_consolidation_resilient_to_malformed_confidence():
    """
    H14 (F8): Verifies sleep cycle consolidation does not crash (RPC -32602) when encountering
    poisoned/corrupted database records with invalid confidence or metadata.
    """
    engine = HybridMemoryEngine.create_default(db_path=":memory:", qdrant_location=":memory:")

    await engine.kv.set_state(
        key="config:theme",
        value="dark",
        confidence=1.0,
        metadata={"source_type": "user_explicit", "importance_score": 0.9},
    )

    async with engine.kv._lock:
        conn = engine.kv._ensure_conn()
        with conn:
            conn.execute(
                "INSERT INTO state_variables (key, value, timestamp, confidence, metadata) VALUES (?, ?, ?, ?, ?)",
                ("poison:corrupt_row", "bad_value", time.time(), "corrupt_confidence_val", "corrupt_metadata"),
            )

    stats = await engine.auditor.run_sleep_cycle_consolidation()
    assert stats.records_analyzed >= 2

    theme_state = await engine.kv.get_state("config:theme")
    assert theme_state is not None
    assert theme_state["value"] == "dark"


def test_h21_qdrant_stale_lock_recovery(tmp_path: Path):
    """
    H21: Verifies auto-recovery from stale Qdrant .lock file left behind by SIGKILL/crash.
    """
    qdrant_dir = tmp_path / "qdrant_store"
    qdrant_dir.mkdir(parents=True, exist_ok=True)

    stale_lock = qdrant_dir / ".lock"
    stale_lock.write_text("99999")
    assert stale_lock.exists()

    store = QdrantVectorStore(location=str(qdrant_dir))
    assert store.client is not None

    rec = MemoryRecord(
        subject="system",
        predicate="status",
        object="recovered",
        confidence=1.0,
    )
    doc_id = store.insert_sync(rec)
    assert doc_id is not None
    if hasattr(store, "close"):
        store.close()


def test_t8_t10_heal_audit_log_chain_and_clamp_confidence(tmp_path: Path):
    """
    T8 & T10: Verifies that heal_audit_log_chain heals broken hash chains,
    clamps out-of-bounds confidence values, and purge_stale_probe_keys removes test keys.
    """
    db_file = tmp_path / "test_heal_chain.db"
    kv = VerifiedKVStore(db_path=str(db_file))

    kv.set_sync("user:pref", "dark_mode", confidence=1.0)
    kv.set_sync("user:lang", "pl", confidence=0.9)
    kv.set_sync("huge:value", "mock_huge_payload", confidence=1.0)
    kv.set_sync("test:adv:probe", "probe_data", confidence=1.0)

    conn = kv._ensure_conn()
    with conn:
        conn.execute("""
            UPDATE state_audit_log
            SET prev_hash = 'corrupted_hash_value', confidence = 2.75
            WHERE seq = 2
        """)

    is_valid, err_seq = kv.verify_audit_log_sync()
    assert not is_valid
    assert err_seq == 2

    deleted_probes = kv.purge_stale_probe_keys()
    assert deleted_probes >= 2
    assert kv.get_sync("huge:value") is None
    assert kv.get_sync("test:adv:probe") is None
    assert kv.get_sync("user:pref") is not None

    repaired_count, heal_ok = kv.heal_audit_log_chain()
    assert heal_ok
    assert repaired_count >= 2

    is_valid_after, err_seq_after = kv.verify_audit_log_sync()
    assert is_valid_after
    assert err_seq_after == 0

    cursor = conn.cursor()
    cursor.execute("SELECT confidence FROM state_audit_log WHERE seq = 2")
    row = cursor.fetchone()
    assert row is not None
    assert float(row[0]) <= 1.0

