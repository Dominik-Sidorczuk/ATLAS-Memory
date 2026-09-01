"""
Unit Tests for Layer 2: Semantic Memory & Dual-Engine Store.
"""
from __future__ import annotations

import pytest

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


import time

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
