"""
Canonical Ingest Pipeline Tests (consolidated).

Validates:
1. MEMORY.md path resolution & HERMES_HOME fallback chain
2. EpistemicWritePath choke point routing for all ingested records
3. 100% Idempotency of re-ingest (0 new records, 0 new edges, 0 re-indexed vectors)
4. Zero-orphaned vectors contract (all vector points link to valid KV entries)
5. DomainRelationExtractor decoupling
"""
from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.ingest.domain_rules import DomainRelationExtractor
from atlas_memory.ingest.mnemosyne_ingest import MnemosyneIngestEngine


@pytest.fixture
def sample_memory_file(tmp_path: Path) -> Path:
    mem_file = tmp_path / "MEMORY.md"
    content = """
Obsidian: natywne mcp__obsidian__vault_* jeśli w katalogu; port 27124.
§
ZASADA: ZAWSZE czytaj skill (skill_view) PRZED użyciem narzędzia.
§
Crawl4AI: silnik researchu; localhost:11235.
""".strip()
    mem_file.write_text(content, encoding="utf-8")
    return mem_file


@pytest.fixture
def test_engine(tmp_path: Path) -> HybridMemoryEngine:
    return HybridMemoryEngine.create_default(
        db_path=str(tmp_path / "test_atlas.db"),
        qdrant_location=":memory:",
        kuzu_path=str(tmp_path / "test_kuzu"),
    )


def test_ingest_memory_md_path_resolution_and_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Weryfikacja fallback chainu ścieżki MEMORY.md (HERMES_HOME oraz legacy ~/.hermes)."""
    import atlas_memory.ingest.mnemosyne_ingest as mod
    fresh = importlib.reload(mod)

    # 1. HERMES_HOME env
    custom_hermes = tmp_path / "custom_hermes"
    monkeypatch.setenv("HERMES_HOME", str(custom_hermes))
    mem_path = custom_hermes / "memories" / "MEMORY.md"
    mem_path.parent.mkdir(parents=True, exist_ok=True)
    mem_path.write_text("Qdrant: kolekcja episodic na porcie 6333.\n", encoding="utf-8")

    eng1 = fresh.MnemosyneIngestEngine()
    assert eng1.is_available is True
    recs1 = eng1.extract_memory_md_records()
    assert len(recs1) == 1

    # 2. Legacy ~/.hermes/memory.md
    fake_home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    legacy_mem = fake_home / ".hermes" / "memory.md"
    legacy_mem.parent.mkdir(parents=True, exist_ok=True)
    legacy_mem.write_text("Kuzu: graph store.\n§\nOPNsense: firewall.\n", encoding="utf-8")

    fresh2 = importlib.reload(mod)
    eng2 = fresh2.MnemosyneIngestEngine()
    assert eng2.is_available is True
    recs2 = eng2.extract_memory_md_records()
    assert len(recs2) == 2


def test_sync_memory_md_routes_all_writes_through_write_path(
    test_engine: HybridMemoryEngine, sample_memory_file: Path
) -> None:
    """Wszystkie rekordy z MEMORY.md muszą przechodzić przez EpistemicWritePath."""
    ingester = MnemosyneIngestEngine(memory_md_path=sample_memory_file)

    original_write = test_engine.write_path.write
    mock_write = MagicMock(side_effect=original_write)
    test_engine.write_path.write = mock_write

    res = ingester.sync_memory_md_into_engine(test_engine, path=sample_memory_file)
    assert res["status"] == "ok"
    assert res["ingested"] == 3
    assert mock_write.call_count == 3


def test_reingest_idempotency_zero_new_records_and_edges(tmp_path: Path, sample_memory_file: Path):
    """Powtórny ingest tego samego pliku to 100% idempotentne NO-OP."""
    engine = HybridMemoryEngine.create_default(
        db_path=str(tmp_path / "atlas_test.db"),
        qdrant_location=":memory:",
        kuzu_path=str(tmp_path / "kuzu_test"),
    )
    ingester = MnemosyneIngestEngine(memory_md_path=sample_memory_file)

    # 1. Pierwszy ingest
    res1 = ingester.sync_memory_md_into_engine(engine, path=sample_memory_file)
    assert res1["status"] == "ok"
    assert res1["ingested"] == 3
    assert res1["vectors_indexed"] == 3

    # 2. Drugi ingest (identyczna treść)
    res2 = ingester.sync_memory_md_into_engine(engine, path=sample_memory_file)
    assert res2["status"] == "ok"
    assert res2["ingested"] == 0
    assert res2["updated"] == 0
    assert res2["vectors_indexed"] == 0
    assert res2["graph_edges_added"] == 0


def test_zero_orphaned_vectors_contract(
    test_engine: HybridMemoryEngine, sample_memory_file: Path
) -> None:
    """Zero osieroconych wektorów: każdy wektor odpowiada obecnemu kluczowi w KV."""
    ingester = MnemosyneIngestEngine(memory_md_path=sample_memory_file)
    res = ingester.sync_memory_md_into_engine(test_engine, path=sample_memory_file)
    assert res["status"] == "ok"

    kv_items = test_engine.kv.get_all_sync()
    assert len(kv_items) == 3

    hits, _ = test_engine.vector_store.client.scroll(
        collection_name=test_engine.vector_store.collection_name,
        with_payload=True,
        limit=100,
    )
    assert len(hits) == 3
    for hit in hits:
        upsert_key = hit.payload.get("upsert_key") or hit.payload.get("subject")
        assert upsert_key in kv_items, f"Vector point {hit.id} has no matching KV record for key {upsert_key}"


def test_domain_relation_extractor_decoupling():
    """DomainRelationExtractor ekstrahuje predykaty domenowe bez zanieczyszczania rdzenia."""
    custom_patterns = [
        (r"(?:listens on|nasłuchuje na)\s+([A-Za-z0-9_.:]+)", "{subject}", "listens_on"),
    ]
    extractor = DomainRelationExtractor(dep_patterns=custom_patterns)
    edges = []

    def mock_add_edge(u: str, p: str, v: str, c: float = 1.0) -> None:
        edges.append((u, p, v, c))

    extractor.extract_and_apply(
        subject="HermesDaemon",
        content="HermesDaemon nasłuchuje na 127.0.0.1:8080 dla żądań klienta.",
        add_edge_fn=mock_add_edge,
        stop_words=set(),
    )
    assert ("HermesDaemon", "listens_on", "127.0.0.1:8080", 1.0) in edges
