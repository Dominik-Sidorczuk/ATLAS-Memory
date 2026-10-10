"""Regression: filtr sesji w `QdrantVectorStore.search_hybrid_async` (ERR-4).

Pierwsza wersja filtra używała wyłącznie `IsNullCondition`, które w Qdrant NIE
dopasowuje brakującego pola — trwałe punkty bez `metadata.session_id` (sesja
wynikająca z prefiksu klucza: `hermes:`, `fact:`) znikały z wyników dla każdej
nazwanej sesji. Test działa na prawdziwym trybie lokalnym Qdrant (bez mocków).
"""
from __future__ import annotations

import asyncio

import pytest

from atlas_memory.core.models import MemoryRecord
from atlas_memory.l2_semantic.qdrant_store import HAS_QDRANT, QdrantVectorStore

pytestmark = pytest.mark.skipif(not HAS_QDRANT, reason="qdrant-client not installed")


def _insert(store: QdrantVectorStore, key: str, text: str, session: str | None) -> None:
    meta = {"session_id": session} if session else {}
    rec = MemoryRecord(subject=key, predicate="stated", object=text, metadata=meta)
    store.insert_sync(rec, raw_text=f"{key} {text}", upsert_key=f"kv:{key}")


def _keys(rows: list) -> set[str]:
    return {str(r.get("kv_key") or "") for r in rows}


def test_named_session_sees_own_global_and_unscoped_but_not_foreign():
    store = QdrantVectorStore(location=":memory:", dimension=64)
    _insert(store, "session_a_note", "obsidian vault config alpha", "A")
    _insert(store, "session_b_note", "obsidian vault config bravo", "B")
    _insert(store, "hermes:memory_md:obsidian", "obsidian vault config durable", None)
    _insert(store, "rule:obsidian", "obsidian vault config global", "global")

    rows = asyncio.run(store.search_hybrid_async("obsidian vault config", limit=10, session_id="A"))
    keys = _keys(rows)

    assert "session_a_note" in keys
    assert "hermes:memory_md:obsidian" in keys, "durable point without metadata.session_id must survive"
    assert "rule:obsidian" in keys
    assert "session_b_note" not in keys, "foreign private session must be filtered at engine level"


@pytest.mark.parametrize("sid", ["", "default", "hermes_default", "global"])
def test_default_class_sessions_do_not_engage_engine_filter(sid: str):
    assert QdrantVectorStore._build_session_filter(sid) is None
