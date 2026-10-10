"""V151.1 F-031 — split-brain tombstone: widok materializowany + bramka read-path KV.

Diagnoza (live): DeclarativeConsolidator._tombstone_discourse_garbage pisał
is_superseded TYLKO do KV; punkty Qdrant i węzły Kùzu nie dziedziczyły flagi,
więc atlas_recall serwował superseded fakty (source=graph / source=vector).

Kontrakt V151.1 (FILAR II — KV = źródło prawdy, Qdrant/Kùzu = projekcje):
  (a) bramka read-path: rekord źródła=graph trafiający do poola MUSI być
      zweryfikowany przeciwko KV — baza klucza (subject bez transportowych
      sufiksów/segmentów) z is_superseded=True → odrzuć (NIE czarna lista).
  (b) spread ekwiwalentny: po tombstonie KV konsolidator KAŻDE muśnięcie
      widoku: punkt Qdrant usunięty LUB oznaczony; węzeł Entity w Kùzu
      oznaczony is_superseded (właściwość, nie DELETE — audyt + krawędzie).
  (c) regresja: istniejące testy v151_* muszą przechodzić (osobny plik).
"""
from __future__ import annotations

import time

import pytest

from atlas_memory.cognitive.retrieval.candidates import (
    _derive_kv_probe_keys,
    _graph_channel_kv_truth_gate,
)
from atlas_memory.core.models import MemoryRecord
from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.l2_semantic.qdrant_store import QdrantVectorStore
from atlas_memory.l3_procedural.declarative_consolidator import DeclarativeConsolidator

# ── (a) jednostkowy: bramka read-path odrzuca superseded rekord grafowy ───────

def test_derive_kv_probe_keys_canonicalization():
    """Derwacja bazy klucza: bez prefiksów transportowych, bez sufiksów segmentu."""
    variants = {
        "jak_obecnie:state_value",
        "kv:jak_obecnie:state",
        "fact:jak_obecnie:state",
    }
    for v in variants:
        probes = set(_derive_kv_probe_keys(v))
        assert "jak_obecnie:state" in probes, v
        assert "fact:jak_obecnie:state" in probes, v


def test_graph_channel_gate_drops_superseded():
    """Rekord source=graph z tombstonem w KV NIE trafia do poola; legit zostaje."""

    class FakeKV:
        def __init__(self, states):
            self._states = states

        async def get_states(self, keys):
            return {k: self._states[k] for k in keys if k in self._states}

    pool = {
        "fact:jak_obecnie:state": {
            "subject": "jak_obecnie", "predicate": "state_value", "object": "x",
            "confidence": 0.9, "timestamp": time.time(), "source": "graph",
            "metadata": {"origin": "graph_channel"},
        },
        "fact:ip_serweranasa:state": {
            "subject": "ip_serweranasa", "predicate": "state_value", "object": "10.0.0.9",
            "confidence": 0.9, "timestamp": time.time(), "source": "graph",
            "metadata": {"origin": "graph_channel"},
        },
    }

    async def _gate(states):
        kv = FakeKV(states)
        out = await _graph_channel_kv_truth_gate(pool, kv)
        return out

    # Tylko 'jak_obecnie' ma tombstone w KV → 'ip_serweranasa' (bez flag) zostaje
    after_sup = pytest_asyncio_run(_gate({"fact:jak_obecnie:state": {"value": "x", "metadata": {"is_superseded": True}}}))
    assert "fact:jak_obecnie:state" not in after_sup
    assert "fact:ip_serweranasa:state" in after_sup

    # Pusty KV (brak flag) → gate nic nie wycina
    after_clean = pytest_asyncio_run(_gate({}))
    assert set(after_clean.keys()) == set(pool.keys())


def pytest_asyncio_run(coro):
    """Pomocnicze: bez pluginu asyncio w tym pliku (czysto jednostkowe)."""
    import asyncio
    return asyncio.run(coro)


# ── (b) kontrakt: po tombstonie punkt Qdrant usunięty LUB oznaczony + węzeł ───

@pytest.mark.asyncio
async def test_tombstone_spread_marks_views(tmp_path):
    """Po _tombstone_discourse_garbage: Qdrant punkt znika; Kùzu Entity ma is_superseded.

    Kontrakt V151.1: tombstone pisany w KV przez konsolidator PROPAGUJE się
    ekwiwalentnie na widoki (Qdrant delete + Kùzu is_superseded właściwość).
    Legit (non-garbage) state NIE jest ruszane.
    """
    kv = VerifiedKVStore(db_path=str(tmp_path / "kv.db"))
    # vector_dim=64 → używany _StubDenseEncoder (brak ONNX w testach)
    vec = QdrantVectorStore(path=str(tmp_path / "vec"), collection_name="atlas_v151_1", dimension=64)
    graph = KuzuGraphStore(db_path=str(tmp_path / "g"))
    try:
        # Węzeł Entity + punkt Qdrant pod kanonicznym kluczem (jak w produkcji)
        graph.add_entity("jak_obecnie", entity_type="Concept")
        garbage_rec = MemoryRecord(
            subject="jak_obecnie",
            predicate="state_value",
            object="x",
            confidence=0.5,
            timestamp=time.time(),
            vector=[0.1] * 64,
            metadata={"subject": "jak_obecnie", "predicate": "state_value"},
        )
        vec.insert_sync(garbage_rec, raw_text="jak_obecnie state_value x", upsert_key="fact:jak_obecnie:state")

        # Ustaw stan w KV BEZ flag — konsolidator sam wykryje barierę dyskursywną
        # dla entity_root='jak_obecnie' (head-token 'jak' ∈ QUERY_META_STOPWORDS).
        await kv.set_state(
            "fact:jak_obecnie:state",
            value="x",
            confidence=0.5,
            metadata={"subject": "jak_obecnie", "predicate": "state_value"},
        )

        cons = DeclarativeConsolidator(kv_store=kv, vector_store=vec, graph_store=graph)
        await cons.consolidate()

        # (1) punkt Qdrant usunięty pod kanonicznym upsert_key:
        got = vec.get_by_key("fact:jak_obecnie:state")
        assert got is None, f"vector leak: {got}"

        # (2) węzeł Kùzu oznaczony właściwością (NIE kasowany — audyt/krawędzie)
        assert graph.is_entity_superseded("jak_obecnie") is True

        # (3) KV źródło prawdy ma flagę
        state = await kv.get_state("fact:jak_obecnie:state")
        assert state is not None
        assert (state.get("metadata") or {}).get("is_superseded") is True

        # (4) regresja: konsolidacja nie zjada legitnych węzłów.
        #     Używamy tematu 'serwer_produkcyjny' — NIE jest śmieciem dyskursywnym
        #     (head-token 'serwer_produkcyjny' nie leci w stopwords/stems).
        graph.add_entity("serwer_produkcyjny", entity_type="Concept")
        await kv.set_state(
            "fact:serwer_produkcyjny:state",
            value="10.0.0.9",
            confidence=0.5,
            metadata={"subject": "serwer_produkcyjny", "predicate": "state_value"},
        )
        await cons.consolidate()
        assert graph.is_entity_superseded("serwer_produkcyjny") is False
    finally:
        try:
            graph.close()
        except Exception:
            pass
