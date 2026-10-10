"""FIX-2 (F-002): Zero Silent Fallbacks in HippoRAG 2 PPR & Kùzu Dual-Node Write Path.

Scenarios:
  1. Monkey-patch `_ppr_engine.run` to raise an exception -> `candidates.gather` raises `HippoRAGPPRUnavailable`
     and increments `metrics.counter("hipporag_ppr_error")` (no silent fallback to 1-hop neighbors).
  2. `kuzu_graph.add_contains` with faulty `conn` -> raises `KuzuWriteError` and increments
     `metrics.counter("kuzu_write_error")` (no silent logger.debug).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas_memory.cognitive.retrieval import candidates
from atlas_memory.cognitive.retrieval.ppr import (
    HippoRAGPPRUnavailable,
    metrics,
    reset_hipporag_telemetry,
)
from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore, KuzuWriteError


@pytest.mark.asyncio
async def test_ppr_error_raises_hipporag_ppr_unavailable_without_silent_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    reset_hipporag_telemetry()
    monkeypatch.setenv("ATLAS_HIPPORAG", "1")

    store = KuzuGraphStore(db_path=str(tmp_path / "kuzu_nosilent"))
    try:
        store.add_relation("fact:1hop_neighbor", "relates_to", "target_node", confidence=0.9)

        class _BrokenPPREngine:
            def run(self, *args, **kwargs):
                raise RuntimeError("Simulated PPR engine crash")

        store._ppr_engine = _BrokenPPREngine()

        with pytest.raises(HippoRAGPPRUnavailable):
            await candidates.gather("fact:1hop_neighbor", graph_store=store, limit=5)

        assert metrics.counter("hipporag_ppr_error").value >= 1
    finally:
        store.close()


def test_kuzu_add_contains_faulty_conn_raises_kuzu_write_error(tmp_path: Path) -> None:
    reset_hipporag_telemetry()
    store = KuzuGraphStore(db_path=str(tmp_path / "kuzu_faulty_conn"))
    try:
        class _FaultyConn:
            def execute(self, *args, **kwargs):
                raise RuntimeError("Simulated Kuzu disk I/O error")

        store.conn = _FaultyConn()
        with pytest.raises(KuzuWriteError):
            store.add_contains("fact:test_passage", "test_entity", weight=1.0, persist_to_kuzu=True)

        assert metrics.counter("kuzu_write_error").value >= 1
    finally:
        store.conn = None
        store.close()
