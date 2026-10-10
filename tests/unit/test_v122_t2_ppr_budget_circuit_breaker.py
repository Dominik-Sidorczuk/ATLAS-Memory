"""V122 (T2) — HippoRAG 2 PPR Budget Enforcement and Circuit Breaker.

Contracts tested:
  1. HippoRAG 2 PPR enforces budget <= 20ms and raises HippoRAGPPRUnavailable on timeout.
  2. gather_candidates catches HippoRAGPPRUnavailable and other exceptions without crashing prefetch.
  3. Graceful fallback to exact/stemmed graph channel when PPR fails or times out.
  4. Failure increments hipporag_error counter cleanly without breaking the response.
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from atlas_memory.cognitive.retrieval.candidates import gather_candidates
from atlas_memory.cognitive.retrieval.ppr import (
    HippoRAG2PPR,
    HippoRAGPPRUnavailable,
    metrics,
)
from atlas_memory.core.protocols import fetch_graph_ppr_candidates
from atlas_memory.l2_semantic.kuzu.ram_multigraph import AtlasMultiDiGraph


class FakeGraphStore:
    def __init__(self, nx_graph: AtlasMultiDiGraph) -> None:
        self.nx_graph = nx_graph

    async def get_node_relations(self, subject: str, predicate: str | None = None) -> List[Dict[str, Any]]:
        if not self.nx_graph.has_node(subject):
            return []
        res = []
        for _u, v, data in self.nx_graph.out_edges(subject, data=True):
            res.append({
                "subject": subject,
                "predicate": data.get("predicate", "related_to"),
                "object": v,
                "confidence": data.get("confidence", 0.9),
                "timestamp": data.get("timestamp", 1000.0),
            })
        return res


def test_t2_ppr_budget_timeout_raises_unavailable() -> None:
    g = AtlasMultiDiGraph()
    g.add_node("CORS", type="entity")
    g.add_edge("CORS", "server", predicate="runs_on", confidence=0.9)

    ppr = HippoRAG2PPR()
    # With budget_ms = 0.0, any execution must exceed the budget and raise HippoRAGPPRUnavailable
    with pytest.raises(HippoRAGPPRUnavailable) as exc_info:
        ppr.run("skonfiguruj CORS", g, top_k=5, budget_ms=0.0001)
    assert "budget exceeded" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_t2_gather_candidates_catches_ppr_exception_and_falls_back() -> None:
    g = AtlasMultiDiGraph()
    g.add_node("CORS", type="entity")
    g.add_node("server", type="entity")
    g.add_edge("CORS", "server", predicate="runs_on", confidence=0.9)

    store = FakeGraphStore(g)

    # Enable HippoRAG in environment, but mock fetch_graph_ppr_candidates to raise
    with patch("atlas_memory.cognitive.retrieval.candidates.is_hipporag_enabled", return_value=True), \
         patch("atlas_memory.cognitive.retrieval.candidates.fetch_graph_ppr_candidates", side_effect=HippoRAGPPRUnavailable("PPR timed out")):
        
        # Must NOT raise exception; must gracefully fallback to exact/stemmed graph
        cands = await gather_candidates(
            query="skonfiguruj CORS",
            limit=5,
            graph_store=store,
            kv_store=None,
            vector_store=None,
            fallback_on_ppr_error=True,
        )
        assert isinstance(cands, list)
        # Fallback should have captured 'CORS' from exact/stemmed graph channel
        keys = [c.get("key") for c in cands]
        assert "CORS" in keys


def test_t2_fetch_graph_ppr_candidates_increments_error_counter() -> None:
    g = AtlasMultiDiGraph()
    store = FakeGraphStore(g)

    errs_before = metrics.counter("hipporag_ppr_error").value
    with patch("atlas_memory.cognitive.retrieval.ppr.HippoRAG2PPR.run", side_effect=RuntimeError("Graph corruption")):
        with pytest.raises(HippoRAGPPRUnavailable):
            with patch("atlas_memory.cognitive.retrieval.ppr.is_hipporag_enabled", return_value=True):
                fetch_graph_ppr_candidates(store, "test query")

    errs_after = metrics.counter("hipporag_ppr_error").value
    assert errs_after == errs_before + 1
