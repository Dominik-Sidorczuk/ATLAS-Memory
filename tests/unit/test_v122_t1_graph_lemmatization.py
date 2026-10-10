"""V122 (T1) — Graph Input-Path Lemmatization, Case-Folding and Canonical Resolution.

Contracts tested:
  1. O(1) canonical_lower index: 'cors' -> 'CORS', 'git' -> 'Git'
  2. O(1) lemma_index: 'narzędzia' -> 'narzędzie', 'wazach' -> 'waza'
  3. Multi-word phrase resolution: 'Jaka jest konfiguracja Obsidian MCP?' -> 'Obsidian MCP'
  4. Exact-match API get_node_relations remains untouched (called with canonical name)
  5. fetch_graph_relations_batch retrieves relations for inflected / case-variant queries
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from atlas_memory.core.protocols import fetch_graph_relations_batch
from atlas_memory.l2_semantic.kuzu.ram_multigraph import AtlasMultiDiGraph


class FakeGraphStore:
    """Mock GraphStore wrapping AtlasMultiDiGraph with exact-match get_node_relations."""

    def __init__(self, nx_graph: AtlasMultiDiGraph) -> None:
        self.nx_graph = nx_graph
        self.calls: List[str] = []

    async def get_node_relations(self, subject: str, predicate: str | None = None) -> List[Dict[str, Any]]:
        self.calls.append(subject)
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


def test_t1_atlas_multidigraph_canonical_lower_and_lemma_indices() -> None:
    g = AtlasMultiDiGraph()
    g.add_node("CORS", type="entity")
    g.add_node("LOOP Engineering", type="entity")
    g.add_node("narzędzie", type="entity")

    assert hasattr(g, "resolve_canonical_node"), "AtlasMultiDiGraph must implement resolve_canonical_node"
    assert g.resolve_canonical_node("CORS") == "CORS"
    assert g.resolve_canonical_node("cors") == "CORS"
    assert g.resolve_canonical_node("Cors") == "CORS"
    assert g.resolve_canonical_node("loop engineering") == "LOOP Engineering"

    assert hasattr(g, "resolve_nodes_by_lemma"), "AtlasMultiDiGraph must implement resolve_nodes_by_lemma"
    nodes = g.resolve_nodes_by_lemma("narzędzie")
    assert "narzędzie" in nodes


def test_t1_atlas_multidigraph_resolve_query_entities() -> None:
    g = AtlasMultiDiGraph()
    g.add_node("CORS", type="entity")
    g.add_node("Obsidian MCP", type="entity")
    g.add_node("narzędzie", type="entity")
    g.add_node("waza", type="entity")

    assert hasattr(g, "resolve_query_entities"), "AtlasMultiDiGraph must implement resolve_query_entities"

    # Case 1: Case-insensitive token
    resolved = g.resolve_query_entities(words=["cors", "serwer"], query="jak skonfigurować cors?")
    assert "CORS" in resolved

    # Case 2: Multi-word entity from query
    resolved = g.resolve_query_entities(words=["konfiguracja", "obsidian", "mcp"], query="Gdzie jest konfiguracja Obsidian MCP?")
    assert "Obsidian MCP" in resolved

    # Case 3: Inflected Polish noun via lemma ('narzędzia' -> 'narzędzie')
    resolved = g.resolve_query_entities(words=["narzędzia"], query="Jakie mamy narzędzia?")
    assert "narzędzie" in resolved

    # Case 4: Inflected Polish noun in locative ('wazach' -> 'waza')
    resolved = g.resolve_query_entities(words=["wazach"], query="Co wiadomo o wazach?")
    assert "waza" in resolved


@pytest.mark.asyncio
async def test_t1_fetch_graph_relations_batch_resolves_canonical_names() -> None:
    g = AtlasMultiDiGraph()
    g.add_node("CORS", type="entity")
    g.add_node("server", type="entity")
    g.add_edge("CORS", "server", predicate="runs_on", confidence=0.95)

    g.add_node("narzędzie", type="entity")
    g.add_node("agent", type="entity")
    g.add_edge("narzędzie", "agent", predicate="used_by", confidence=0.88)

    store = FakeGraphStore(g)

    # 1. Query with lowercase 'cors' -> must call get_node_relations with 'CORS'
    edges = await fetch_graph_relations_batch(store, words=["cors"], query="skonfiguruj cors")
    assert len(edges) == 1
    assert edges[0]["subject"] == "CORS"
    assert edges[0]["object"] == "server"
    assert "CORS" in store.calls

    # 2. Query with inflected 'narzędzia' -> must call get_node_relations with 'narzędzie'
    store.calls.clear()
    edges = await fetch_graph_relations_batch(store, words=["narzędzia"], query="pokaż narzędzia")
    assert len(edges) == 1
    assert edges[0]["subject"] == "narzędzie"
    assert "narzędzie" in store.calls
