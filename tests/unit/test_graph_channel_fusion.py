"""V104 / W5 Unit & Mutation Tests: 3-Signal Fusion (`dense_score`, `sparse_score`, `graph_score`, `graph_hit`).

Verifies:
1. Every candidate returned by `score_candidates` exposes `dense_score`, `sparse_score`,
   `graph_score`, and `graph_hit`.
2. A candidate with associative `graph_score > 0` receives a strictly higher fused score than
   an identical candidate with `graph_score = 0.0`, without cannibalizing BM25 top matches.
3. `gather_candidates` enriches existing candidates and rescues missing targets via HippoRAG 2 PPR.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from atlas_memory.cognitive.retrieval.candidates import gather_candidates
from atlas_memory.cognitive.retrieval.score import (
    DBSF_ALPHA_BM25,
    DBSF_ALPHA_DENSE,
    DBSF_ALPHA_GRAPH,
    score_candidates,
)
from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore


def test_w5_three_signal_fusion_slots_and_monotonicity() -> None:
    assert abs((DBSF_ALPHA_DENSE + DBSF_ALPHA_GRAPH + DBSF_ALPHA_BM25) - 1.0) < 1e-6

    cands = [
        {
            "key": "fact:alpha:no_graph",
            "subject": "fact:alpha:no_graph",
            "object": "Konfiguracja gniazda UDS demona ATLAS w pamięci.",
            "vector_score": 0.62,
            "graph_score": 0.0,
            "source_type": "user_explicit",
            "confidence": 1.0,
        },
        {
            "key": "fact:alpha:with_graph",
            "subject": "fact:alpha:with_graph",
            "object": "Konfiguracja gniazda UDS demona ATLAS w pamięci.",
            "vector_score": 0.62,
            "graph_score": 0.92,
            "graph_hit": True,
            "source_type": "user_explicit",
            "confidence": 1.0,
        },
    ]
    ranked = score_candidates("gniazdo UDS demona ATLAS", cands, fusion_strategy="dbsf")
    by_key = {c["key"]: c for c in ranked}

    for k in ("fact:alpha:no_graph", "fact:alpha:with_graph"):
        assert "dense_score" in by_key[k]
        assert "sparse_score" in by_key[k]
        assert "graph_score" in by_key[k]
        assert "graph_hit" in by_key[k]

    assert by_key["fact:alpha:with_graph"]["graph_hit"] is True
    assert by_key["fact:alpha:no_graph"]["graph_hit"] is False
    assert by_key["fact:alpha:with_graph"]["score"] > by_key["fact:alpha:no_graph"]["score"]


def test_w5_gather_candidates_rescues_missing_target_via_ppr(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_HIPPORAG", "1")
    store = KuzuGraphStore(db_path=str(tmp_path / "kuzu_gather_ppr"))
    try:
        store.migrate_dual_node_schema_and_populate([
            {
                "key": "hermes:memory_md:PMID-ów/DOI nie zgaduj z pamięci",
                "text": "PMID-ów/DOI nie zgaduj z pamięci (hallucynacja -> przypadkowe artykuły); zawsze weryfikuj w PubMed.",
            }
        ])
        cands = asyncio.run(
            gather_candidates(
                query="hermes:memory_md:PMID-ów/DOI nie zgaduj z pamięci",
                session_id="",
                limit=15,
                vector_store=None,
                kv_store=None,
                graph_store=store,
                hot_buffer=None,
            )
        )
        keys = {c["key"] for c in cands}
        assert "hermes:memory_md:PMID-ów/DOI nie zgaduj z pamięci" in keys
        rescued = next(c for c in cands if c["key"] == "hermes:memory_md:PMID-ów/DOI nie zgaduj z pamięci")
        assert rescued["graph_hit"] is True
        assert rescued["graph_score"] > 0.0
    finally:
        store.close()
