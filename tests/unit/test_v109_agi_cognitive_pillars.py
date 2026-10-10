"""V109 AGI-Level Cognitive Pillars Unit & Integration Tests.

Tests the 3 Pillars of V109:
1. Pillar 1 (L1 -> L2 Dynamic Reconciliation & Superseded Eviction):
   - Session end consolidates L1 working facts, arbitrates conflicts via ConflictArbiter,
     and evicts superseded records from HotWorkingBuffer, Qdrant, and Kùzu.
2. Pillar 2 (Multi-Hop Associative Expansion in HippoRAG 2):
   - 2-hop seed expansion (CONTAINS -> SYNONYM -> REGULATES) projects query concepts
     onto 2nd-degree entities and regulatory rules (rule:*, constraint:*).
3. Pillar 3 (Adaptive Entropy Salience Thresholding):
   - Normalized Shannon entropy filter and ungrounded graph-tail damping suppress
     weak graph-only associations while preserving grounded multi-signal candidates.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from atlas_memory.cognitive.retrieval.ppr import HippoRAG2PPR, expand_associative_seeds_2hop
from atlas_memory.cognitive.retrieval.score import (
    apply_entropy_salience_filter,
    compute_normalized_shannon_entropy,
    score_candidates,
)
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.hermes.prefix_guard import HermesSessionHook
from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore


@pytest.mark.asyncio
async def test_pillar1_l1_to_l2_dynamic_reconciliation_and_superseded_eviction(tmp_path: Path) -> None:
    """V109 Pillar 1: Session end arbitrates L1 working memory conflicts and evicts superseded records."""
    engine = HybridMemoryEngine.create_default(
        db_path=str(tmp_path / "kv_v109.db"),
        qdrant_location=":memory:",
        kuzu_path=str(tmp_path / "kuzu_v109"),
    )
    try:
        await engine.kv.set_state(
            key="rule:storage:engine_old",
            value="Zawsze używaj silnika SQLite bez grafu Kùzu.",
            confidence=0.80,
            metadata={"session_id": "sess_v109"},
        )
        await engine.kv.set_state(
            key="rule:storage:engine_new",
            value="Nigdy nie używaj samego SQLite, zawsze używaj grafu Kùzu.",
            confidence=0.98,
            metadata={"session_id": "sess_v109"},
        )

        engine.hot_buffer.update(
            key="rule:storage:engine_old",
            value="Zawsze używaj silnika SQLite bez grafu Kùzu.",
            confidence=0.80,
            metadata={"session_id": "sess_v109"},
        )
        engine.hot_buffer.update(
            key="rule:storage:engine_new",
            value="Nigdy nie używaj samego SQLite, zawsze używaj grafu Kùzu.",
            confidence=0.98,
            metadata={"session_id": "sess_v109"},
        )

        evicted_keys = []
        orig_del = getattr(engine.vector_store, "delete_by_upsert_key_sync", None)
        engine.vector_store.delete_by_upsert_key_sync = MagicMock(
            side_effect=lambda k: (evicted_keys.append(k), orig_del(k) if callable(orig_del) else 1)[1]
        )

        hook = HermesSessionHook(engine)
        stats = await hook.on_session_end(session_id="sess_v109", trigger_sleep_cycle=True)

        assert stats.consolidated_records >= 1, "session/end must consolidate L1/superseded records (> 0)"
        assert stats.facts_superseded >= 1, "ConflictArbiter in consolidation must supersede the old conflicting rule"

        active_keys = {e.key for e in engine.hot_buffer.get_active(session_id="sess_v109")}
        assert "rule:storage:engine_old" not in active_keys
        assert "rule:storage:engine_new" in active_keys

        old_state = await engine.kv.get_state("rule:storage:engine_old")
        assert old_state is not None
        assert old_state.get("metadata", {}).get("is_superseded") is True
    finally:
        await engine.kv.close()


def test_pillar2_hipporag2_multihop_associative_expansion(tmp_path: Path) -> None:
    """V109 Pillar 2: 2-hop associative expansion connects concept query to 2nd-degree regulatory rule."""
    store = KuzuGraphStore(db_path=str(tmp_path / "kuzu_v109_ppr"))
    try:
        store.add_entity("architektura_bazy", entity_type="concept")
        store.add_entity("warstwa_grafowa", entity_type="concept")
        store.add_entity("rule:db:kuzu_acid", entity_type="rule")

        # Hop 1: architektura_bazy --SYNONYM--> warstwa_grafowa
        store.add_synonym("architektura_bazy", "warstwa_grafowa", sim=0.92)
        # Hop 2: warstwa_grafowa --REGULATES--> rule:db:kuzu_acid
        store.add_relation("warstwa_grafowa", "REGULATES", "rule:db:kuzu_acid", confidence=0.95)

        # Passage linked only to the 2nd-hop regulatory rule
        store.add_passage(
            "passage:rule:db:kuzu_acid",
            "rule:db:kuzu_acid",
            "Zawsze stosuj transakcje ACID i dual-node rollback w warstwie Kùzu.",
        )
        store.add_contains("passage:rule:db:kuzu_acid", "rule:db:kuzu_acid", weight=1.0)

        expanded = expand_associative_seeds_2hop(
            nx_graph=store.nx_graph,
            phrase_seeds={"architektura_bazy": 1.0},
            max_expanded=16,
        )
        assert "warstwa_grafowa" in expanded, "Hop-1 synonym entity must be in expanded seeds"
        assert "rule:db:kuzu_acid" in expanded, "Hop-2 regulated rule must be in expanded seeds"
        assert expanded["rule:db:kuzu_acid"] > 0.15

        ppr = HippoRAG2PPR(damping=0.85, synonym_weight=1.0, contains_weight=1.0)
        res = ppr.run("architektura_bazy", store, top_k=5)
        assert any(c["key"] == "rule:db:kuzu_acid" for c in res.candidates)
    finally:
        store.close()


def test_pillar3_adaptive_entropy_salience_gate_and_ungrounded_graph_damping() -> None:
    """V109 Pillar 3: Ungrounded graph-only hits are damped and pruned by the Shannon entropy salience filter."""
    candidates = [
        {
            "key": "fact:grounded:architecture",
            "subject": "fact:grounded:architecture",
            "object": "Architektura bazy danych ATLAS opiera się na Qdrant, Kùzu oraz SQLite WAL.",
            "vector_score": 0.88,
            "lexical_score": 4.2,
            "graph_score": 0.65,
            "confidence": 0.95,
            "source_type": "user_explicit",
        },
        {
            "key": "fact:noise:weak_graph_hop",
            "subject": "fact:noise:weak_graph_hop",
            "object": "Pogoda w Warszawie jutro będzie słoneczna i bezwietrzna.",
            "vector_score": 0.0,
            "lexical_score": 0.0,
            "graph_score": 0.42,
            "confidence": 0.60,
            "source_type": "agent_inference",
        },
    ]

    scored = score_candidates(query="architektura bazy danych ATLAS", candidates=candidates)
    by_key = {c["key"]: c for c in scored}

    assert by_key["fact:grounded:architecture"]["score"] >= 0.70
    assert by_key["fact:noise:weak_graph_hop"]["score"] <= 0.22, (
        f"Ungrounded graph-only candidate must be damped, got {by_key['fact:noise:weak_graph_hop']['score']}"
    )

    filtered, eff_thresh = apply_entropy_salience_filter(scored, base_threshold=0.25)
    filtered_keys = [c["key"] for c in filtered]
    assert "fact:grounded:architecture" in filtered_keys
    assert "fact:noise:weak_graph_hop" not in filtered_keys

    flat_scores = [0.31, 0.30, 0.29, 0.30, 0.28, 0.31]
    h_norm = compute_normalized_shannon_entropy(flat_scores)
    assert h_norm > 0.98
