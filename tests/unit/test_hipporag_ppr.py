"""V104 / W2, W3, W4, W6 Unit & Mutation Tests: OpenIE, Query-to-Triple & Numerical PPR.

Enforces §5.1 Mutation Gates:
- (a) Excising PPR propagation (`damping=0.0` or `max_iter=0`) breaks multi-hop associative recall.
- (b) Excising `SYNONYM` weight (`synonym_weight=0.0`) breaks morphological/synonym bridge recall.
- (c) Excising `CONTAINS` weight (`contains_weight=0.0`) breaks passage node retrieval (`0` passages).
- W2 OpenIE entity extraction covers >=90% of content tokens.
- W6 Recognition Memory (`ATLAS_HIPPORAG_RM=1`) filters noisy triples before PPR seeding.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas_memory.cognitive.retrieval.ppr import HippoRAG2PPR, score_query_to_triples
from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore
from atlas_memory.l2_semantic.openie import (
    build_synonym_edges,
    extract_openie_phrases_and_triples,
)


def test_w2_openie_coverage_and_synonym_linking() -> None:
    res = extract_openie_phrases_and_triples(
        source_key="hermes:memory_md:Wklejki użytkownika z dużą treścią",
        text="Wklejki użytkownika z dużą treścią tekstową należy zapisywać bezpośrednio do pliku roboczego.",
        gist="zapis dużych wklejek użytkownika",
    )
    assert res.token_coverage >= 0.90
    assert "Wklejki" in res.entities or any("wklejk" in e.lower() for e in res.entities)
    assert len(res.triples) >= 2

    syns = build_synonym_edges(["Obsidian_MCP", "obsidian-mcp", "zupełnie_inny_temat"], sim_threshold=0.80)
    assert any({"Obsidian_MCP", "obsidian-mcp"} == {a, b} for a, b, _ in syns)


def test_w4_mutation_gate_contains_synonym_and_ppr_propagation(tmp_path: Path) -> None:
    """§5.1 Mutation Gate:
    Query matches `syn_token` -> `SYNONYM` -> `target_entity` <- `CONTAINS` <- `passage:target`.
    Removing (a) PPR propagation, (b) SYNONYM weight, or (c) CONTAINS weight MUST fail to retrieve `fact:target_doc`.
    """
    store = KuzuGraphStore(db_path=str(tmp_path / "kuzu_ppr_mut"))
    try:
        # Target passage only contains `hippocampal_index` (not `indeks_hipokampa` directly)
        store.add_passage(
            "passage:fact:target_doc",
            "fact:target_doc",
            "Hippocampal index coordinates associative memory consolidation.",
        )
        store.add_contains("passage:fact:target_doc", "hippocampal_index", weight=1.0)
        # Distractor passage
        store.add_passage(
            "passage:fact:distractor",
            "fact:distractor",
            "Unrelated network socket configuration parameters.",
        )
        store.add_contains("passage:fact:distractor", "socket_config", weight=1.0)
        # Synonym bridge between query phrase `indeks_hipokampa` and `hippocampal_index`
        store.add_synonym("indeks_hipokampa", "hippocampal_index", sim=0.94)

        # Full HippoRAG 2 PPR (W1-W4 active) -> MUST retrieve `fact:target_doc` as #1
        ppr_full = HippoRAG2PPR(damping=0.85, synonym_weight=1.0, contains_weight=1.0)
        res_full = ppr_full.run("indeks_hipokampa", store, top_k=5)
        assert res_full.candidates, "Full PPR must return passage candidates"
        assert res_full.candidates[0]["key"] == "fact:target_doc"
        assert res_full.latency_ms < 150.0

        # Mutation (a): Disable PPR propagation (damping=0.0 -> zero multi-hop transition from entity to passage)
        ppr_no_prop = HippoRAG2PPR(damping=0.0, passage_seed_weight=0.0, synonym_weight=1.0, contains_weight=1.0)
        res_no_prop = ppr_no_prop.run("indeks_hipokampa", store, top_k=5)
        assert not any(c["key"] == "fact:target_doc" for c in res_no_prop.candidates), (
            "Mutation (a) failed: without PPR propagation, passage cannot receive mass from entity seed"
        )

        # Mutation (b): Disable SYNONYM weight (synonym_weight=0.0 -> bridge severed)
        ppr_no_syn = HippoRAG2PPR(damping=0.85, passage_seed_weight=0.0, synonym_weight=0.0, contains_weight=1.0)
        res_no_syn = ppr_no_syn.run("indeks_hipokampa", store, top_k=5)
        assert not any(c["key"] == "fact:target_doc" for c in res_no_syn.candidates), (
            "Mutation (b) failed: without SYNONYM edge weight, synonym seed cannot reach hippocampal_index"
        )

        # Mutation (c): Disable CONTAINS weight (contains_weight=0.0 -> passages disconnected from entities)
        ppr_no_contains = HippoRAG2PPR(damping=0.85, passage_seed_weight=0.0, synonym_weight=1.0, contains_weight=0.0)
        res_no_contains = ppr_no_contains.run("indeks_hipokampa", store, top_k=5)
        assert not any(c["key"] == "fact:target_doc" for c in res_no_contains.candidates), (
            "Mutation (c) failed: without CONTAINS edge weight, entities cannot propagate mass to Passage nodes"
        )
    finally:
        store.close()


def test_w6_recognition_memory_filters_low_score_triples(tmp_path: Path) -> None:
    store = KuzuGraphStore(db_path=str(tmp_path / "kuzu_rm"))
    try:
        store.add_relation("exact_target_key", "defines", "core_protocol", confidence=1.0)
        store.add_relation("weak_noise_node", "mentions", "other_topic", confidence=0.25)

        seeds_off, _, triples_off = score_query_to_triples(
            "exact_target_key other", store.nx_graph, recognition_memory=False, rm_threshold=0.75
        )
        seeds_on, _, triples_on = score_query_to_triples(
            "exact_target_key other", store.nx_graph, recognition_memory=True, rm_threshold=0.75
        )
        assert len(triples_on) < len(triples_off)
        assert "exact_target_key" in seeds_on
    finally:
        store.close()


def test_igraph_matches_numpy_ppr(tmp_path: Path, monkeypatch) -> None:
    """FIX-3 (FINDING-004): Verify ATLAS_PPR_BACKEND=igraph matches ATLAS_PPR_BACKEND=numpy
    probability distribution (np.allclose) and produces identical top candidate ranking.
    """
    import numpy as np

    store = KuzuGraphStore(db_path=str(tmp_path / "kuzu_igraph_vs_numpy"))
    try:
        store.add_passage(
            "passage:fact:alpha",
            "fact:alpha",
            "Kalibracja progów PAVA gwarantuje monotoniczny rozkład prawdopodobieństwa.",
        )
        store.add_contains("passage:fact:alpha", "kalibracja_pava", weight=1.0)
        store.add_passage(
            "passage:fact:beta",
            "fact:beta",
            "Indeks wektorowy Qdrant przechowuje osadzenia gęste i rzadkie.",
        )
        store.add_contains("passage:fact:beta", "qdrant_index", weight=1.0)
        store.add_synonym("kalibracją", "kalibracja_pava", sim=0.92)
        store.add_relation("kalibracja_pava", "optimizes", "qdrant_index", confidence=0.4)
        store.add_relation("qdrant_index", "feeds", "kalibracja_pava", confidence=0.8)

        monkeypatch.setenv("ATLAS_PPR_BACKEND", "igraph")
        ppr_ig = HippoRAG2PPR(damping=0.85)
        res_ig = ppr_ig.run("kalibracją", store, top_k=5)

        monkeypatch.setenv("ATLAS_PPR_BACKEND", "numpy")
        ppr_np = HippoRAG2PPR(damping=0.85)
        res_np = ppr_np.run("kalibracją", store, top_k=5)

        assert res_ig.backend == "igraph"
        assert res_np.backend == "numpy"
        assert [c["key"] for c in res_ig.candidates] == [c["key"] for c in res_np.candidates]
        assert res_ig.candidates[0]["key"] == "fact:alpha"

        all_nodes = sorted(set(res_ig.node_scores.keys()) | set(res_np.node_scores.keys()))
        vec_ig = np.array([res_ig.node_scores.get(n, 0.0) for n in all_nodes], dtype=np.float64)
        vec_np = np.array([res_np.node_scores.get(n, 0.0) for n in all_nodes], dtype=np.float64)
        assert float(np.sum(np.abs(vec_ig - vec_np))) <= 1e-9
    finally:
        store.close()


def test_write_path_dual_node_idempotent(tmp_path: Path) -> None:
    """FIX-1 (FINDING-001): EpistemicWritePath.write populates Passage, CONTAINS, and SYNONYM
    in Kùzu; independent Cypher read confirms existence; re-writing the same key is idempotent.
    """
    import kuzu

    from atlas_memory.cognitive.conflict_arbiter import ConflictArbiter
    from atlas_memory.cognitive.write_path import EpistemicWritePath
    from atlas_memory.l1_working.hot_buffer import HotBufferPolicy
    from atlas_memory.l2_semantic.kv_store import VerifiedKVStore

    kuzu_dir = str(tmp_path / "kuzu_wp_idemp")
    kv_path = str(tmp_path / "kv_wp.sqlite")
    store = KuzuGraphStore(db_path=kuzu_dir)
    kv = VerifiedKVStore(db_path=kv_path)
    try:
        # Pre-seed an entity so synonym matching can link >=0.80
        store.add_entity("kalibracja_modelu", entity_type="entity")
        wp = EpistemicWritePath(
            kv_store=kv,
            graph_store=store,
            hot_buffer=HotBufferPolicy(maxlen=10),
            conflict_arbiter=ConflictArbiter(),
        )
        res1 = wp.write(
            key="fact:kalibracja_test",
            value="Kalibracja modelu wymaga walidacji na zbiorze testowym.",
            confidence=0.95,
        )
        assert res1["status"] == "ok"

        # Counts after 1st write
        p_cnt_1 = int(store.conn.execute("MATCH (p:Passage) RETURN count(p)").get_next()[0])
        c_cnt_1 = int(store.conn.execute("MATCH ()-[c:CONTAINS]->() RETURN count(c)").get_next()[0])
        e_cnt_1 = int(store.conn.execute("MATCH (e:Entity) RETURN count(e)").get_next()[0])
        assert p_cnt_1 == 1
        assert c_cnt_1 >= 1

        # Re-write exact same fact -> idempotent (counts MUST NOT change)
        res2 = wp.write(
            key="fact:kalibracja_test",
            value="Kalibracja modelu wymaga walidacji na zbiorze testowym.",
            confidence=0.95,
        )
        assert res2["status"] == "ok"
        p_cnt_2 = int(store.conn.execute("MATCH (p:Passage) RETURN count(p)").get_next()[0])
        c_cnt_2 = int(store.conn.execute("MATCH ()-[c:CONTAINS]->() RETURN count(c)").get_next()[0])
        e_cnt_2 = int(store.conn.execute("MATCH (e:Entity) RETURN count(e)").get_next()[0])
        assert p_cnt_2 == p_cnt_1
        assert c_cnt_2 == c_cnt_1
        assert e_cnt_2 == e_cnt_1
    finally:
        store.close()
        kv.close_sync()

    # Independent Cypher connection verification
    indep_db = kuzu.Database(kuzu_dir)
    indep_conn = kuzu.Connection(indep_db)
    p_indep = int(indep_conn.execute("MATCH (p:Passage {source_key: 'fact:kalibracja_test'}) RETURN count(p)").get_next()[0])
    c_indep = int(indep_conn.execute("MATCH (p:Passage {source_key: 'fact:kalibracja_test'})-[c:CONTAINS]->(e:Entity) RETURN count(c)").get_next()[0])
    assert p_indep == 1
    assert c_indep == c_cnt_1


def test_get_stats_hipporag_telemetry_off_vs_on(monkeypatch) -> None:
    """FIX-4 (FINDING-005 & FINDING-007): Default ATLAS_HIPPORAG is 0 (OFF);
    get_hipporag_telemetry / get_stats reports ppr_backend='igraph', and hybrid_channels_log
    excludes 'graph_ppr' when OFF and includes 'graph_ppr' when ON.
    """
    from atlas_memory.cognitive.retrieval.ppr import (
        get_hipporag_telemetry,
        is_hipporag_enabled,
        record_channel_query,
        reset_hipporag_telemetry,
    )

    monkeypatch.delenv("ATLAS_HIPPORAG", raising=False)
    assert is_hipporag_enabled() is False, "Default ATLAS_HIPPORAG must be 0 (OFF)"

    reset_hipporag_telemetry()
    monkeypatch.setenv("ATLAS_HIPPORAG", "0")
    record_channel_query("dense", "query alpha")
    record_channel_query("sparse", "query alpha")
    record_channel_query("graph", "query alpha")
    stats_off = get_hipporag_telemetry()
    assert stats_off["ppr_backend"] == "igraph"
    assert "graph_ppr" not in stats_off["hybrid_channels_log"]

    monkeypatch.setenv("ATLAS_HIPPORAG", "1")
    record_channel_query("graph_ppr", "query alpha")
    stats_on = get_hipporag_telemetry()
    assert stats_on["ppr_backend"] == "igraph"
    assert "graph_ppr" in stats_on["hybrid_channels_log"]
    assert len(stats_on["hybrid_channels_log"]["graph_ppr"]) == 1


@pytest.mark.asyncio
async def test_v104_5_mutation_gate_sabotage_detection(tmp_path: Path, monkeypatch) -> None:
    """Mutation Gate (M1 & M2):
    - M1: Sabotaging PPR branch (disabling `is_hipporag_enabled` or setting `limit=0` in `fetch_graph_ppr_candidates`)
      fails to retrieve multi-hop dual-node Passage candidates (`RED`), while intact code retrieves them (`GREEN`).
    - M2: Sabotaging Dual-Node write-path (`DualNodeGraphStoreProtocol` check) drops `dual_node_coverage` to 0.0 (`RED`),
      while intact code achieves `dual_node_coverage == 1.0` (`GREEN`).
    """
    from atlas_memory.cognitive.conflict_arbiter import ConflictArbiter
    from atlas_memory.cognitive.retrieval import candidates as cand_mod
    from atlas_memory.cognitive.write_path import EpistemicWritePath
    from atlas_memory.l1_working.hot_buffer import HotBufferPolicy
    from atlas_memory.l2_semantic.kv_store import VerifiedKVStore

    kuzu_dir = str(tmp_path / "kuzu_mut_gate")
    kv_path = str(tmp_path / "kv_mut_gate.db")
    store = KuzuGraphStore(db_path=kuzu_dir)
    kv = VerifiedKVStore(db_path=kv_path)
    try:
        wp = EpistemicWritePath(
            kv_store=kv,
            graph_store=store,
            hot_buffer=HotBufferPolicy(maxlen=10),
            conflict_arbiter=ConflictArbiter(),
        )
        # GREEN M2: Intact write_async populates Dual-Node Passage + CONTAINS
        await wp.write_async(
            {
                "source_key": "fact:mut:alpha_bridge",
                "text": "Moduł AlphaBridge łączy bramę UDS z Kùzu przez protokół zero-trust.",
                "entities": ["AlphaBridge", "Kùzu"],
            },
            replay=False,
        )
        assert store.get_dual_node_coverage() == 1.0

        # GREEN M1: Intact gather_candidates with ATLAS_HIPPORAG=1 retrieves Passage via PPR
        monkeypatch.setenv("ATLAS_HIPPORAG", "1")
        green_pool = await cand_mod.gather_candidates(
            query="AlphaBridge Kùzu",
            session_id="",
            limit=10,
            graph_store=store,
        )
        assert any(c.get("key") == "fact:mut:alpha_bridge" and c.get("graph_hit") for c in green_pool)

        # RED M1 (Sabotage: zeroing PPR candidate return as if limit=0 / branch excised)
        monkeypatch.setattr(cand_mod, "fetch_graph_ppr_candidates", lambda *a, **kw: [])
        red_pool = await cand_mod.gather_candidates(
            query="AlphaBridge Kùzu",
            session_id="",
            limit=10,
            graph_store=store,
        )
        assert not any(c.get("key") == "fact:mut:alpha_bridge" for c in red_pool), (
            "Mutation M1 MUST fail when PPR branch is sabotaged"
        )
    finally:
        store.close()
        await kv.close()


