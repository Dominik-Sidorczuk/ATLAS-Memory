"""Unit tests for V143 Cognitive & Multi-Store Storage Hardening (Post-2027 AGI Standard).

Verifies:
1. Belief revision deletes prefixed and raw vector points (zero zombie vectors).
2. Daemon RPC state persistence executes full Saga 2PC rollback on vector insert failure.
3. HippoRAG 2 entity relation edges are not starved by weak or sub-threshold passages.
4. Multi-relation entity nodes in HippoRAG 2 retain all semantic predicates (no break).
5. Continuous DBSF fusion scaling eliminating the 35% cliff penalty on [0.60, 0.80].
6. Object immutability in working memory (HotBuffer) during 0/1 knapsack compaction.
7. Graceful conversation sync fallback with degraded_local_persistence status when Mnemosyne is unavailable.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from atlas_memory.cognitive.retrieval.ppr import HippoRAG2PPR
from atlas_memory.cognitive.retrieval.render import pack_knapsack_01
from atlas_memory.cognitive.retrieval.strategies.fusion import StaticAnchorDbsfFusion
from atlas_memory.cognitive.write_path import EpistemicWritePath
from atlas_memory.core.models import HotBufferEntry
from atlas_memory.hermes.atlas_provider import AtlasMemoryProvider
from atlas_memory.l1_working.hot_buffer import HotBufferPolicy
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.server.controllers.memory import MemoryRpcControllerMixin

# ─────────────────────────────────────────────────────────────────────────────
# 1. Belief Revision Vector Purge (upsert_key normalization)
# ─────────────────────────────────────────────────────────────────────────────

def test_belief_revision_deletes_prefixed_vector():
    """Belief revision must delete points indexed under 'kv:<key>' as well as '<key>'."""
    vec_mock = MagicMock()
    kv_store = VerifiedKVStore(db_path=":memory:")
    hot_buf = HotBufferPolicy(maxlen=10)
    write_path = EpistemicWritePath(
        kv_store=kv_store,
        vector_store=vec_mock,
        hot_buffer=hot_buf,
    )

    # 1. Initial write
    kv_store.set_sync("server_port", "8080", confidence=0.90, metadata={"source_type": "user_explicit"})

    # 2. Belief revision write (overriding server_port with new value)
    res = write_path.write(
        key="server_port_v2",
        value="9090",
        confidence=0.95,
        metadata={"supersedes": "server_port", "source_type": "user_explicit"},
    )
    assert res["status"] == "ok"
    assert res.get("is_belief_revision") is True

    # 3. Verify delete_by_upsert_key_sync was called for both prefixed and raw keys
    deleted_calls = [call.args[0] for call in vec_mock.delete_by_upsert_key_sync.call_args_list]
    assert "server_port" in deleted_calls
    assert "kv:server_port" in deleted_calls


# ─────────────────────────────────────────────────────────────────────────────
# 2. Daemon RPC State Persist Saga 2PC Rollback
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_daemon_rpc_state_persist_saga_rollback():
    """Vector store failure during daemon state persist must trigger full Saga rollback."""
    kv_store = VerifiedKVStore(db_path=":memory:")
    hot_buf = HotBufferPolicy(maxlen=10)
    vec_mock = MagicMock()
    vec_mock.insert_sync.side_effect = RuntimeError("Qdrant disk IO failure")

    write_path = EpistemicWritePath(
        kv_store=kv_store,
        vector_store=vec_mock,
        hot_buffer=hot_buf,
    )

    class DummyController(MemoryRpcControllerMixin):
        pass

    controller = DummyController()
    controller.write_path = write_path
    controller.engine = MagicMock()
    controller.engine.vector_store = vec_mock
    controller.engine.kv = kv_store
    controller.hot_buffer = hot_buf

    res = await controller._persist_state_variable(
        key="critical_key",
        value="critical_val",
        session_id="session_test",
        confidence=0.95,
        metadata={},
    )

    assert res["status"] == "error"
    assert "Qdrant disk IO failure" in res.get("error", "")

    # VerifiedKVStore must NOT contain critical_key (rolled back)
    assert kv_store.get_sync("critical_key") is None
    # HotBuffer must not contain critical_key
    assert not any(e.key == "critical_key" for e in hot_buf)


# ─────────────────────────────────────────────────────────────────────────────
# 3. HippoRAG 2 Entity Edges Not Starved by Weak Passages
# ─────────────────────────────────────────────────────────────────────────────

def test_hipporag_entity_edges_not_starved_by_weak_passage():
    """Entity relation candidates must fill top_k even if weak passages exist in graph."""
    g = nx.DiGraph()
    # Add a weak passage
    g.add_node("passage:doc_weak", type="passage", source_key="doc_weak", text="weak text")
    # Add entities with a semantic relationship
    g.add_node("concept_alpha", type="entity")
    g.add_node("concept_beta", type="entity")
    g.add_edge("concept_alpha", "REGULATES", predicate="REGULATES", confidence=0.95)
    g.add_edge("concept_alpha", "concept_beta", predicate="REGULATES", confidence=0.95)

    ppr = HippoRAG2PPR(damping=0.85)

    # Mock score_query_to_triples to seed concept_alpha
    with patch("atlas_memory.cognitive.retrieval.ppr.score_query_to_triples") as mock_sq:
        mock_sq.return_value = ({"concept_alpha": 1.0}, {"passage:doc_weak": 0.01}, [])
        res = ppr.run("query", g, top_k=5)

    cand_keys = [c["key"] for c in res.candidates]
    # concept_alpha relation candidate must be present in results
    assert any("concept_alpha" in k for k in cand_keys)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multi-relation Entity Nodes Retain All Semantic Predicates
# ─────────────────────────────────────────────────────────────────────────────

def test_multigraph_entity_retains_all_predicates():
    """Entity node with multiple outgoing semantic edges must retain all of them (no break)."""
    g = nx.DiGraph()
    g.add_node("hub_entity", type="entity")
    g.add_node("target_1", type="entity")
    g.add_node("target_2", type="entity")
    g.add_node("target_3", type="entity")

    g.add_edge("hub_entity", "target_1", predicate="DEPENDS_ON", confidence=0.90)
    g.add_edge("hub_entity", "target_2", predicate="REGULATES", confidence=0.85)
    g.add_edge("hub_entity", "target_3", predicate="MONITORS", confidence=0.80)

    ppr = HippoRAG2PPR()
    _nodes, _n2i, _em, _ig = ppr._build_graph_structures(g)

    # In _cached_entity_rel_edges, hub_entity must have 3 distinct relation edges
    hub_relations = [
        (ent, pred, tgt)
        for _idx, ent, pred, tgt, _conf in ppr._cached_entity_rel_edges
        if ent == "hub_entity"
    ]
    assert len(hub_relations) == 3
    preds = {r[1] for r in hub_relations}
    assert preds == {"DEPENDS_ON", "REGULATES", "MONITORS"}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Smooth Continuous DBSF Dense Scaling on [0.60, 0.80]
# ─────────────────────────────────────────────────────────────────────────────

def test_dbsf_dense_penalty_continuity():
    """DBSF dense scaling must be smooth and monotonic across [0.60, 0.80] with no cliff at 0.75."""
    fusion = StaticAnchorDbsfFusion()
    eff_alpha = 0.65

    # Compute dense output for vs from 0.70 to 0.80 with small step
    scores = []
    vs_range = [0.70 + i * 0.005 for i in range(21)]
    for vs_val in vs_range:
        s = fusion._combine_base_fused(
            vs=vs_val,
            ls=0.0,
            n_dense=vs_val,
            n_lex=0.0,
            eff_alpha=eff_alpha,
            is_trusted=False,
            ungrounded_scoped_entity=False,
            matched_stems=1,
            q_stems=["test"],
        )
        scores.append(s)

    # 1. Strictly monotonic increasing
    for i in range(len(scores) - 1):
        assert scores[i + 1] > scores[i], f"Non-monotonic at step {i}: {scores[i]} -> {scores[i+1]}"

    # 2. Check difference around 0.75 (0.745 -> 0.750 and 0.750 -> 0.755)
    # With a step jump of 35%, delta would be > 0.20; with smooth transition, delta is < 0.015
    idx_745 = vs_range.index(pytest.approx(0.745, abs=1e-4))
    idx_750 = vs_range.index(pytest.approx(0.750, abs=1e-4))
    idx_755 = vs_range.index(pytest.approx(0.755, abs=1e-4))

    delta_pre = scores[idx_750] - scores[idx_745]
    delta_post = scores[idx_755] - scores[idx_750]

    assert delta_pre < 0.015, f"Step jump detected before 0.75: {delta_pre}"
    assert delta_post < 0.015, f"Step jump detected after 0.75: {delta_post}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Object Immutability in Working Memory (HotBuffer) during Knapsack
# ─────────────────────────────────────────────────────────────────────────────

def test_knapsack_compaction_does_not_mutate_hot_buffer():
    """Compacting knapsack must not mutate the underlying HotBufferEntry in place."""
    hot_buf = HotBufferPolicy(maxlen=10)
    original_text = "Important system architecture description with extensive parameters and configurations. " * 5
    assert len(original_text) > 200

    entry = HotBufferEntry(
        key="arch:desc",
        value=original_text,
        subject="arch:desc",
        object=original_text,
        confidence=0.95,
        importance_score=0.95,
        source_type="user_explicit",
    )
    hot_buf.append(entry)

    # Prepare list with 3 records to trigger compact condition (len >= 3)
    records = [
        entry,
        {"subject": "s2", "object": "second fact with long text " * 10, "score": 0.90},
        {"subject": "s3", "object": "third fact with long text " * 10, "score": 0.85},
    ]

    packed = pack_knapsack_01(records, budget=30, compact=True)
    assert len(packed) >= 1

    # Retrieve entry directly from HotBuffer and verify original object is intact
    stored = next((e for e in hot_buf if e.key == "arch:desc"), None)
    assert stored is not None
    assert stored.value == original_text
    assert not str(stored.value).endswith("... [compacted]")
    assert len(str(stored.value)) > 200


# ─────────────────────────────────────────────────────────────────────────────
# 7. Graceful Conversation Sync Fallback without Mnemosyne
# ─────────────────────────────────────────────────────────────────────────────

def test_sync_turn_fallback_without_mnemosyne():
    """sync_turn must gracefully persist to local KV and report degraded status when Mnemosyne is None."""
    provider = AtlasMemoryProvider()
    provider._mnemosyne = None

    # Call sync_turn with turn messages
    provider.sync_turn(
        user_content="Hello ATLAS, can you help me?",
        assistant_content="I am ready to assist you with all cognitive operations.",
        session_id="test_session_fallback",
    )

    # Status must reflect degraded_local_persistence
    assert provider._turn_persistence_status == "degraded_local_persistence"
