"""Unit tests for V142 Cognitive & Multi-Store Storage Remediation.

Verifies:
1. HippoRAG 2 Dual-Node 2-Hop Multihop Propagation (Entity-Passage-Entity CONTAINS bridging)
2. Continuous degree damping (degree 20 node not hard-cut, damped by 1/sqrt(deg))
3. Atomic WritePath rollback on vector store failure (Saga compensation)
4. Atomic WritePath rollback on graph store failure (Qdrant deleted, KV & HotBuffer restored)
5. Superseded vector tombstone handling & filtering on Qdrant delete failure
6. C1 smooth, zero-shifted Platt scaling in DenseCosineStrategy (no step jump at zero)
7. Adaptive Shannon Entropy filter stability for small K (K=2/3 cliff bypass & shrinkage)
8. Anti-collision full-content fingerprinting for long template strings
9. Zero-alloc estimate_tokens exact parity with legacy formatter
"""

import math
from typing import Any
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from atlas_memory.cognitive.retrieval.ppr import expand_associative_seeds_2hop
from atlas_memory.cognitive.retrieval.render import estimate_tokens
from atlas_memory.cognitive.retrieval.strategies.dense import DenseCosineStrategy
from atlas_memory.cognitive.retrieval.strategies.fusion import (
    EntropySalienceFilter,
    compute_content_fingerprint,
)
from atlas_memory.cognitive.retrieval.strategies.pipeline import ScoringPipeline
from atlas_memory.cognitive.write_path import EpistemicWritePath
from atlas_memory.core.models import HotBufferEntry
from atlas_memory.core.protocols import normalize_vector_hit, verify_kv_keys_batch

# ─────────────────────────────────────────────────────────────────────────────
# 1. HippoRAG 2 Dual-Node Multihop Propagation & Continuous Damping
# ─────────────────────────────────────────────────────────────────────────────

def test_hipporag_dual_node_multihop_propagation():
    """Entity A bridges to Entity B through intermediate Passage P via CONTAINS edges."""
    g = nx.DiGraph()
    # (Passage) -[:CONTAINS]-> (EntityA), (Passage) -[:CONTAINS]-> (EntityB)
    g.add_node("passage:doc_123", type="passage")
    g.add_node("entity_alpha", type="entity")
    g.add_node("entity_beta", type="entity")

    g.add_edge("passage:doc_123", "entity_alpha", relation="CONTAINS", weight=1.0)
    g.add_edge("passage:doc_123", "entity_beta", relation="CONTAINS", weight=1.0)

    # Seed with entity_alpha
    seeds = {"entity_alpha": 1.0}
    expanded = expand_associative_seeds_2hop(g, seeds, max_expanded=10)

    # entity_beta should be activated via passage:doc_123 bridging
    assert "entity_beta" in expanded
    assert expanded["entity_beta"] > 0.0
    # Passage node itself must not be exposed as seed
    assert "passage:doc_123" not in expanded


def test_continuous_degree_damping():
    """Node with degree 20 must not be discarded, but damped by 1/sqrt(deg)."""
    g = nx.DiGraph()
    g.add_node("seed_concept", type="entity")
    g.add_node("hub_concept", type="entity")

    g.add_edge("seed_concept", "hub_concept", relation="RELATES_TO", weight=1.0)

    # Attach 19 other neighbors to make degree(hub_concept) = 20
    for i in range(19):
        other_node = f"peripheral_{i}"
        g.add_node(other_node, type="entity")
        g.add_edge("hub_concept", other_node, relation="RELATES_TO", weight=0.5)

    assert g.degree("hub_concept") == 20

    seeds = {"seed_concept": 1.0}
    expanded = expand_associative_seeds_2hop(g, seeds, max_expanded=10)

    # hub_concept must be present and damped
    assert "hub_concept" in expanded
    expected_damping = 1.0 / math.sqrt(20.0 / 8.0)
    # The score should reflect seed_w * 0.42 * (edge_w / sqrt(20/8))
    expected_w1 = 1.0 * 0.42 * (1.0 * expected_damping) * 1.0 * 1.0
    assert pytest.approx(expanded["hub_concept"], abs=1e-3) == round(expected_w1, 6)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transactional Saga / 2PC Rollback in EpistemicWritePath
# ─────────────────────────────────────────────────────────────────────────────

def test_write_path_atomic_rollback_on_vector_failure():
    """Simulated vector_store.insert_sync failure aborts transaction and rolls back KV."""
    mock_kv = MagicMock()
    mock_kv.get_arbitration_candidates_sync.return_value = {}
    mock_kv.set_sync = MagicMock()
    mock_kv.delete_sync = MagicMock()

    mock_vec = MagicMock()
    mock_vec.insert_sync.side_effect = RuntimeError("Qdrant connection timeout")

    mock_hot = MagicMock()
    mock_hot._buffer = []
    mock_hot.purge.return_value = 0

    wp = EpistemicWritePath(
        kv_store=mock_kv,
        vector_store=mock_vec,
        hot_buffer=mock_hot,
    )

    with pytest.raises(RuntimeError, match="Qdrant connection timeout"):
        wp.write(
            key="fact:db_port",
            value="5432",
            confidence=1.0,
            vector_record={"dense": [0.1, 0.2]},
        )

    # KV must be deleted/rolled back on new key insert failure
    assert mock_kv.rollback_savepoint.called or mock_kv.delete_sync.called


def test_write_path_atomic_rollback_on_graph_failure():
    """Simulated failure in graph insertion rolls back Qdrant, KV, and HotBuffer."""
    mock_kv = MagicMock()
    # Key previously existed with old value
    mock_kv.get_arbitration_candidates_sync.return_value = {
        "fact:db_port": {
            "value": "5431",
            "confidence": 0.9,
            "metadata": {"source_type": "user_explicit"},
        }
    }
    mock_kv.set_sync = MagicMock()
    mock_kv.delete_sync = MagicMock()

    mock_vec = MagicMock()
    mock_vec.insert_sync = MagicMock()
    mock_vec.delete_by_upsert_key_sync = MagicMock()

    mock_hot = MagicMock()
    old_hot_entry = HotBufferEntry(
        key="fact:db_port",
        value="5431",
        confidence=0.9,
        source_type="user_explicit",
        timestamp=100.0,
        is_superseded=False,
    )
    mock_hot._buffer = [old_hot_entry]
    mock_hot.purge = MagicMock(return_value=1)
    mock_hot.append = MagicMock()

    mock_graph = MagicMock()
    # Graph store fails on add_relation
    mock_graph.add_relation.side_effect = RuntimeError("Kùzu disk full")

    wp = EpistemicWritePath(
        kv_store=mock_kv,
        vector_store=mock_vec,
        hot_buffer=mock_hot,
        graph_store=mock_graph,
    )

    with pytest.raises(RuntimeError, match="Kùzu disk full"):
        wp.write(
            key="fact:db_port",
            value={"port": "5432"},  # triggers relation insertion
            confidence=1.0,
            vector_record={"dense": [0.1, 0.2]},
        )

    # 1. Vector point deleted (dual-key symmetrical compensation)
    mock_vec.delete_by_upsert_key_sync.assert_any_call("fact:db_port")

    # 2. KV restored/rolled back to previous state
    assert mock_kv.rollback_savepoint.called or mock_kv.set_sync.called

    # 3. Hot buffer restored with old entry
    mock_hot.purge.assert_called_with("fact:db_port")
    mock_hot.append.assert_called_with(old_hot_entry)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Superseded Vector Tombstone Handling
# ─────────────────────────────────────────────────────────────────────────────

def test_superseded_vector_tombstone_handling():
    """If vector delete fails on belief revision, KV sets vector_tombstone_pending and prefetch filters it."""
    mock_kv = MagicMock()
    old_item = {
        "value": "5431",
        "confidence": 0.9,
        "metadata": {"source_type": "user_explicit"},
    }
    mock_kv.get_arbitration_candidates_sync.return_value = {
        "fact:db_port_old": old_item
    }
    mock_kv.get_sync.return_value = old_item
    mock_kv.set_sync = MagicMock()

    mock_vec = MagicMock()
    mock_vec.delete_by_upsert_key_sync.side_effect = RuntimeError("Qdrant unreachable")

    mock_hot = MagicMock()
    mock_hot._buffer = []
    mock_hot.purge.return_value = 1

    wp = EpistemicWritePath(
        kv_store=mock_kv,
        vector_store=mock_vec,
        hot_buffer=mock_hot,
    )

    # Force belief revision supersession
    with patch("atlas_memory.cognitive.conflict_arbiter.ConflictArbiter.arbitrate") as mock_arb:
        from atlas_memory.cognitive.models import ArbitrationResult
        mock_arb.return_value = ArbitrationResult(
            allowed=True,
            is_conflict=True,
            is_belief_revision=True,
            superseded_key="fact:db_port_old",
        )
        wp.write(
            key="fact:db_port_new",
            value="5432",
            confidence=1.0,
        )

    # Verify KV store set_sync called with vector_tombstone_pending flag
    tombstone_calls = [
        call for call in mock_kv.set_sync.call_args_list
        if call[1].get("reason") == "vector_tombstone_pending"
    ]
    assert len(tombstone_calls) == 1
    assert tombstone_calls[0][1]["metadata"].get("vector_tombstone_pending") is True

    # Verify normalize_vector_hit and verify_kv_keys_batch filter out tombstoned hits
    raw_hit = {
        "id": "fact:db_port_old",
        "score": 0.85,
        "metadata": {"vector_tombstone_pending": True},
    }
    norm = normalize_vector_hit(raw_hit)
    assert norm["vector_tombstone_pending"] is True

    # Test verify_kv_keys_batch rejects keys whose KV entry has vector_tombstone_pending
    kv_store_with_tombstone = MagicMock(spec=["get_sync"])
    kv_store_with_tombstone.get_sync.return_value = {
        "value": "5431",
        "metadata": {"vector_tombstone_pending": True},
    }
    import asyncio
    verified = asyncio.run(verify_kv_keys_batch(kv_store_with_tombstone, ["fact:db_port_old"]))
    assert "fact:db_port_old" not in verified


# ─────────────────────────────────────────────────────────────────────────────
# 4. Platt Scaling Continuity & Monotonicity
# ─────────────────────────────────────────────────────────────────────────────

def test_platt_scaling_continuity():
    """Verify zero-shift Platt scaling is continuous at 0 and strictly monotonic on [0, 1]."""
    strat = DenseCosineStrategy()

    # Exact zero
    assert strat.normalize_dense(0.0) == 0.0
    assert strat.normalize_dense(-0.5) == 0.0

    # Near-zero limit: vs = 1e-7 must be < 1e-5 (no step jump)
    near_zero = strat.normalize_dense(1e-7)
    assert 0.0 < near_zero < 1e-5

    # Monotonicity check across grid
    prev = 0.0
    for i in range(1, 101):
        vs = i / 100.0
        val = strat.normalize_dense(vs)
        assert val > prev, f"Non-monotonic at vs={vs}: {val} <= {prev}"
        assert 0.0 <= val <= 1.0
        prev = val


# ─────────────────────────────────────────────────────────────────────────────
# 5. Shannon Entropy Filter Small-K Stability
# ─────────────────────────────────────────────────────────────────────────────

def test_entropy_filter_small_k_stability():
    """For K=2 candidates with scores 0.51 and 0.49, second candidate must NOT be cut by cliff."""
    filt = EntropySalienceFilter()
    cands = [
        {"subject": "fact:a", "score": 0.51, "dense_score": 0.5, "sparse_score": 0.0, "graph_score": 0.0},
        {"subject": "fact:b", "score": 0.49, "dense_score": 0.48, "sparse_score": 0.0, "graph_score": 0.0},
    ]

    filtered, eff_thresh = filt.filter(cands, base_threshold=0.0)

    # Both must pass without false cliff pruning
    assert len(filtered) == 2
    assert eff_thresh == 0.0
    assert filtered[0]["subject"] == "fact:a"
    assert filtered[1]["subject"] == "fact:b"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Anti-Collision Full-Content Fingerprint
# ─────────────────────────────────────────────────────────────────────────────

def test_fingerprint_long_template_collision():
    """Two strings identical in first 20 words but differing at word 25 produce unique fingerprints and survive dedup."""
    prefix_words = "the quick brown fox jumps over the lazy dog in the middle of a sunny day with great energy and speed"
    # 20 words prefix
    doc_a = f"{prefix_words} variant alpha parameter setting"
    doc_b = f"{prefix_words} variant omega parameter setting"

    fp_a = compute_content_fingerprint(doc_a)
    fp_b = compute_content_fingerprint(doc_b)

    assert fp_a != fp_b, "Long template strings must not collide"

    pipe = ScoringPipeline()
    cands = [
        {"key": "cfg:a", "subject": "cfg:a", "object": doc_a, "source": "kv", "vector_score": 0.8},
        {"key": "cfg:b", "subject": "cfg:b", "object": doc_b, "source": "kv", "vector_score": 0.8},
    ]
    deduped = pipe.dedup_candidates(cands)
    assert len(deduped) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 7. Zero-Alloc Token Estimation Parity
# ─────────────────────────────────────────────────────────────────────────────

def test_estimate_tokens_zero_alloc():
    """Zero-alloc estimate_tokens returns identical value to legacy string formatting."""
    def legacy_estimate(record: Any) -> int:
        s = record.get("subject", "") if isinstance(record, dict) else getattr(record, "subject", "")
        o = str(record.get("object", "") if isinstance(record, dict) else getattr(record, "object", ""))
        return max(1, math.ceil(len(f"• [{s}] {o[:1200]}") / 3.5) + 1)

    test_samples = [
        {"subject": "k1", "object": "short"},
        {"subject": "config:database:postgresql:port", "object": "5432"},
        {"subject": "long_subject_" * 10, "object": "long_object_" * 50},
        {"subject": "boundary", "object": "x" * 1200},
        {"subject": "overflow", "object": "y" * 5000},
        {"subject": "", "object": ""},
    ]

    for sample in test_samples:
        legacy_val = legacy_estimate(sample)
        new_val = estimate_tokens(sample)
        assert new_val == legacy_val, f"Mismatch on {sample['subject'][:20]}: {new_val} != {legacy_val}"
