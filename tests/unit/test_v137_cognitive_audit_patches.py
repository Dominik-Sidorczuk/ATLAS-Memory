"""Unit tests verifying Milestone v137.0: Architectural remediation of cognitive audit findings.

Findings covered:
FLE-1 / K-03 [P0]: Concurrency thread-safety of AtlasMultiDiGraph under parallel mutations.
FLE-2 / K-04 [P1]: Stale-signature invalidation of _get_or_build_triple_index via revision.
FLE-3 / K-05 [P0]: Elimination of fusion cap 0.18 bottleneck for strong graph hits (>= 0.70) matching 0.20 gate.
FLE-4 [P2]: PII whitelist precision: secret = <val> detected as sensitive; token/tokenizer/shared_secret whitelisted.
FLE-5 [P2]: Config & .env parsing robustness: strip comments without space; expanded parse_bool vocabulary.
FLE-6 [P2]: Canonical Knapsack 0/1 delegation in prefetch without premature break.
FLE-7 [P3]: SemanticRiskAssessor cache eviction without full-wipe storm.
K-01 [P0]: Session ID preservation for multi-underscore Hermes session IDs.
K-02 [P0]: JEPA What-If simulation purity (no state pollution).
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from atlas_memory.causal.risk_assessor import SemanticRiskAssessor
from atlas_memory.cognitive.retrieval.gate import evaluate_gate
from atlas_memory.cognitive.retrieval.ppr import _get_or_build_triple_index
from atlas_memory.cognitive.retrieval.prefetch import PrefetchEngine
from atlas_memory.cognitive.retrieval.strategies.fusion import StaticAnchorDbsfFusion
from atlas_memory.core.config import _load_env_file_dict, parse_bool
from atlas_memory.core.models import ActionPlan, is_pii_entity
from atlas_memory.l1_working.hot_buffer import extract_effective_session, is_session_accessible
from atlas_memory.l1_working.jepa_latent import JEPALatentBuffer
from atlas_memory.l2_semantic.kuzu.ram_multigraph import AtlasMultiDiGraph


def test_atlas_multigraph_concurrency_thread_safety():
    """FLE-1 / K-03: Concurrent node/edge mutations do not corrupt AtlasMultiDiGraph."""
    g = AtlasMultiDiGraph()

    def writer(thread_id: int):
        for i in range(100):
            node_name = f"t{thread_id}_n{i}"
            g.add_node(node_name)
            if i > 0:
                g.add_edge(f"t{thread_id}_n{i-1}", node_name, predicate="NEXT", confidence=0.9)
            if i % 10 == 0:
                g.remove_node(node_name)

    def reader():
        count = 0
        for _ in range(50):
            with g.lock:
                for _u, _v, _data in g.edges(data=True):
                    count += 1
        return count

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        writers = [executor.submit(writer, tid) for tid in range(4)]
        readers = [executor.submit(reader) for _ in range(2)]
        for f in concurrent.futures.as_completed(writers + readers):
            f.result()  # Must not raise RuntimeError: dictionary changed size during iteration

    assert g.number_of_nodes() > 0


def test_triple_index_revision_cache_invalidation():
    """FLE-2 / K-04: _get_or_build_triple_index invalidates cache when revision changes even if (N, E) is identical."""
    g = AtlasMultiDiGraph()
    g.add_node("passage:1", type="passage", source_key="rule:1", text="original text")
    g.add_node("concept:a")
    g.add_edge("concept:a", "passage:1", predicate="CONTAINS")

    idx1 = _get_or_build_triple_index(g)
    assert len(idx1.passage_entries) == 1
    assert idx1.passage_entries[0][1] == "rule:1"

    # Replace node data while keeping node count (2) and edge count (1) identical
    g.remove_node("passage:1")
    g.add_node("passage:2", type="passage", source_key="rule:2", text="updated text")
    g.add_edge("concept:a", "passage:2", predicate="CONTAINS")

    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1

    idx2 = _get_or_build_triple_index(g)
    assert idx2 is not idx1
    assert idx2.passage_entries[0][1] == "rule:2"


def test_fusion_cap_harmonized_with_gate():
    """FLE-3 / K-05: Strong graph hits (gs >= 0.70) receive multihop tiebreak to exceed 0.20 and pass the gate."""
    strategy = StaticAnchorDbsfFusion()
    candidates = [
        {
            "key": "scoped:rule:kuzu_indexing",
            "subject": "scoped:rule:kuzu_indexing",
            "object": "graph relation details",
            "vector_score": 0.05,
            "bm25_score": 0.10,
            "graph_score": 0.85,
            "source": "graph",
        }
    ]
    strategy.score("kuzu indexing query", candidates)
    final_score = candidates[0]["score"]
    assert final_score >= 0.20, f"Expected final_score >= 0.20, got {final_score}"

    gate = evaluate_gate("kuzu indexing query", candidates)
    assert gate.should_retrieve is True
    assert gate.reason in ("graph_hit", "accepted")


def test_pii_whitelist_and_secret_detection():
    """FLE-4: secret assignment is flagged as sensitive while domain terms remain whitelisted."""
    assert is_pii_entity("secret = my_super_secret_token_123") is True
    assert is_pii_entity("secret: admin_password") is True
    assert is_pii_entity("secret_key") is True
    assert is_pii_entity("app_secret") is True
    assert is_pii_entity("secret") is True

    # Domain / LLM terms remain clean
    assert is_pii_entity("tokenizer") is False
    assert is_pii_entity("token_budget") is False
    assert is_pii_entity("max_tokens") is False
    assert is_pii_entity("token") is False
    assert is_pii_entity("tokens") is False
    assert is_pii_entity("shared_secret") is False


def test_config_env_comment_stripping_and_bool_parsing(tmp_path: Path):
    """FLE-5: .env comments stripped without preceding space, expanded parse_bool vocabulary."""
    env_file = tmp_path / "test.env"
    env_file.write_text(
        "KEY1=val1#inline_comment\n"
        "KEY2=val2 #spaced_comment\n"
        "KEY3=\"val3#inside_quotes\"\n"
        "ATLAS_FLAG_1=enabled\n"
        "ATLAS_FLAG_2=disabled\n",
        encoding="utf-8",
    )
    parsed = _load_env_file_dict(env_file)
    assert parsed["KEY1"] == "val1"
    assert parsed["KEY2"] == "val2"
    assert parsed["KEY3"] == "val3#inside_quotes"

    assert parse_bool("enabled") is True
    assert parse_bool("enable") is True
    assert parse_bool("disabled") is False
    assert parse_bool("disable") is False
    assert parse_bool("none") is False
    assert parse_bool("stop") is False
    # Unrecognized falls back to default
    assert parse_bool("random_noise", default=False) is False
    assert parse_bool("random_noise", default=True) is True


def test_prefetch_knapsack_canonical_delegation():
    """FLE-6: PrefetchEngine._trim_by_token_budget uses canonical knapsack 0/1 without early break."""
    records = [
        {"subject": "high_val_heavy", "object": "x" * 200, "score": 0.70},
        {"subject": "light_1", "object": "short fact 1", "score": 0.85},
        {"subject": "light_2", "object": "short fact 2", "score": 0.80},
    ]
    # Total budget allows light_1 + light_2, but not heavy + light_1
    budgeted, tot = PrefetchEngine._trim_by_token_budget(records, budget=40)
    subjs = [r["subject"] for r in budgeted]
    assert "light_1" in subjs
    assert "light_2" in subjs
    assert "high_val_heavy" not in subjs


def test_semantic_risk_assessor_cache_eviction():
    """FLE-7: Cache does not full-wipe at capacity, avoiding re-encoding storms."""
    assessor = SemanticRiskAssessor()
    mock_enc = MagicMock()
    mock_enc.encode.side_effect = lambda t: np.ones(16, dtype=np.float32)
    assessor.encoder = mock_enc

    # Fill cache up to capacity
    for i in range(4100):
        assessor._encode_text(f"query_{i}")

    # Cache should be bounded around 4096 and NOT completely empty
    assert len(assessor._encode_cache) >= 3500


def test_hermes_session_full_id_preservation():
    """K-01: Multi-underscore Hermes session IDs are preserved completely."""
    session_id = "session_20261004_170710_6717e1"
    key = f"{session_id}:rule:database"

    eff = extract_effective_session(key)
    assert eff == "20261004_170710_6717e1"

    # Both forms accessible
    assert is_session_accessible(eff, session_id) is True
    assert is_session_accessible(eff, "20261004_170710_6717e1") is True


def test_jepa_what_if_purity():
    """K-02: Counterfactual What-If simulation from same state preserves initial state purity."""
    buf = JEPALatentBuffer(state_dim=8, action_dim=4)
    s0 = buf.current_state

    a1 = ActionPlan(name="action_explore_left", parameters={})
    a2 = ActionPlan(name="action_explore_right", parameters={})

    t1 = buf.predict_transition(s0, a1)
    t2 = buf.predict_transition(s0, a2)

    # Initial state must remain intact
    assert buf.current_state == s0
    assert len(buf.history) == 1

    # Advancing step updates state (vectors and step_index match predicted transition)
    buf.step(a1)
    assert np.allclose(buf.current_state.vector, t1.predicted_state.vector)
    assert buf.current_state.step_index == t1.predicted_state.step_index
    assert len(buf.history) == 2

    # Direct commit_transition sets the exact transition state
    buf.commit_transition(t2)
    assert buf.current_state == t2.predicted_state
    assert len(buf.history) == 3
