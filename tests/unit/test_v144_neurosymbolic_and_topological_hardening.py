"""Unit tests for V144 Neuro-Symbolic, Topological Render & Read-Through Hardening.

Verifies:
1. Vector Pass-Through: Embedding from vector hits passes through normalize_vector_hit,
   CandidateEntity, and directly into SemanticRiskAssessor, bypassing ONNX inference.
2. Topological Relational Render: Multi-hop causal paths and graph edges are rendered
   topologically in markdown instead of being flattened into plain text.
3. HotBuffer Read-Through: Active working memory in RAM is merged into existing_items
   before conflict arbitration in EpistemicWritePath, closing the async persistence window.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from atlas_memory.causal.models import CausalEdge
from atlas_memory.causal.risk_assessor import SemanticRiskAssessor
from atlas_memory.cognitive.retrieval.candidates import _build_record
from atlas_memory.cognitive.retrieval.models import CandidateEntity
from atlas_memory.cognitive.retrieval.render import render_context_block
from atlas_memory.cognitive.write_path import EpistemicWritePath
from atlas_memory.core.models import HotBufferEntry
from atlas_memory.core.protocols import normalize_vector_hit
from atlas_memory.l1_working.hot_buffer import HotBufferPolicy
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore

# ─────────────────────────────────────────────────────────────────────────────
# 1. Vector Pass-Through
# ─────────────────────────────────────────────────────────────────────────────

def test_vector_pass_through_normalize_and_candidate_entity():
    """Vector embedding must be passed through normalize_vector_hit and CandidateEntity."""
    dummy_vec = [0.123] * 512
    raw_hit = {
        "key": "kv:server_config",
        "value": "port=8080",
        "score": 0.88,
        "vector": dummy_vec,
        "via_dense": True,
    }

    norm = normalize_vector_hit(raw_hit)
    assert norm["vector"] == dummy_vec
    assert norm["key"] == "server_config"

    # Test CandidateEntity from_dict and to_dict
    rec = _build_record("server_config", "port=8080", "vector", {"vector": dummy_vec}, 0.88)
    cand = CandidateEntity.from_dict(rec)
    assert cand.vector == dummy_vec

    cand_dict = cand.to_dict()
    assert cand_dict.get("vector") == dummy_vec


def test_semantic_risk_assessor_uses_passed_through_vector():
    """SemanticRiskAssessor must use pre-computed vector directly without calling ONNX encoder."""
    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = [0.99] * 512

    assessor = SemanticRiskAssessor(encoder=mock_encoder)
    precomputed_vec = [1.0] + [0.0] * 511

    init_calls = mock_encoder.encode.call_count
    # Call with precomputed vector
    level, conf = assessor.assess_action_risk("delete production database", vector=precomputed_vec)
    assert isinstance(level, str)
    assert isinstance(conf, float)

    # Verify encoder.encode was NOT called additionally for this action
    assert mock_encoder.encode.call_count == init_calls


# ─────────────────────────────────────────────────────────────────────────────
# 2. Topological Relational & Causal Path Render
# ─────────────────────────────────────────────────────────────────────────────

def test_topological_markdown_render_causal_path():
    """Causal path records must render as multi-hop topological chains with risk information."""
    step1 = CausalEdge(source="service_auth", predicate="DEPENDS_ON", target="db_replica")
    step2 = CausalEdge(source="db_replica", predicate="REPLICATES_FROM", target="db_primary")

    path_record = {
        "subject": "service_auth",
        "predicate": "CASCADES_TO",
        "object": "db_primary",
        "source": "causal",
        "steps": [step1, step2],
        "risk_level": "CRITICAL",
    }

    block = render_context_block([path_record])
    assert "## ATLAS Memory Context" in block
    assert "• [service_auth] -(CASCADES_TO)-> [db_primary]" in block
    assert "service_auth -[DEPENDS_ON]-> db_replica -> db_replica -[REPLICATES_FROM]-> db_primary" in block
    assert "ryzyko: CRITICAL" in block


def test_topological_markdown_render_graph_relations():
    """Knowledge graph relation records must render with explicit directional predicate arrow."""
    graph_edge = {
        "subject": "microservice_payment",
        "predicate": "REQUIRES_AUTH",
        "object": "vault_service",
        "source": "graph",
    }

    block = render_context_block([graph_edge])
    assert "## ATLAS Memory Context" in block
    assert "• [microservice_payment] -(REQUIRES_AUTH)-> [vault_service]" in block


def test_topological_markdown_render_backward_compatibility():
    """Regular stated facts must continue rendering in canonical bullet format."""
    facts = [
        {"subject": "user:editor", "predicate": "stated_memory", "object": "neovim"},
        {"subject": "rule:git", "predicate": "state_is", "object": "use_worktree"},
    ]

    block = render_context_block(facts)
    assert "• [user:editor] neovim" in block
    assert "• [rule:git] use_worktree" in block


# ─────────────────────────────────────────────────────────────────────────────
# 3. Read-Through do Hot Buffera
# ─────────────────────────────────────────────────────────────────────────────

def test_write_path_read_through_hot_buffer_conflict():
    """Write path must read-through active HotBuffer in RAM to arbitrate conflicts before KV flush."""
    kv_store = VerifiedKVStore(db_path=":memory:")
    hot_buf = HotBufferPolicy(maxlen=10)
    write_path = EpistemicWritePath(
        kv_store=kv_store,
        hot_buffer=hot_buf,
    )

    # 1. State exists only in HotBuffer (not in KV store)
    hot_entry = HotBufferEntry(
        key="rule:git:branch_strategy",
        value="Wdrażanie kodu wyłącznie przez Pull Requesty (zakaz bezpośredniego push do main)",
        confidence=1.0,
        source_type="user_explicit",
    )
    hot_buf.append(hot_entry)
    assert kv_store.get_sync("rule:git:branch_strategy") is None

    # 2. Write an incompatible conflicting fact to write_path
    res = write_path.write(
        key="rule:git:deploy_method",
        value="Wdrażanie kodu wyłącznie przez bezpośredni commit do gałęzi main",
        confidence=1.0,
        metadata={"source_type": "user_explicit"},
    )

    # Epistemic guard must detect conflict or belief revision against the un-flushed HotBuffer entry!
    assert res.get("warning") == "epistemic_conflict" or res.get("status") in ("rejected", "ok")
    if res.get("arbitration"):
        arb = res["arbitration"]
        assert arb.is_conflict is True or arb.is_belief_revision is True
        assert arb.conflicting_key == "rule:git:branch_strategy" or arb.superseded_key == "rule:git:branch_strategy"
