"""
Canonical unit tests for RetrievalPipeline / PrefetchEngine, EpistemicWritePath, HotBufferPolicy, and Static-Anchor DBSF.
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path

import pytest

from atlas_memory.cognitive.conflict_arbiter import ConflictArbiter
from atlas_memory.cognitive.models import RetrievalRecord
from atlas_memory.cognitive.retrieval.pipeline import RetrievalPipeline
from atlas_memory.cognitive.retrieval.score import score_candidates
from atlas_memory.cognitive.write_path import EpistemicWritePath
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.extensions.compactor import ContextCompactor
from atlas_memory.l1_working.hot_buffer import HotBufferPolicy, is_session_accessible
from atlas_memory.server.atlas_daemon import AtlasDaemon


class MockGraphStore:
    """Mock graph store tracking added relations."""

    def __init__(self) -> None:
        self.relations = []

    def add_relation(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 1.0,
        timestamp: float | None = None,
    ) -> None:
        self.relations.append({
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "confidence": confidence,
            "timestamp": timestamp,
        })


def test_retrieval_pipeline_continuation_and_gating():
    """Verifies intent gating and continuation detection."""
    pipeline = RetrievalPipeline()

    assert pipeline.is_continuation_query("Kontynuuj działanie") is True
    assert pipeline.is_continuation_query("dalej") is True
    assert pipeline.is_continuation_query("continue") is True
    assert pipeline.is_continuation_query("go on") is True
    assert pipeline.is_continuation_query("kolejny krok") is True
    assert pipeline.is_continuation_query("Jaka jest pogoda?") is False

    assert pipeline.is_trivial_prompt("cześć!") is True
    assert pipeline.is_trivial_prompt("dziękuję bardzo") is True
    assert pipeline.is_trivial_prompt("Zbuduj raport finansowy") is False

    can_retrieve, entities, reason = pipeline.should_retrieve("cześć, co tam?")
    assert can_retrieve is False
    assert reason in ("trivial_prompt", "conversational_turn_no_entity")

    can_retrieve, entities, reason = pipeline.should_retrieve("Sprawdź konfigurację test:cluster:node1")
    assert can_retrieve is True
    assert "test:cluster:node1" in entities or any("cluster" in e for e in entities)


@pytest.mark.asyncio
async def test_retrieval_pipeline_hot_buffer_injection():
    """Verifies that empty or continuation queries inject active hot working memory and skip when empty."""
    hot_buffer = HotBufferPolicy(maxlen=10, ttl_seconds=900.0)
    pipeline = RetrievalPipeline(hot_buffer=hot_buffer)

    now = time.time()
    hot_buffer.update(
        key="task:active_task",
        value="Migracja bazy danych w toku",
        confidence=1.0,
        metadata={"session_id": "sess_unit"},
        timestamp=now,
    )
    hot_buffer.update(
        key="task:step",
        value="Krok 4: Weryfikacja schematu",
        confidence=1.0,
        metadata={"session_id": "sess_unit"},
    )

    res_empty = await pipeline.retrieve(query="", session_id="sess_unit")
    assert res_empty.skipped is False
    assert res_empty.count == 2
    assert any(r.subject == "task:active_task" for r in res_empty.records)
    assert "## ATLAS Memory Context" in pipeline.generate_context_block(res_empty.records)

    # V114 (F-114, ADR-114): continuation NIE serwuje już nonszorowanych hoty.
    # Kandydaci hot_buffer mają z konstrukcji score=0.0 (candidates.py
    # `_build_record(..., 0.0)`), więc poniżej progu CONTINUATION_MIN_SCORE=0.18
    # — dokładnie ten szum (1244/1500 tokenów) odrzucił defekt v93.
    # Gorący dowód przechodzi bramkę dopiero gdy fuzja/channel nada mu
    # mierzony score >= 0.18. Gałąź empty_query_hot (powyżej) bez zmian.
    res_cont = await pipeline.retrieve(query="Kontynuuj działanie", session_id="sess_unit")
    assert res_cont.skipped is True
    assert res_cont.count == 0
    assert res_cont.metadata.get("reason") == "continuation_below_threshold"

    hot_buffer.clear()
    res_empty_cont = await pipeline.retrieve(query="Kontynuuj działanie", session_id="sess_unit", force=False)
    assert res_empty_cont.skipped is True
    assert res_empty_cont.count == 0


def test_veracity_first_ranking_and_recency_prior():
    """Verifies veracity hierarchy, authority scoring, logarithmic recency prior, and superseded suppression."""
    pipeline = RetrievalPipeline()
    now = time.time()

    rec_user = RetrievalRecord(subject="fact:cache_ttl", predicate="is", object="300", confidence=1.0, source_type="user_explicit", timestamp=now)
    rec_tool = RetrievalRecord(subject="fact:cache_ttl", predicate="is", object="300", confidence=1.0, source_type="tool_output", timestamp=now)
    rec_doc = RetrievalRecord(subject="fact:cache_ttl", predicate="is", object="300", confidence=1.0, source_type="external_doc", timestamp=now)
    rec_agent = RetrievalRecord(subject="fact:cache_ttl", predicate="is", object="300", confidence=1.0, source_type="agent_inference", timestamp=now)

    assert pipeline.compute_record_rank(rec_user, now=now) > pipeline.compute_record_rank(rec_tool, now=now) > pipeline.compute_record_rank(rec_doc, now=now) > pipeline.compute_record_rank(rec_agent, now=now)

    assert pipeline.get_authority_score("rule:git:commit_format") == 1.10
    assert pipeline.get_authority_score("hermes:user_md:preference") == 1.00
    assert pipeline.get_authority_score("fact:readme_doc") == 0.70
    assert pipeline.get_authority_score("working_memory:temp") == 0.50

    rec_rule_fresh = RetrievalRecord(subject="rule:git:strategy", predicate="use", object="trunk_based", confidence=1.0, source_type="user_explicit", timestamp=now)
    rec_rule_old = RetrievalRecord(subject="rule:git:strategy", predicate="use", object="git_flow", confidence=1.0, source_type="user_explicit", timestamp=now - 30 * 86400.0)

    rank_fresh = pipeline.compute_record_rank(rec_rule_fresh, now=now)
    rank_old = pipeline.compute_record_rank(rec_rule_old, now=now)
    assert rank_fresh > rank_old
    expected_old_bonus = 1.0 / (1.0 + 0.15 * math.log1p(30.0))
    assert math.isclose(rank_fresh - rank_old, 1.0 - expected_old_bonus, rel_tol=1e-3)

    rec_superseded = RetrievalRecord(
        subject="rule:git:strategy",
        predicate="use",
        object="git_flow",
        confidence=1.0,
        source_type="user_explicit",
        is_superseded=True,
        metadata={"is_superseded": True},
        timestamp=now,
    )
    assert pipeline.compute_record_rank(rec_superseded, now=now) == -5.0


def test_knapsack_token_budget_and_compactor():
    """Verifies 0/1 Knapsack respects token budget, suppresses superseded items, and works with ContextCompactor."""
    pipeline = RetrievalPipeline()
    now = time.time()

    records = [
        RetrievalRecord(
            subject=f"fact:key_{i}",
            predicate="value",
            object="data_" * 20,
            confidence=1.0,
            source_type="user_explicit",
            timestamp=now,
        )
        for i in range(10)
    ]
    records.append(
        RetrievalRecord(
            subject="rule:superseded_rule",
            predicate="rule",
            object="This rule is invalid",
            confidence=1.0,
            source_type="user_explicit",
            is_superseded=True,
            metadata={"is_superseded": True},
            timestamp=now,
        )
    )

    token_cost_single = pipeline.estimate_tokens(records[0])
    budget = token_cost_single * 3 + 5
    packed = pipeline.pack_knapsack_01(records, budget=budget, now=now)
    total_tokens = sum(pipeline.estimate_tokens(r) for r in packed)

    assert len(packed) <= 4
    assert total_tokens <= budget
    assert not any(r.is_superseded for r in packed)
    assert not any(r.subject == "rule:superseded_rule" for r in packed)
    assert isinstance(pipeline.compactor, ContextCompactor)


@pytest.mark.asyncio
async def test_epistemic_write_path_belief_revision(tmp_path: Path):
    """Verifies EpistemicWritePath belief revision, KV supersession, SUPERSEDED_BY edge, and hot buffer purge."""
    db_path = str(tmp_path / "test_wp.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    mock_graph = MockGraphStore()
    hot_buffer = HotBufferPolicy()
    arbiter = ConflictArbiter()

    write_path = EpistemicWritePath(
        kv_store=engine.kv,
        graph_store=mock_graph,
        hot_buffer=hot_buffer,
        conflict_arbiter=arbiter,
    )

    res1 = write_path.write(
        key="rule:deployment:strategy",
        value="Wdrażamy wyłącznie przez Docker Swarm",
        confidence=1.0,
        metadata={"source_type": "user_explicit"},
    )
    assert res1["status"] == "ok"
    assert res1["persisted"] is True
    assert len(hot_buffer) == 1

    res2 = write_path.write(
        key="rule:deployment:kubernetes",
        value="Wdrażamy wyłącznie przez Kubernetes Helm",
        confidence=1.0,
        metadata={
            "source_type": "user_explicit",
            "supersedes": "rule:deployment:strategy",
            "is_belief_revision": True,
        },
    )
    assert res2["status"] == "ok"
    assert res2["is_belief_revision"] is True

    old_item = engine.kv.get_sync("rule:deployment:strategy")
    assert old_item is not None
    assert old_item["metadata"]["is_superseded"] is True
    assert old_item["metadata"]["superseded_by"] == "rule:deployment:kubernetes"

    superseded_edges = [rel for rel in mock_graph.relations if rel["predicate"] == "SUPERSEDED_BY"]
    assert len(superseded_edges) >= 1
    assert superseded_edges[0]["subject"] == "rule:deployment:strategy"
    assert superseded_edges[0]["object"] == "rule:deployment:kubernetes"

    active_hot_keys = [e.key for e in hot_buffer.get_active()]
    assert "rule:deployment:strategy" not in active_hot_keys
    assert "rule:deployment:kubernetes" in active_hot_keys


@pytest.mark.asyncio
async def test_epistemic_write_path_same_key_version_revision(tmp_path: Path):
    """Verifies same-key value revision tracks prior_value, increments revision_count, and wires graph transitions."""
    db_path = str(tmp_path / "test_wp_rev.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    mock_graph = MockGraphStore()
    write_path = EpistemicWritePath(kv_store=engine.kv, graph_store=mock_graph)

    key = "config:system:max_threads"
    assert write_path.write(key=key, value="8", confidence=1.0)["revision_count"] == 0
    assert write_path.write(key=key, value="16", confidence=1.0)["revision_count"] == 1
    assert write_path.write(key=key, value="32", confidence=1.0)["revision_count"] == 2

    stored_v2 = engine.kv.get_sync(key)
    assert stored_v2["value"] == "32"
    assert stored_v2["metadata"]["prior_value"] == "16"
    assert stored_v2["metadata"]["revision_count"] == 2


@pytest.mark.asyncio
async def test_epistemic_write_path_multihop_and_no_prose_graph_edges(tmp_path: Path):
    """Verifies multi-hop arrow chain extraction while blocking prose sentences with colons and self-loops."""
    db_path = str(tmp_path / "test_wp_edges.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    mock_graph = MockGraphStore()
    write_path = EpistemicWritePath(kv_store=engine.kv, graph_store=mock_graph)

    res_chain = write_path.write(key="pipeline:flow", value="NodeA -> NodeB -> NodeC", confidence=0.9)
    assert res_chain["persisted"] is True
    leads_to_edges = [rel for rel in mock_graph.relations if rel["predicate"] == "leads_to"]
    assert len(leads_to_edges) == 3

    mock_graph.relations.clear()
    write_path.write(
        key="fact:note",
        value="Spotkanie z zespołem odbędzie się jutro po południu w biurze",
        confidence=0.9,
    )
    assert len(mock_graph.relations) == 0


@pytest.mark.asyncio
async def test_session_isolation_strict_and_prefetch_filtering():
    """Verifies session_a working facts never leak to session_b or anonymous callers, while durable rules remain accessible."""
    assert is_session_accessible("s1", "s1") is True
    assert is_session_accessible("s1", "s2") is False
    assert is_session_accessible("s1", "") is False
    assert is_session_accessible("", "") is True

    with tempfile.TemporaryDirectory() as tmpdir:
        sock_path = Path(tmpdir) / "test_atlas.sock"
        pid_path = Path(tmpdir) / "test_atlas.pid"
        db_path = Path(tmpdir) / "test_atlas.db"

        engine = HybridMemoryEngine.create_default(db_path=str(db_path))
        daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path, engine=engine)

        await daemon._handle_set({"key": "session_a_secret", "value": "TOP_SECRET_ALPHA_42", "session_id": "session_a", "confidence": 1.0})
        await daemon._handle_set({"key": "session_b_secret", "value": "TOP_SECRET_BETA_99", "session_id": "session_b", "confidence": 1.0})

        res_a = await daemon._handle_prefetch({"query": "secret", "session_id": "session_a"})
        subjects_a = [r["subject"] for r in res_a["records"]]
        assert "session_a_secret" in subjects_a
        assert "session_b_secret" not in subjects_a


def test_static_anchor_dbsf_fusion_ranking():
    """Verifies Static-Anchor DBSF suppresses low-dense chatter noise and elevates semantic + lexical hits."""
    cands = [
        {
            "key": "fact:chatter",
            "subject": "fact:chatter",
            "object": "Lubię kawę rano przy pracy nad kodem",
            "vector_score": 0.28,
            "confidence": 0.8,
            "source_type": "agent_inference",
        },
        {
            "key": "rule:api:port",
            "subject": "rule:api:port",
            "object": "Serwer produkcyjny REST API nasłuchuje na porcie 27124",
            "vector_score": 0.68,
            "confidence": 1.0,
            "source_type": "user_explicit",
        },
    ]
    ranked = score_candidates("Jaki jest port produkcyjnego serwera REST API?", cands, fusion_strategy="dbsf")
    assert ranked[0]["key"] == "rule:api:port"
    assert ranked[0]["score"] > ranked[1]["score"] * 2.0
