"""
Unit tests for RetrievalPipeline, EpistemicWritePath, and HotBufferPolicy.

Validates:
1. RetrievalPipeline:
   - Intent and continuation query gating (should_retrieve, continuation pattern matching).
   - Empty & continuation query injection of active hot working memory from HotBufferPolicy.
   - Veracity-First Ranking with Dynamic Recency Prior:
     * user_explicit (1.0) > tool_output (0.85) > external_doc (0.65) > agent_inference (0.50)
     * Authority scoring: system/rules > user > doc > agent
     * Logarithmic Recency Prior: delta_days logarithmic bonus for explicit user rules
     * Superseded belief suppression: meta['is_superseded'] -> rank = -5.0
   - 0/1 Knapsack token budget packing.
   - Clean formatted context block generation ("## ATLAS Cognitive Context").
2. EpistemicWritePath:
   - Epistemic conflict arbitration via ConflictArbiter.
   - Belief revision:
     * Atomically supersedes old key in KV (metadata['is_superseded'] = True)
     * Wires (old)-[:SUPERSEDED_BY]->(new) in graph store
     * Purges old key from hot buffer
   - Same-key value revision:
     * Tracks prior_value
     * Increments revision_count
     * Wires version transitions in graph
   - VerifiedKVStore persistence (set_sync) with SHA-256 audit log.
   - Hot working buffer updates.
   - Multi-hop graph auto-insertion.
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from atlas_memory.cognitive.conflict_arbiter import ConflictArbiter
from atlas_memory.cognitive.hot_buffer import HotBufferPolicy, is_session_accessible
from atlas_memory.cognitive.models import RetrievalRecord
from atlas_memory.cognitive.retrieval_pipeline import RetrievalPipeline
from atlas_memory.cognitive.write_path import EpistemicWritePath
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.extensions.compactor import ContextCompactor
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.server.atlas_daemon import (
    AtlasDaemon,
    compute_content_fingerprint,
    get_record_authority_score,
    is_conversational_churn,
)


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

    # Continuation queries
    assert pipeline.is_continuation_query("Kontynuuj działanie") is True
    assert pipeline.is_continuation_query("dalej") is True
    assert pipeline.is_continuation_query("continue") is True
    assert pipeline.is_continuation_query("go on") is True
    assert pipeline.is_continuation_query("kolejny krok") is True
    assert pipeline.is_continuation_query("Jaka jest pogoda?") is False

    # Trivial prompt gating
    assert pipeline.is_trivial_prompt("cześć!") is True
    assert pipeline.is_trivial_prompt("dziękuję bardzo") is True
    assert pipeline.is_trivial_prompt("Zbuduj raport finansowy") is False

    # should_retrieve policy
    can_retrieve, entities, reason = pipeline.should_retrieve("cześć, co tam?")
    assert can_retrieve is False
    assert reason == "trivial_prompt"

    can_retrieve, entities, reason = pipeline.should_retrieve("Sprawdź konfigurację test:cluster:node1")
    assert can_retrieve is True
    assert "test:cluster:node1" in entities or any("cluster" in e for e in entities)


@pytest.mark.asyncio
async def test_retrieval_pipeline_hot_buffer_injection():
    """Verifies that empty or continuation queries inject active hot working memory."""
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

    # 1. Empty query -> injects hot buffer
    res_empty = await pipeline.retrieve(query="", session_id="sess_unit")
    assert res_empty.skipped is False
    assert res_empty.count == 2
    assert any(r.subject == "task:active_task" for r in res_empty.records)
    assert "## ATLAS Cognitive Context" in pipeline.generate_context_block(res_empty.records)

    # 2. Continuation query -> injects hot buffer
    res_cont = await pipeline.retrieve(query="Kontynuuj działanie", session_id="sess_unit")
    assert res_cont.skipped is False
    assert res_cont.count == 2
    assert any(r.subject == "task:step" for r in res_cont.records)

    # 3. Continuation query with empty hot buffer and force=False -> skipped
    hot_buffer.clear()
    res_empty_cont = await pipeline.retrieve(query="Kontynuuj działanie", session_id="sess_unit", force=False)
    assert res_empty_cont.skipped is True
    assert res_empty_cont.count == 0


def test_veracity_first_ranking_and_recency_prior():
    """
    Verifies:
    1. user_explicit > tool_output > external_doc > agent_inference
    2. Authority scoring: rules > user > doc > agent
    3. Logarithmic recency prior: fresh rule outranks stale rule
    4. Superseded belief suppression: meta['is_superseded'] = True -> rank = -5.0
    """
    pipeline = RetrievalPipeline()
    now = time.time()

    # 1. Veracity hierarchy (all same subject type and freshness)
    rec_user = RetrievalRecord(
        subject="fact:cache_ttl",
        predicate="is",
        object="300",
        confidence=1.0,
        source_type="user_explicit",
        timestamp=now,
    )
    rec_tool = RetrievalRecord(
        subject="fact:cache_ttl",
        predicate="is",
        object="300",
        confidence=1.0,
        source_type="tool_output",
        timestamp=now,
    )
    rec_doc = RetrievalRecord(
        subject="fact:cache_ttl",
        predicate="is",
        object="300",
        confidence=1.0,
        source_type="external_doc",
        timestamp=now,
    )
    rec_agent = RetrievalRecord(
        subject="fact:cache_ttl",
        predicate="is",
        object="300",
        confidence=1.0,
        source_type="agent_inference",
        timestamp=now,
    )

    rank_user = pipeline.compute_record_rank(rec_user, now=now)
    rank_tool = pipeline.compute_record_rank(rec_tool, now=now)
    rank_doc = pipeline.compute_record_rank(rec_doc, now=now)
    rank_agent = pipeline.compute_record_rank(rec_agent, now=now)

    assert rank_user > rank_tool > rank_doc > rank_agent

    # 2. Authority scoring: rule > user fact > doc fact > agent inference
    rec_rule = RetrievalRecord(
        subject="rule:git:commit_format",
        predicate="must_be",
        object="conventional_commits",
        confidence=1.0,
        source_type="user_explicit",
        timestamp=now,
    )
    assert pipeline.get_authority_score(rec_rule.subject) == 1.10
    assert pipeline.get_authority_score("hermes:user_md:preference") == 1.00
    assert pipeline.get_authority_score("fact:readme_doc") == 0.70
    assert pipeline.get_authority_score("working_memory:temp") == 0.50

    # 3. Logarithmic recency prior:
    # delta_days = max(0.0, (now - rec_ts) / 86400.0)
    # rule_bonus = 1.0 / (1.0 + 0.15 * math.log1p(delta_days))
    rec_rule_fresh = RetrievalRecord(
        subject="rule:git:strategy",
        predicate="use",
        object="trunk_based",
        confidence=1.0,
        source_type="user_explicit",
        timestamp=now,  # delta_days = 0.0 -> rule_bonus = 1.0
    )
    rec_rule_old = RetrievalRecord(
        subject="rule:git:strategy",
        predicate="use",
        object="git_flow",
        confidence=1.0,
        source_type="user_explicit",
        timestamp=now - 30 * 86400.0,  # 30 days old
    )

    rank_fresh = pipeline.compute_record_rank(rec_rule_fresh, now=now)
    rank_old = pipeline.compute_record_rank(rec_rule_old, now=now)
    assert rank_fresh > rank_old

    # Exact rule bonus verification
    delta_days = 30.0
    expected_old_bonus = 1.0 / (1.0 + 0.15 * math.log1p(delta_days))
    assert math.isclose(rank_fresh - rank_old, 1.0 - expected_old_bonus, rel_tol=1e-3)

    # 4. Superseded belief suppression: rank must be -5.0
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
    rank_sup = pipeline.compute_record_rank(rec_superseded, now=now)
    assert rank_sup == -5.0


def test_knapsack_token_budget_packing():
    """Verifies 0/1 Knapsack respects token budget and suppresses superseded items."""
    pipeline = RetrievalPipeline()
    now = time.time()

    records = [
        RetrievalRecord(
            subject=f"fact:key_{i}",
            predicate="value",
            object="data_" * 20,  # roughly 30 tokens each
            confidence=1.0,
            source_type="user_explicit",
            timestamp=now,
        )
        for i in range(10)
    ]
    # Add a superseded item that should be suppressed
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

    # Budget for roughly 3 records
    token_cost_single = pipeline.estimate_tokens(records[0])
    budget = token_cost_single * 3 + 5

    packed = pipeline.pack_knapsack_01(records, budget=budget, now=now)
    total_tokens = sum(pipeline.estimate_tokens(r) for r in packed)

    assert len(packed) <= 4
    assert total_tokens <= budget
    assert not any(r.is_superseded for r in packed)
    assert not any(r.subject == "rule:superseded_rule" for r in packed)


@pytest.mark.asyncio
async def test_epistemic_write_path_belief_revision(tmp_path: Path):
    """
    Verifies EpistemicWritePath:
    1. Epistemic conflict arbitration.
    2. Belief revision:
       - Atomically supersedes old key in KV (metadata['is_superseded'] = True)
       - Wires (old)-[:SUPERSEDED_BY]->(new) in graph store
       - Purges old key from hot buffer
    """
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

    # 1. Write initial established belief
    res1 = write_path.write(
        key="rule:deployment:strategy",
        value="Wdrażamy wyłącznie przez Docker Swarm",
        confidence=1.0,
        metadata={"source_type": "user_explicit"},
    )
    assert res1["status"] == "ok"
    assert res1["persisted"] is True
    assert len(hot_buffer) == 1

    # 2. User explicitly updates belief to a new key
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

    # Verify old key was marked superseded in KV store
    old_item = engine.kv.get_sync("rule:deployment:strategy")
    assert old_item is not None
    assert old_item["metadata"]["is_superseded"] is True
    assert old_item["metadata"]["superseded_by"] == "rule:deployment:kubernetes"

    # Verify graph edge (old)-[:SUPERSEDED_BY]->(new) was wired
    superseded_edges = [
        rel for rel in mock_graph.relations
        if rel["predicate"] == "SUPERSEDED_BY"
    ]
    assert len(superseded_edges) >= 1
    assert superseded_edges[0]["subject"] == "rule:deployment:strategy"
    assert superseded_edges[0]["object"] == "rule:deployment:kubernetes"

    # Verify old key was purged from hot working buffer
    active_hot_keys = [e.key for e in hot_buffer.get_active()]
    assert "rule:deployment:strategy" not in active_hot_keys
    assert "rule:deployment:kubernetes" in active_hot_keys


@pytest.mark.asyncio
async def test_epistemic_write_path_same_key_version_revision(tmp_path: Path):
    """
    Verifies same-key value revision:
    - Tracks prior_value
    - Increments revision_count
    - Wires version transition in graph store
    """
    db_path = str(tmp_path / "test_wp_rev.db")
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

    # Initial write v0
    key = "config:system:max_threads"
    res1 = write_path.write(key=key, value="8", confidence=1.0)
    assert res1["status"] == "ok"
    assert res1["revision_count"] == 0

    # Revision 1 (same key, new value "16")
    res2 = write_path.write(key=key, value="16", confidence=1.0)
    assert res2["status"] == "ok"
    assert res2["revision_count"] == 1

    stored_v1 = engine.kv.get_sync(key)
    assert stored_v1["value"] == "16"
    assert stored_v1["metadata"]["prior_value"] == "8"
    assert stored_v1["metadata"]["revision_count"] == 1

    # Revision 2 (same key, new value "32")
    res3 = write_path.write(key=key, value="32", confidence=1.0)
    assert res3["status"] == "ok"
    assert res3["revision_count"] == 2

    stored_v2 = engine.kv.get_sync(key)
    assert stored_v2["value"] == "32"
    assert stored_v2["metadata"]["prior_value"] == "16"
    assert stored_v2["metadata"]["revision_count"] == 2

    # Check graph transitions
    rev_edges = [
        rel for rel in mock_graph.relations
        if rel["predicate"] in ("REVISED_FROM", "TRANSITIONED_TO")
    ]
    assert len(rev_edges) >= 4


@pytest.mark.asyncio
async def test_epistemic_write_path_multihop_graph_edges(tmp_path: Path):
    """Verifies multi-hop arrow syntax and dict auto-wiring into graph store."""
    db_path = str(tmp_path / "test_wp_edges.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    mock_graph = MockGraphStore()
    write_path = EpistemicWritePath(kv_store=engine.kv, graph_store=mock_graph)

    # Arrow chain: NodeA -> NodeB -> NodeC
    res_chain = write_path.write(
        key="pipeline:flow",
        value="NodeA -> NodeB -> NodeC",
        confidence=0.9,
    )
    assert res_chain["persisted"] is True

    leads_to_edges = [rel for rel in mock_graph.relations if rel["predicate"] == "leads_to"]
    assert len(leads_to_edges) == 3
    assert leads_to_edges[0]["subject"] == "pipeline:flow"
    assert leads_to_edges[0]["object"] == "NodeA"
    assert leads_to_edges[1]["subject"] == "NodeA"
    assert leads_to_edges[1]["object"] == "NodeB"
    assert leads_to_edges[2]["subject"] == "NodeB"
    assert leads_to_edges[2]["object"] == "NodeC"


@pytest.mark.asyncio
async def test_chit_chat_gate_polish_stop_words_and_greetings():
    """Verify Chit-Chat gate closes for English and Polish greetings & gratitude, opens for queries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sock_path = Path(tmpdir) / "test_atlas.sock"
        pid_path = Path(tmpdir) / "test_atlas.pid"
        db_path = Path(tmpdir) / "test_atlas.db"

        engine = HybridMemoryEngine.create_default(db_path=str(db_path))
        daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path, engine=engine)

        await daemon._handle_set({
            "key": "test:greeting",
            "value": "Cześć i dziękuję serdecznie za pomoc!",
            "confidence": 1.0,
            "source_type": "user_explicit",
        })

        chit_chat_queries = ["hello", "thanks", "dziękuję", "cześć", "czesc", "dzięki", "siema", "witaj", "ok", "do widzenia"]
        for q in chit_chat_queries:
            res = await daemon._handle_prefetch({"query": q})
            assert res.get("skipped") is True, f"Query '{q}' should be skipped by chit-chat gate"
            assert len(res.get("records", [])) == 0, f"Query '{q}' returned records"
            assert res.get("context_block") == ""

        res_valid = await daemon._handle_prefetch({"query": "ATLAS Memory architecture"})
        assert res_valid.get("skipped") is not True


@pytest.mark.asyncio
async def test_session_isolation_secrets_never_leak():
    """Verify session_a secrets are invisible to session_b and anonymous sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sock_path = Path(tmpdir) / "test_atlas.sock"
        pid_path = Path(tmpdir) / "test_atlas.pid"
        db_path = Path(tmpdir) / "test_atlas.db"

        engine = HybridMemoryEngine.create_default(db_path=str(db_path))
        daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path, engine=engine)

        await daemon._handle_set({
            "key": "session_a_secret",
            "value": "TOP_SECRET_ALPHA_42",
            "session_id": "session_a",
            "confidence": 1.0,
        })
        await daemon._handle_set({
            "key": "session_b_secret",
            "value": "TOP_SECRET_BETA_99",
            "session_id": "session_b",
            "confidence": 1.0,
        })
        await daemon._handle_set({
            "key": "global_config_secret",
            "value": "GLOBAL_SHARED_INFO",
            "confidence": 1.0,
        })

        res_a = await daemon._handle_prefetch({"query": "secret", "session_id": "session_a"})
        subjects_a = [r["subject"] for r in res_a["records"]]
        assert "session_a_secret" in subjects_a
        assert "session_b_secret" not in subjects_a, "session_b_secret leaked to session_a!"

        res_b = await daemon._handle_prefetch({"query": "secret", "session_id": "session_b"})
        subjects_b = [r["subject"] for r in res_b["records"]]
        assert "session_b_secret" in subjects_b
        assert "session_a_secret" not in subjects_b, "session_a_secret leaked to session_b!"

        res_anon = await daemon._handle_prefetch({"query": "secret", "session_id": ""})
        subjects_anon = [r["subject"] for r in res_anon["records"]]
        assert "session_a_secret" not in subjects_anon, "session_a_secret leaked to anonymous!"
        assert "session_b_secret" not in subjects_anon, "session_b_secret leaked to anonymous!"


@pytest.mark.asyncio
async def test_user_explicit_rule_persistence_over_noise():
    """Verify high-confidence user_explicit rule persists and tops the ranking above low-confidence noise."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sock_path = Path(tmpdir) / "test_atlas.sock"
        pid_path = Path(tmpdir) / "test_atlas.pid"
        db_path = Path(tmpdir) / "test_atlas.db"

        engine = HybridMemoryEngine.create_default(db_path=str(db_path))
        daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path, engine=engine)

        await daemon._handle_set({
            "key": "user:rule:persistent",
            "value": "RAPORTY ZAWSZE 400 LINII I BEZ ZBEDNYCH WSTEPÓW",
            "confidence": 1.0,
            "source_type": "user_explicit",
        })

        for i in range(10):
            await daemon._handle_set({
                "key": f"noise:turn:record_{i}",
                "value": f"raporty linii dyskusja {i}",
                "confidence": 0.3,
                "source_type": "agent_inference",
            })

        res = await daemon._handle_prefetch({"query": "RAPORTY ZAWSZE 400 LINII", "limit": 15})
        records = res.get("records", [])

        assert len(records) >= 1
        assert records[0]["subject"] == "user:rule:persistent"
        assert records[0]["confidence"] == 1.0
        assert records[0]["source_type"] == "user_explicit"


@pytest.mark.asyncio
async def test_semantic_sanity_nonsense_query_returns_zero():
    """Verify random nonsense query triggers relevance cutoff and does not dump random records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sock_path = Path(tmpdir) / "test_atlas.sock"
        pid_path = Path(tmpdir) / "test_atlas.pid"
        db_path = Path(tmpdir) / "test_atlas.db"

        engine = HybridMemoryEngine.create_default(db_path=str(db_path))
        daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path, engine=engine)

        await daemon._handle_set({
            "key": "peptide:ghk_cu",
            "value": "GHK-Cu dawkowanie peptydy i badania regeneracji tkankowej",
            "confidence": 0.95,
            "source_type": "external_doc",
        })
        await daemon._handle_set({
            "key": "system:obsidian_mcp",
            "value": "Obsidian MCP integracja notatek wiedzy",
            "confidence": 0.90,
            "source_type": "user_explicit",
        })

        res_real = await daemon._handle_prefetch({"query": "GHK-Cu dawkowanie peptydy"})
        assert len(res_real.get("records", [])) >= 1
        subjects_real = [r["subject"] for r in res_real["records"]]
        assert any("peptide" in s.lower() or "ghk" in s.lower() for s in subjects_real)

        res_nonsense = await daemon._handle_prefetch({"query": "kompletnie losowy nonsens xyz123"})
        assert len(res_nonsense.get("records", [])) == 0, f"Expected 0 records for nonsense query, got: {res_nonsense['records']}"


def test_content_fingerprint_normalization():
    """Verify that identical facts with varied punctuation or prefixes yield identical fingerprints."""
    t1 = "Obsidian MCP: UŻYWAĆ natywnych narzędzi mcp__obsidian__vault_write/vault_read/vault_patch."
    t2 = "Obsidian MCP: UŻYWAĆ natywnych narzędzi `mcp__obsidian__vault_write` (mcp-obsidian plugin)"
    t3 = "User prefers ONE consolidated report file (explicitly rejected multi-file output:stated_memory)"
    t4 = "User prefers ONE consolidated report file (explicitly rejected multi-file output)"

    fp1 = compute_content_fingerprint(t1)
    fp2 = compute_content_fingerprint(t2)
    fp3 = compute_content_fingerprint(t3)
    fp4 = compute_content_fingerprint(t4)

    assert fp1.startswith("używać natywnych")
    assert fp2.startswith("używać natywnych")
    assert fp3 == fp4


def test_record_authority_scoring():
    """Verify authority hierarchy where markdown and native state outrank legacy mnemosyne working memory."""
    auth_user_md = get_record_authority_score("hermes:user_md:srodowisko")
    auth_memory_md = get_record_authority_score("hermes:memory_md:obsidian_mcp")
    auth_native_kv = get_record_authority_score("database_cluster_status")
    auth_fact = get_record_authority_score("fact:obsidian_mcp:stated_memory")
    auth_canon = get_record_authority_score("mnemosyne:canonical_facts:peptides")
    auth_working = get_record_authority_score("mnemosyne:mnemosyne_working_memory:prompt_1")

    assert auth_user_md == 100
    assert auth_memory_md == 100
    assert auth_native_kv == 90
    assert auth_fact == 80
    assert auth_canon == 70
    assert auth_working == 30

    assert auth_user_md > auth_native_kv > auth_fact > auth_canon > auth_working


def test_conversational_churn_detection():
    """Verify detection of raw conversational noise and prompt dumps."""
    assert is_conversational_churn("[USER] Test 1: Test Aktywnego Użycia Narzędzi", "Sample text")
    assert is_conversational_churn("turn:fact", "[ASSISTANT] Odpowiedź modelu na prompt")
    assert is_conversational_churn("error_key", "Traceback (most recent call last):\n  File 'test.py'")
    assert is_conversational_churn("order", "MISSION ORDER: Execute audit")
    assert not is_conversational_churn("hermes:memory_md:Obsidian MCP", "UŻYWAĆ natywnych narzędzi")


@pytest.mark.asyncio
async def test_prefetch_content_deduplication(tmp_path):
    """Test that prefetch collapses 3 identical facts under different keys into 1 winning record."""
    daemon = AtlasDaemon(socket_path=tmp_path / "daemon.sock", pid_path=tmp_path / "daemon.pid")

    mock_kv = MagicMock()
    mock_kv.get_all_sync.return_value = {
        "hermes:memory_md:Obsidian MCP": {
            "value": "Obsidian MCP: UŻYWAĆ natywnych narzędzi mcp__obsidian__vault_write/vault_read/vault_patch.",
            "confidence": 1.0,
            "metadata": {"source_type": "user_explicit"},
        },
        "fact:obsidian_mcp:stated_memory": {
            "value": "Obsidian MCP: UŻYWAĆ natywnych narzędzi mcp__obsidian__vault_write/vault_read/vault_patch.",
            "confidence": 0.95,
            "metadata": {"source_type": "tool_output"},
        },
        "mnemosyne:mnemosyne_working_memory:Obsidian MCP:working_fact": {
            "value": "Obsidian MCP: UŻYWAĆ natywnych narzędzi mcp__obsidian__vault_write/vault_read/vault_patch.",
            "confidence": 0.70,
            "metadata": {"source_type": "agent_inference"},
        },
        "mnemosyne:mnemosyne_working_memory:[USER] Raw Prompt": {
            "value": "[USER] Chciałbym, abyś przygotował nową wersję",
            "confidence": 0.90,
            "metadata": {"source_type": "user_explicit"},
        },
    }

    mock_engine = MagicMock()
    mock_engine.kv = mock_kv
    daemon.engine = mock_engine

    res = await daemon._handle_prefetch({"query": "Obsidian MCP", "force": True})
    records = res["records"]

    assert len(records) == 1
    assert records[0]["subject"] == "hermes:memory_md:Obsidian MCP"
    assert all("[USER]" not in r["subject"] for r in records)


@pytest.mark.asyncio
async def test_f4_hot_kv_buffer_prefetch_visibility(tmp_path: Path):
    """Finding F-4: Hot-KV working buffer injects recent writes even on continuation queries."""
    db_path = str(tmp_path / "test_atlas_f4.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    daemon = AtlasDaemon(engine=engine)

    await daemon._handle_set({
        "key": "session:working_plan",
        "value": "Refactor phase 2 in progress",
        "confidence": 1.0,
        "session_id": "sess_f4",
    })

    prefetch_res = await daemon._handle_prefetch({
        "query": "Kontynuuj działanie",
        "session_id": "sess_f4",
    })

    assert prefetch_res.get("skipped") is not True
    assert prefetch_res["count"] >= 1
    assert any("Refactor phase 2" in r["object"] for r in prefetch_res["records"])
    assert "## ATLAS Cognitive Context" in prefetch_res["context_block"]


@pytest.mark.asyncio
async def test_f4_hot_kv_buffer_startup_seeding(tmp_path: Path):
    """Finding F-4 (AGI-6): _hot_kv_buffer is seeded from SQLite on startup to prevent cold working memory."""
    db_path = str(tmp_path / "test_atlas.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )

    engine.kv.set_sync(
        key="state:task_status",
        value="Wykonywanie refaktoryzacji V45",
        confidence=1.0,
        metadata={"source_type": "user_explicit", "session_id": "session_test_45"},
    )

    daemon = AtlasDaemon(engine=engine)
    assert len(daemon._hot_kv_buffer) >= 1
    hot_keys = [item["key"] for item in daemon._hot_kv_buffer]
    assert "state:task_status" in hot_keys

    prefetch_res = await daemon._handle_prefetch({
        "query": "Kontynuuj działanie",
        "session_id": "session_test_45",
    })
    assert prefetch_res.get("skipped") is not True
    assert prefetch_res.get("count", 0) >= 1
    assert "state:task_status" in prefetch_res["context_block"]


@pytest.mark.asyncio
async def test_atlas_recall_continuation_hot_kv_injection(tmp_path: Path):
    """Verifies prefetch injects active hot-KV working memory for continuation queries even with force=True."""
    db_path = str(tmp_path / "test_atlas.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    daemon = AtlasDaemon(engine=engine)

    now = time.time()
    daemon._hot_kv_buffer.append({
        "key": "task:pipeline:step_3",
        "value": "Wykonywanie testów integracyjnych w toku",
        "timestamp": now,
        "confidence": 1.0,
        "source_type": "user_explicit",
        "metadata": {"session_id": "sess_v47"},
    })
    daemon._hot_kv_buffer.append({
        "key": "task:pipeline:state",
        "value": "Pipeline aktywny, oczekiwanie na decyzję",
        "timestamp": now,
        "confidence": 1.0,
        "source_type": "user_explicit",
        "metadata": {"session_id": "sess_v47"},
    })

    res = await daemon._handle_prefetch({
        "query": "Kontynuuj działanie",
        "session_id": "sess_v47",
        "force": True,
    })

    assert res.get("skipped") is not True
    assert res["count"] >= 2
    subjects = [r["subject"] for r in res["records"]]
    assert "task:pipeline:step_3" in subjects
    assert "task:pipeline:state" in subjects
    for r in res["records"]:
        assert r["predicate"] == "active_working_memory"

    res_empty = await daemon._handle_prefetch({
        "query": "",
        "session_id": "sess_v47",
    })
    assert res_empty["count"] >= 2
    assert "task:pipeline:step_3" in [r["subject"] for r in res_empty["records"]]


@pytest.mark.asyncio
async def test_veracity_ranking_user_rule_dominance(tmp_path: Path):
    """Verifies user explicit rules outrank background facts in prefetch ranking."""
    db_path = str(tmp_path / "test_atlas.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    daemon = AtlasDaemon(engine=engine)

    await daemon._handle_set({
        "key": "fact:devops_pipeline",
        "value": "Strategia git i wdrażanie kodu w środowisku CI/CD zautomatyzowane",
        "confidence": 0.85,
        "metadata": {"source_type": "tool_output", "vector_score": 0.92},
    })

    await daemon._handle_set({
        "key": "rule:git:branch_strategy",
        "value": "Wdrażanie kodu WYŁĄCZNIE przez Pull Requesty (zakaz bezpośredniego push do main)",
        "confidence": 1.0,
        "metadata": {"source_type": "user_explicit"},
    })

    res = await daemon._handle_prefetch({
        "query": "Jak wdrażać kod i jaka jest strategia git?",
        "force": True,
    })

    assert len(res["records"]) >= 2
    assert res["records"][0]["subject"] == "rule:git:branch_strategy"


@pytest.mark.asyncio
async def test_h5_continuation_prefetch_working_buffer_session_match(tmp_path: Path):
    """
    H5: Verifies 'Kontynuuj działanie' prefetch injects working memory for matching session_id,
    preserves session isolation, and respects both force=False and force=True.
    """
    db_path = str(tmp_path / "test_atlas.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    daemon = AtlasDaemon(engine=engine)

    now = time.time()
    daemon._hot_kv_buffer.append({
        "key": "state:active_goal",
        "value": "Dokończyć strangler fig L0-L3 dla prefetch",
        "timestamp": now,
        "confidence": 1.0,
        "source_type": "user_explicit",
        "metadata": {"session_id": "session_worker_h5"},
    })
    daemon._hot_kv_buffer.append({
        "key": "state:other_session_goal",
        "value": "Zadanie z innej sesji roboczej",
        "timestamp": now,
        "confidence": 1.0,
        "source_type": "user_explicit",
        "metadata": {"session_id": "session_beta_isolated"},
    })

    res_default_force = await daemon._handle_prefetch({
        "query": "Kontynuuj działanie",
        "session_id": "session_worker_h5",
        "force": False,
    })

    assert res_default_force.get("skipped") is not True
    assert res_default_force["count"] == 1
    record = res_default_force["records"][0]
    assert record["subject"] == "state:active_goal"
    assert "Dokończyć strangler fig" in record["object"]
    assert record["predicate"] == "active_working_memory"
    assert "## ATLAS Cognitive Context" in res_default_force["context_block"]
    assert "state:active_goal" in res_default_force["context_block"]
    assert "state:other_session_goal" not in res_default_force["context_block"]

    res_force_true = await daemon._handle_prefetch({
        "query": "Kontynuuj działanie",
        "session_id": "session_worker_h5",
        "force": True,
    })
    assert res_force_true.get("skipped") is not True
    assert res_force_true["count"] == 1
    assert res_force_true["records"][0]["subject"] == "state:active_goal"

    res_beta = await daemon._handle_prefetch({
        "query": "Kontynuuj działanie",
        "session_id": "session_beta_isolated",
        "force": False,
    })
    assert res_beta.get("skipped") is not True
    assert res_beta["count"] == 1
    assert res_beta["records"][0]["subject"] == "state:other_session_goal"
    assert "state:active_goal" not in res_beta["context_block"]

    res_empty_sess = await daemon._handle_prefetch({
        "query": "Kontynuuj działanie",
        "session_id": "session_nonexistent",
        "force": False,
    })
    assert res_empty_sess.get("skipped") is True
    assert res_empty_sess.get("reason") == "continuation_prompt"
    assert res_empty_sess["count"] == 0

    res_direct = await daemon.retrieval_pipeline.execute_prefetch(
        {
            "query": "Kontynuuj działanie",
            "session_id": "session_worker_h5",
            "force": False,
        },
        hot_buffer=daemon._hot_kv_buffer,
    )
    assert res_direct.get("skipped") is not True
    assert res_direct["count"] == 1
    assert res_direct["records"][0]["subject"] == "state:active_goal"


@pytest.mark.asyncio
async def test_f1_continuation_respects_limit_and_budget(tmp_path: Path):
    """
    F1: Verifies execute_prefetch clamps limit and applies token budget
    on continuation ('Kontynuuj działanie') and empty queries without dumping unbudgeted 25 items.
    """
    hot_buffer = HotBufferPolicy(maxlen=25, ttl_seconds=900.0)
    pipeline = RetrievalPipeline(hot_buffer=hot_buffer, default_token_budget=1500)

    now = time.time()
    # Seed hot buffer with 20 items for session "sess_f1"
    for i in range(20):
        hot_buffer.append({
            "key": f"task:step_{i:02d}",
            "value": f"Wykonaj krok numer {i} procedury hardeningowej systemu ATLAS",
            "timestamp": now,
            "confidence": 1.0,
            "source_type": "user_explicit",
            "metadata": {"session_id": "sess_f1"},
        })

    # 1. Continuation query with limit=5 (out of 20 seeded)
    res_cont = await pipeline.execute_prefetch({
        "query": "Kontynuuj działanie",
        "session_id": "sess_f1",
        "limit": 5,
    })
    assert res_cont.get("skipped") is not True
    assert res_cont["count"] == 5
    assert len(res_cont["records"]) == 5
    bullet_lines = [ln for ln in res_cont["context_block"].splitlines() if ln.startswith("•")]
    assert len(bullet_lines) == 5

    # 2. Empty query with limit=3
    res_empty = await pipeline.execute_prefetch({
        "query": "",
        "session_id": "sess_f1",
        "limit": 3,
    })
    assert res_empty.get("skipped") is not True
    assert res_empty["count"] == 3
    assert len(res_empty["records"]) == 3
    empty_bullets = [ln for ln in res_empty["context_block"].splitlines() if ln.startswith("•")]
    assert len(empty_bullets) == 3

    # 3. Token budget truncation
    res_budget = await pipeline.execute_prefetch({
        "query": "Kontynuuj",
        "session_id": "sess_f1",
        "limit": 10,
        "max_tokens": 45,
    })
    assert res_budget.get("skipped") is not True
    assert res_budget["count"] <= 2
    assert len(res_budget["records"]) == res_budget["count"]
    budget_bullets = [ln for ln in res_budget["context_block"].splitlines() if ln.startswith("•")]
    assert len(budget_bullets) == res_budget["count"]


def test_f2_session_isolation_strict():
    """
    F2: Verifies strict session isolation in is_session_accessible:
    hermes_default is NOT accessible to foreign sessions (e.g. obca_sesja_xyz).
    Only explicit 'global' or empty effective session is globally accessible.
    """
    # Foreign session accessing hermes_default -> MUST BE REJECTED
    assert not is_session_accessible("hermes_default", "obca_sesja_xyz")
    assert not is_session_accessible("default", "obca_sesja_xyz")

    # Matching session -> MUST BE ALLOWED
    assert is_session_accessible("hermes_default", "hermes_default")
    assert is_session_accessible("default", "default")
    assert is_session_accessible("session_alpha", "session_alpha")
    assert is_session_accessible("session_alpha", "alpha")
    assert is_session_accessible("alpha", "session_alpha")

    # Foreign session accessing another session -> MUST BE REJECTED
    assert not is_session_accessible("session_alpha", "session_beta")

    # Global sessions -> MUST BE ACCESSIBLE TO ALL
    assert is_session_accessible("global", "obca_sesja_xyz")
    # Empty or None effective session belongs to default environment -> REJECT foreign session
    assert not is_session_accessible(None, "obca_sesja_xyz")
    assert not is_session_accessible("", "obca_sesja_xyz")
    assert is_session_accessible(None, "hermes_default")
    assert is_session_accessible("", "default")


@pytest.mark.asyncio
async def test_f2_session_isolation_prefetch_integration(tmp_path: Path):
    """
    F2 Integration: Verifies prefetch does not leak hermes_default records to foreign sessions.
    """
    hot_buffer = HotBufferPolicy(maxlen=25, ttl_seconds=900.0)
    pipeline = RetrievalPipeline(hot_buffer=hot_buffer)
    now = time.time()

    hot_buffer.append({
        "key": "secret:token",
        "value": "super_secret_hermes_token",
        "timestamp": now,
        "metadata": {"session_id": "hermes_default"},
    })
    hot_buffer.append({
        "key": "global:announcement",
        "value": "System maintenance at 02:00",
        "timestamp": now,
        "metadata": {"session_id": "global"},
    })

    # Foreign session queries
    res_foreign = await pipeline.execute_prefetch({
        "query": "Kontynuuj",
        "session_id": "obca_sesja_xyz",
    })
    subjects_foreign = [r["subject"] for r in res_foreign.get("records", [])]
    assert "secret:token" not in subjects_foreign
    assert "global:announcement" in subjects_foreign

    # Hermes default session queries
    res_hermes = await pipeline.execute_prefetch({
        "query": "Kontynuuj",
        "session_id": "hermes_default",
    })
    subjects_hermes = [r["subject"] for r in res_hermes.get("records", [])]
    assert "secret:token" in subjects_hermes
    assert "global:announcement" in subjects_hermes


def test_f3_no_self_superseded_by_edge():
    """
    F3: Verifies that SUPERSEDED_BY self-edges (key)-[:SUPERSEDED_BY]->(key)
    are strictly prevented when old_key == key.
    """
    recorded_edges: List[tuple[str, str, str]] = []

    class MockGraph:
        def add_relation(self, sub: str, rel: str, obj: str, confidence: float = 1.0) -> None:
            recorded_edges.append((sub, rel, obj))

    write_path = EpistemicWritePath(graph_store=MockGraph())

    arbiter_mock = MagicMock()
    arbiter_mock.arbitrate.return_value = MagicMock(
        is_conflict=False,
        is_belief_revision=True,
        superseded_key="same_key",
        message="Self revision",
    )
    write_path.conflict_arbiter = arbiter_mock

    write_path.write("same_key", "new_value")

    for sub, rel, obj in recorded_edges:
        if rel == "SUPERSEDED_BY":
            assert sub != obj, f"Forbidden self-loop detected: ({sub})-[:SUPERSEDED_BY]->({obj})"


def test_f3_no_prose_node_on_colon():
    """
    F3: Verifies that prose sentences containing colons do NOT create
    giant sentence target nodes with relates_to relations.
    Only concise identifiers or test:* keys create relates_to.
    """
    recorded_edges: List[tuple[str, str, str]] = []

    class MockGraph:
        def add_relation(self, sub: str, rel: str, obj: str, confidence: float = 1.0) -> None:
            recorded_edges.append((sub, rel, obj))

    write_path = EpistemicWritePath(graph_store=MockGraph())

    # 1. Prose sentence with colon
    prose = "Uwaga: proces wymaga restartu usługi po zmianie konfiguracji produkcyjnej."
    write_path.write("notice:sys_alert", prose)

    # Must NOT add relates_to edge pointing to the entire prose sentence
    relates_to_prose = [
        edge for edge in recorded_edges
        if edge[1] == "relates_to" and edge[2] == prose
    ]
    assert len(relates_to_prose) == 0

    # 2. Concise key identifier with colon
    concise_key = "config:port:8080"
    write_path.write("service:gateway", concise_key)
    assert ("service:gateway", "relates_to", concise_key) in recorded_edges

    # 3. test:* prefix
    test_key = "test:hop:target_node"
    write_path.write("probe:source", test_key)
    assert ("probe:source", "relates_to", test_key) in recorded_edges


def test_compactor_initialization():
    """Weryfikuje poprawne wstrzykiwanie lub domyślną inicjalizację ContextCompactor."""
    custom_compactor = ContextCompactor(session_window_size=4)
    pipeline = RetrievalPipeline(compactor=custom_compactor)
    assert pipeline.compactor is custom_compactor
    assert pipeline.compactor.window_size == 4

    default_pipeline = RetrievalPipeline()
    assert default_pipeline.compactor is not None
    assert isinstance(default_pipeline.compactor, ContextCompactor)


def test_compactor_knapsack_aggressive_fitting():
    """Weryfikuje, że kompakcja zagęszcza długie fakty i pozwala zmieścić więcej rekordów w małym budżecie."""
    pipeline = RetrievalPipeline()

    long_text = "Bardzo szczegółowy opis architektury systemowej zawierający wielokrotne powtórzenia faktów technicznych, specyfikacji protokołów sieciowych oraz analizę wpływu parametrów bufora na opóźnienia komunikacji UDS IPC w milisekundach."
    assert len(long_text) > 150

    records = [
        RetrievalRecord(
            subject="rule:first",
            predicate="value",
            object="Zasada numer jeden: deterministyczne przetwarzanie",
            confidence=1.0,
            source_type="user_explicit",
            score=10.0,
        ),
        RetrievalRecord(
            subject="system:doc_high",
            predicate="value",
            object=long_text,
            confidence=0.85,
            source_type="tool_output",
            score=5.0,
        ),
        RetrievalRecord(
            subject="system:doc_medium",
            predicate="value",
            object=long_text,
            confidence=0.75,
            source_type="tool_output",
            score=4.0,
        ),
        RetrievalRecord(
            subject="agent:inference_low",
            predicate="value",
            object="Krótki domysł agenta",
            confidence=0.50,
            source_type="agent_inference",
            score=2.0,
        ),
    ]

    budget = 90
    raw_tokens = sum(pipeline.estimate_tokens(r) for r in records)
    assert raw_tokens > budget * 1.5

    packed = pipeline.pack_knapsack_01(records, budget=budget)

    assert len(packed) >= 2
    compacted_found = False
    for r in packed:
        if "... [compacted]" in str(r.object):
            compacted_found = True
            assert len(str(r.object)) <= 80 + len("... [compacted]")

    assert compacted_found is True
    total_packed_cost = sum(pipeline.estimate_tokens(r) for r in packed)
    assert total_packed_cost <= budget


@pytest.mark.asyncio
async def test_h6_session_isolation_in_graph_expansion():
    """
    H6 (F7): Verifies that relations stored under session 's1' are NEVER returned
    during prefetch in session 's2', while accessible to session 's1'.
    """
    engine = HybridMemoryEngine.create_default(db_path=":memory:", qdrant_location=":memory:")

    engine.graph.add_relation(
        subject="apollo",
        predicate="uses_database",
        object="postgresql_s1_secret",
        confidence=0.95,
        session_id="s1",
    )

    engine.graph.add_relation(
        subject="artemis",
        predicate="uses_cache",
        object="redis_cluster_s2",
        confidence=0.90,
        session_id="s2",
    )

    pipeline = RetrievalPipeline(engine=engine, graph_store=engine.graph)

    # 1. Prefetch z kontekstem zawierającym "apollo" i "artemis" dla sesji s2
    res_s2 = await pipeline.prefetch_for_context("apollo and artemis services", session_id="s2")
    objects_s2 = [r["object"] for r in res_s2]

    assert "postgresql_s1_secret" not in objects_s2
    assert "redis_cluster_s2" in objects_s2

    # 2. Prefetch dla sesji s1
    res_s1 = await pipeline.prefetch_for_context("apollo and artemis services", session_id="s1")
    objects_s1 = [r["object"] for r in res_s1]

    assert "postgresql_s1_secret" in objects_s1
    assert "redis_cluster_s2" not in objects_s1


@pytest.mark.asyncio
async def test_t1_prefetch_truncates_huge_payload_and_caps_context():
    """
    T1: Verifies that huge object payloads (>1200 chars) are truncated in formatted context
    and that total context_block length is capped at max(4000, token_budget * 4).
    """
    pipeline = RetrievalPipeline()
    huge_text = "A" * 6000
    huge_record = {
        "subject": "huge:payload",
        "predicate": "active_working_memory",
        "object": huge_text,
        "importance_score": 1.0,
        "confidence": 1.0,
        "source_type": "user_explicit",
    }

    cost = pipeline.estimate_tokens(huge_record)
    assert cost < 500

    hot = HotBufferPolicy(maxlen=10, ttl_seconds=900.0)
    hot.append({
        "key": "test:huge",
        "value": huge_text,
        "confidence": 1.0,
        "timestamp": time.time(),
        "metadata": {"session_id": "hermes_default"},
    })
    pipe_with_hot = RetrievalPipeline(hot_buffer=hot)
    res = await pipe_with_hot.execute_prefetch({
        "query": "",
        "session_id": "hermes_default",
        "token_budget": 500,
    })

    assert res["count"] == 1
    ctx = res["context_block"]
    assert "...[truncated]" in ctx
    assert len(ctx) <= max(4000, 500 * 4)


@pytest.mark.asyncio
async def test_t2_prefetch_suppresses_superseded_beliefs(tmp_path: Path):
    """
    T2: Verifies that superseded beliefs (in hot buffer, KV store, and vector results)
    are strictly suppressed and not injected into context.
    """
    db_file = tmp_path / "test_superseded.db"
    kv = VerifiedKVStore(db_path=str(db_file))
    now = time.time()

    await kv.set_state("pref:active", "use_python_314", confidence=1.0)
    await kv.set_state("pref:old", "use_python_27", confidence=1.0, metadata={"is_superseded": True})

    hot = HotBufferPolicy(maxlen=10, ttl_seconds=900.0)
    hot.append({
        "key": "hot:superseded",
        "value": "outdated_hot_fact",
        "confidence": 1.0,
        "timestamp": now,
        "is_superseded": True,
        "metadata": {"superseded": True},
    })
    hot.append({
        "key": "hot:current",
        "value": "fresh_hot_fact",
        "confidence": 1.0,
        "timestamp": now,
        "is_superseded": False,
    })

    class FakeEngine:
        def __init__(self, kv_ref):
            self.kv = kv_ref
            self.vector_store = None
            self.graph = None

    pipeline = RetrievalPipeline(hot_buffer=hot, engine=FakeEngine(kv))
    res = await pipeline.execute_prefetch({
        "query": "python version preference",
        "session_id": "hermes_default",
        "force": True,
    })

    subjects = [r["subject"] for r in res["records"]]
    assert "pref:old" not in subjects
    assert "hot:superseded" not in subjects
    assert "pref:active" in subjects


@pytest.mark.asyncio
async def test_t3_policy_gate_respects_limit_and_budget():
    """
    T3: Verifies that when Policy Gate returns should_retrieve=False,
    hot buffer injection strictly respects the 'limit' and 'token_budget' parameters.
    """
    hot = HotBufferPolicy(maxlen=20, ttl_seconds=900.0)
    now = time.time()
    for i in range(10):
        hot.append({
            "key": f"key:{i}",
            "value": f"value_payload_{i}_{'x' * 50}",
            "confidence": 1.0,
            "timestamp": now + i,
            "metadata": {"session_id": "hermes_default"},
        })

    class MockOrchestrator:
        def should_retrieve(self, query: str):
            return False, None, "chit_chat_gate"

    pipeline = RetrievalPipeline(hot_buffer=hot)
    res = await pipeline.execute_prefetch(
        {
            "query": "system architecture status check",
            "session_id": "hermes_default",
            "limit": 3,
            "token_budget": 1000,
        },
        orchestrator=MockOrchestrator(),
    )

    assert res["count"] == 3
    assert len(res["records"]) == 3
    for r in res["records"]:
        assert r["predicate"] == "active_working_memory"


@pytest.mark.asyncio
async def test_t9_hot_buffer_preserves_freshest_entries_first():
    """
    T9: Verifies that hot buffer records are ordered newest-first so that
    slicing [:limit] preserves the freshest entries rather than oldest ones.
    """
    hot = HotBufferPolicy(maxlen=10, ttl_seconds=900.0)
    now = time.time()
    hot.append({
        "key": "entry:oldest",
        "value": "v1_oldest",
        "timestamp": now - 100,
        "metadata": {"session_id": "hermes_default"},
    })
    hot.append({
        "key": "entry:middle",
        "value": "v2_middle",
        "timestamp": now - 50,
        "metadata": {"session_id": "hermes_default"},
    })
    hot.append({
        "key": "entry:newest",
        "value": "v3_newest",
        "timestamp": now,
        "metadata": {"session_id": "hermes_default"},
    })

    pipeline = RetrievalPipeline(hot_buffer=hot)
    res = await pipeline.execute_prefetch({
        "query": "",
        "session_id": "hermes_default",
        "limit": 2,
    })

    assert res["count"] == 2
    subjects = [r["subject"] for r in res["records"]]
    assert subjects == ["entry:newest", "entry:middle"]
    assert "entry:oldest" not in subjects


@pytest.mark.asyncio
async def test_t6_vector_and_kv_session_isolation_strict():
    """
    T6: Verifies strict session isolation:
    - Default environment records (session_id=None or '') are NOT accessible to foreign sessions.
    - Vector hits and KV records tagged with an alien session are not leaked.
    """
    assert not is_session_accessible(None, "foreign_session_999")
    assert not is_session_accessible("", "foreign_session_999")
    assert not is_session_accessible("hermes_default", "foreign_session_999")
    assert not is_session_accessible("session_alpha", "session_beta")

    assert is_session_accessible("global", "foreign_session_999")
    assert is_session_accessible(None, "hermes_default")
    assert is_session_accessible(None, "default")
    assert is_session_accessible("session_alpha", "session_alpha")

    class MockVectorStore:
        async def search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
            return [
                {
                    "id": "vec_doc_1",
                    "score": 0.90,
                    "record": {
                        "subject": "alien:secret",
                        "object": "alien confidential payload",
                        "metadata": {"session_id": "session_alien_777"},
                    },
                },
                {
                    "id": "vec_doc_2",
                    "score": 0.88,
                    "record": {
                        "subject": "global:announcement",
                        "object": "global shared maintenance schedule",
                        "metadata": {"session_id": "global"},
                    },
                },
            ]

    class FakeEngine:
        def __init__(self):
            self.kv = None
            self.vector_store = MockVectorStore()
            self.graph = None

    pipeline = RetrievalPipeline(engine=FakeEngine())
    res = await pipeline.execute_prefetch({
        "query": "confidential payload schedule",
        "session_id": "session_user_target",
        "force": True,
    })

    subjects = [r["subject"] for r in res["records"]]
    assert "alien:secret" not in subjects
    assert "global:announcement" in subjects


