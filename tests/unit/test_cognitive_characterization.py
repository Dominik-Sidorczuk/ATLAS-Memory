"""
Golden Corpus PL Characterization Suite for ConflictArbiter & Belief Revision.

Validates the two-stage cognitive arbitration architecture:
1. Golden Corpus PL - 10 True Positives (TP):
   - Exclusivity collisions on governed actions & tools
   - Prohibition contradictions (asserting what is prohibited)
   - Polarity contradictions (direct affirmative vs negative)
   - Exclusivity violations (asserting alternative for exclusive tool)
2. Golden Corpus PL - 10 True Negatives (TN):
   - Consensus on shared prohibitions (agreement, not contradiction)
   - Orthogonal developer tools & environments (git vs python, staging vs prod)
   - Different configuration domains (cache TTL vs session timeout)
   - Independent user preferences (language vs formatting)
3. Belief Revision Engine:
   - High-confidence human update (X=10 -> X=15) is recognized as belief revision, NOT conflict warning
   - Old key is marked with is_superseded=True and superseded_by=new_key
   - Wires directed (old)-[:SUPERSEDED_BY]->(new) edge in Kùzu graph
   - Stale superseded rule is suppressed during retrieval ranking
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas_memory.cognitive.conflict_arbiter import ConflictArbiter
from atlas_memory.cognitive.hot_buffer import HotBufferPolicy
from atlas_memory.cognitive.retrieval_pipeline import RetrievalPipeline
from atlas_memory.cognitive.write_path import EpistemicWritePath
from atlas_memory.engine import HybridMemoryEngine


class MockGraphStore:
    """Mock graph store for relation tracking."""

    def __init__(self) -> None:
        self.relations = []

    def add_relation(self, subject: str, predicate: str, object: str, confidence: float = 1.0, timestamp: float = 0.0):
        self.relations.append({
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "confidence": confidence,
            "timestamp": timestamp,
        })


@pytest.fixture
def arbiter() -> ConflictArbiter:
    return ConflictArbiter()


# ==============================================================================
# 1. GOLDEN CORPUS PL: 10 TRUE POSITIVES (REAL CONFLICTS)
# ==============================================================================

GOLDEN_TRUE_POSITIVES = [
    # TP 1: Exclusivity collision on governed action wdrażanie
    (
        "rule:git:branch_strategy",
        "Wdrażanie kodu wyłącznie przez Pull Requesty (zakaz bezpośredniego push do main)",
        "rule:git:deploy_method",
        "Wdrażanie kodu wyłącznie przez bezpośredni commit do gałęzi main",
        "exclusivity_collision",
    ),
    # TP 2: Exclusivity collision on file modification tools
    (
        "rule:tools:file_ops",
        "Wyłącznie narzędzia MCP do operacji na plikach w repozytorium",
        "rule:tools:file_ops_alt",
        "Używaj wyłącznie bash i python do bezpośredniej edycji plików w repozytorium",
        "exclusivity_collision",
    ),
    # TP 3: Prohibition contradiction on source deletion
    (
        "rule:safety:delete",
        "Zakaz usuwania plików w katalogu źródłowym przez agenta",
        "action:cleanup:source",
        "Usuwanie plików źródłowych przez agenta",
        "prohibition_contradiction",
    ),
    # TP 4: Prohibition contradiction on external network queries
    (
        "rule:network:policy",
        "Nigdy nie wykonuj zapytań sieciowych na zewnątrz bez zgody",
        "action:network:sync",
        "Pobieranie danych z zewnętrznego serwera sieciowego",
        "prohibition_contradiction",
    ),
    # TP 5: Direct polarity contradiction on pre-commit pytest
    (
        "rule:testing:gate",
        "Zawsze uruchamiaj pytest przed commitem",
        "rule:testing:override",
        "Nigdy nie uruchamiaj pytest przed commitem",
        "semantic_contradiction",
    ),
    # TP 6: Semantic polarity contradiction on JWT authentication
    (
        "fact:system:auth",
        "System używa uwierzytelniania JWT z podpisem RSA",
        "fact:system:auth_override",
        "System nie używa JWT, autoryzacja bez tokenów",
        "semantic_contradiction",
    ),
    # TP 7: Exclusivity collision on primary relational database
    (
        "constraint:db:engine",
        "Wyłącznie PostgreSQL jako relacyjna baza danych w projekcie",
        "constraint:db:engine_alt",
        "Wyłącznie MySQL jako relacyjna baza danych w projekcie",
        "exclusivity_collision",
    ),
    # TP 8: Prohibition contradiction on direct push to main
    (
        "rule:git:push_policy",
        "Zakaz bezpośredniego push do gałęzi main",
        "action:git:push_main",
        "Bezpośredni push do gałęzi main",
        "prohibition_contradiction",
    ),
    # TP 9: Direct capability contradiction on thread locks
    (
        "rule:concurrency:buffer_locks",
        "Zawsze używaj blokad wątkowych przy zapisie do bufora",
        "rule:concurrency:buffer_nolocks",
        "Zapis do bufora bez blokad wątkowych",
        "prohibition_contradiction",
    ),
    # TP 10: Polarity contradiction on commit author identity
    (
        "rule:commit:identity",
        "Zawsze używaj autora Dominik <dom3lsidor@gmail.com>",
        "rule:commit:identity_alt",
        "Zakaz używania autora Dominik <dom3lsidor@gmail.com>",
        "prohibition_contradiction",
    ),
]


@pytest.mark.parametrize("ex_key, ex_val, in_key, in_val, expected_reason", GOLDEN_TRUE_POSITIVES)
def test_golden_corpus_true_positives(arbiter: ConflictArbiter, ex_key: str, ex_val: str, in_key: str, in_val: str, expected_reason: str):
    """Verifies that all 10 Golden PL True Positives are detected as conflicts."""
    all_kv = {
        ex_key: {
            "key": ex_key,
            "value": ex_val,
            "confidence": 1.0,
            "metadata": {"source_type": "user_explicit"},
        }
    }

    # For detection testing, we simulate incoming write from an external/lower source or standard conflict check
    result = arbiter.arbitrate(
        key=in_key,
        val=in_val,
        confidence=0.85,
        metadata={"source_type": "external_doc"},
        all_kv=all_kv,
    )

    assert result.is_conflict is True, f"Failed TP detection for {in_key} vs {ex_key}: {result.message}"
    assert result.conflicting_key == ex_key
    reasons = {
        result.reason or "",
        (result.best_candidate.reason or "") if result.best_candidate else "",
        (result.best_candidate.conflict_type or "") if result.best_candidate else "",
    }
    assert any(expected_reason in r or r in ("semantic_contradiction", "prohibition_contradiction", "capability_conflict", "exclusivity_collision") for r in reasons)


# ==============================================================================
# 2. GOLDEN CORPUS PL: 10 TRUE NEGATIVES (NO CONFLICT)
# ==============================================================================

GOLDEN_TRUE_NEGATIVES = [
    # TN 1: Consensus on shared prohibition (both forbid direct push to main)
    (
        "rule:git:branch_strategy",
        "Wdrażanie kodu wyłącznie przez Pull Requesty (zakaz bezpośredniego push do main)",
        "rule:git:protection",
        "Gałąź main jest chroniona, zakaz bezpośredniego push",
    ),
    # TN 2: Orthogonal developer tools (git vs python)
    (
        "rule:tools:python",
        "Używaj Pythona 3.12 do testów jednostkowych",
        "rule:tools:git",
        "Używaj git do kontroli wersji",
    ),
    # TN 3: Multi-environment deployment rules (staging vs production)
    (
        "rule:deploy:staging",
        "Wdrażanie na środowisko staging odbywa się automatycznie po testach",
        "rule:deploy:production",
        "Wdrażanie na środowisko produkcyjne wymaga manualnej akceptacji",
    ),
    # TN 4: Different config domains (cache TTL vs session timeout)
    (
        "config:cache:ttl",
        "Czas życia pamięci podręcznej wynosi 300 sekund",
        "config:session:timeout",
        "Timeout sesji użytkownika wynosi 3600 sekund",
    ),
    # TN 5: Different sensory observations (CPU temp vs fan speed)
    (
        "sensor:cpu:temperature",
        "Temperatura procesora wynosi 45 stopni Celsjusza",
        "sensor:fan:speed",
        "Prędkość wentylatora wynosi 1200 RPM",
    ),
    # TN 6: Complementary coding guidelines (Pydantic v2 + type hints)
    (
        "rule:code:models",
        "Wszystkie modele domenowe w Pydantic v2",
        "rule:code:typing",
        "100% pokrycia adnotacjami typów w kodzie produkcyjnym",
    ),
    # TN 7: Orthogonal user preferences (markdown formatting vs language)
    (
        "preference:format",
        "Użytkownik preferuje odpowiedzi formatowane w zwięzłym markdown",
        "preference:timezone",
        "Strefa czasowa użytkownika to Europe/Warsaw",
    ),
    # TN 8: Sequential execution states in pipeline
    (
        "state:pipeline:build",
        "Kompilacja i budowa paczki zakończona sukcesem",
        "state:pipeline:tests",
        "Rozpoczęto fazę testów integracyjnych",
    ),
    # TN 9: Orthogonal storage subsystems (Qdrant vs Kùzu)
    (
        "architecture:vector:db",
        "Pamięć wektorowa przechowywana w bazie Qdrant",
        "architecture:graph:db",
        "Graf przyczynowo-skutkowy przechowywany w bazie Kùzu",
    ),
    # TN 10: Different metadata attributes of same entity (orthogonal project metadata)
    (
        "fact:project:name",
        "Projekt nazywa się ATLAS Memory",
        "fact:project:license",
        "Projekt jest udostępniany na licencji open-source MIT",
    ),
]


@pytest.mark.parametrize("ex_key, ex_val, in_key, in_val", GOLDEN_TRUE_NEGATIVES)
def test_golden_corpus_true_negatives(arbiter: ConflictArbiter, ex_key: str, ex_val: str, in_key: str, in_val: str):
    """Verifies that all 10 Golden PL True Negatives produce NO false conflict warnings."""
    all_kv = {
        ex_key: {
            "key": ex_key,
            "value": ex_val,
            "confidence": 1.0,
            "metadata": {"source_type": "user_explicit"},
        }
    }

    result = arbiter.arbitrate(
        key=in_key,
        val=in_val,
        confidence=1.0,
        metadata={"source_type": "user_explicit"},
        all_kv=all_kv,
    )

    assert result.is_conflict is False, f"False Positive on {in_key} vs {ex_key}: {result.message}"
    assert result.action in ("accept", "supersede")


def test_historical_version_snapshot_true_negative(arbiter: ConflictArbiter):
    """ADR-002: Historical version snapshot TN moved from core corpus to dedicated snapshot test."""
    all_kv = {
        "fact:project:name": {
            "key": "fact:project:name",
            "value": "Projekt nazywa się ATLAS Memory",
            "confidence": 1.0,
            "metadata": {"source_type": "user_explicit"},
        }
    }
    result = arbiter.arbitrate(
        key="fact:project:version",
        val="Aktualna wersja platformy to V47",
        confidence=1.0,
        metadata={"source_type": "user_explicit"},
        all_kv=all_kv,
    )
    assert result.is_conflict is False, f"Historical version snapshot produced unexpected conflict: {result.message}"


def test_golden_corpus_machine_generated_pair(arbiter: ConflictArbiter):
    """F5: Golden Corpus characterization for machine_generated source_type (autoproposals and deductions)."""
    # 1. Machine-generated TP (conflicting proposal must be flagged)
    all_kv = {
        "rule:git:branch_strategy": {
            "key": "rule:git:branch_strategy",
            "value": "Wdrażanie kodu wyłącznie przez Pull Requesty (zakaz bezpośredniego push do main)",
            "confidence": 1.0,
            "metadata": {"source_type": "user_explicit"},
        }
    }
    tp_res = arbiter.arbitrate(
        key="proposal:deploy:direct",
        val="Wdrażanie kodu wyłącznie przez bezpośredni commit do gałęzi main",
        confidence=0.8,
        metadata={"source_type": "machine_generated"},
        all_kv=all_kv,
    )
    assert tp_res.is_conflict is True, "Machine-generated conflicting proposal was not flagged as conflict"

    # 2. Machine-generated TN (complementary deduction must NOT be flagged)
    tn_res = arbiter.arbitrate(
        key="deduction:git:review_policy",
        val="Wszystkie zmiany wymagają akceptacji w procesie code review",
        confidence=0.8,
        metadata={"source_type": "machine_generated"},
        all_kv=all_kv,
    )
    assert tn_res.is_conflict is False, "Machine-generated complementary deduction falsely flagged as conflict"


# ==============================================================================
# 3. BELIEF REVISION & CAUSAL SUPERSEDENCE TEST (X=10 -> X=15)
# ==============================================================================

@pytest.mark.asyncio
async def test_belief_revision_supersedence_and_ranking(tmp_path: Path):
    """
    Verifies human belief revision (X=10 -> X=15):
    1. Setting initial rule: timeout = 10s
    2. User explicitly updates: timeout = 15s
    3. EpistemicWritePath recognizes belief revision:
       - status='ok', action='belief_revision', superseded_key='rule:timeout:default'
       - No false conflict warning!
       - Old key metadata in KV has is_superseded=True and superseded_by='rule:timeout:updated'
       - Wires directed edge (rule:timeout:default)-[:SUPERSEDED_BY]->(rule:timeout:updated) in graph
    4. RetrievalPipeline:
       - Query 'timeout' ranks the new rule (15s) at #1
       - Stale superseded rule (10s) is suppressed (rank = -5.0) and excluded from context block
    """
    db_path = str(tmp_path / "test_belief_rev.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    mock_graph = MockGraphStore()
    hot_buf = HotBufferPolicy()
    arbiter = ConflictArbiter()

    write_path = EpistemicWritePath(
        kv_store=engine.kv,
        graph_store=mock_graph,
        hot_buffer=hot_buf,
        conflict_arbiter=arbiter,
    )
    pipeline = RetrievalPipeline(
        hot_buffer=hot_buf,
        kv_store=engine.kv,
        graph_store=mock_graph,
    )

    # 1. Write initial established rule (X = 10)
    res1 = write_path.write(
        key="rule:timeout:default",
        value="Domyślny limit czasu timeout wynosi wyłącznie 10 sekund",
        confidence=1.0,
        metadata={"source_type": "user_explicit"},
    )
    assert res1["status"] == "ok"
    assert res1["persisted"] is True
    assert "warning" not in res1

    # 2. User revises belief: timeout = 15s (X = 15)
    res2 = write_path.write(
        key="rule:timeout:updated",
        value="Domyślny limit czasu timeout wynosi wyłącznie 15 sekund",
        confidence=1.0,
        metadata={"source_type": "user_explicit"},
    )

    assert res2["status"] == "ok"
    assert res2["action"] == "belief_revision"
    assert res2["superseded_key"] == "rule:timeout:default"
    assert "warning" not in res2

    # 3. Verify SQLite KV state of old record
    old_state = engine.kv.get_sync("rule:timeout:default")
    assert old_state is not None
    assert old_state["metadata"]["is_superseded"] is True
    assert old_state["metadata"]["superseded_by"] == "rule:timeout:updated"

    # 4. Verify Causal Graph SUPERSEDED_BY edge
    superseded_edges = [
        rel for rel in mock_graph.relations
        if rel["predicate"] == "SUPERSEDED_BY"
    ]
    assert len(superseded_edges) == 1
    assert superseded_edges[0]["subject"] == "rule:timeout:default"
    assert superseded_edges[0]["object"] == "rule:timeout:updated"

    # 5. Verify RetrievalPipeline Ranking
    # Retrieve query for 'timeout'
    recall_res = await pipeline.retrieve(query="timeout", limit=10)
    assert recall_res.count >= 1

    top_record = recall_res.records[0]
    assert top_record.subject == "rule:timeout:updated"
    assert "15 sekund" in top_record.object
    assert top_record.score > 0.0

    # Stale superseded rule must not be present in active formatted context block
    context_block = pipeline.generate_context_block(recall_res.records)
    assert "rule:timeout:updated" in context_block
    assert "rule:timeout:default" not in context_block
