"""Unit tests for OOP ScoringPipeline, Domain Models, and Scoring Strategies."""
from __future__ import annotations

import pytest

from atlas_memory.cognitive.retrieval.models import (
    CandidateEntity,
    CandidateFeatures,
    ScoredCandidate,
)
from atlas_memory.cognitive.retrieval.pipeline import ScoringPipeline
from atlas_memory.cognitive.retrieval.score import score_candidates
from atlas_memory.cognitive.retrieval.strategies import (
    Bm25LexicalStrategy,
    DenseCosineStrategy,
    EntropySalienceFilter,
    GraphSynergyStrategy,
    RrfStrategy,
    StaticAnchorDbsfFusion,
)

# ---------------------------------------------------------------------------
# Domain Models: CandidateEntity, CandidateFeatures, ScoredCandidate
# ---------------------------------------------------------------------------

def test_candidate_entity_instantiation_and_typed_access() -> None:
    entity = CandidateEntity(
        key="fact:arch",
        value="Hermes Agent Architecture",
        subject="fact:arch",
        predicate="stated_memory",
        object="Hermes Agent Architecture",
        confidence=0.95,
        source="kv",
        metadata={"priority": "high"},
        raw_scores={"dense_score": 0.88, "bm25_score": 0.72},
    )

    assert entity.key == "fact:arch"
    assert entity.subject == "fact:arch"
    assert entity.confidence == 0.95
    assert entity.metadata["priority"] == "high"
    assert entity.raw_scores["dense_score"] == 0.88


def test_candidate_entity_from_and_to_dict() -> None:
    raw = {
        "key": "rule:test",
        "value": "Always write tests",
        "subject": "rule:test",
        "object": "Always write tests",
        "confidence": 1.0,
        "source": "rules",
        "vector_score": 0.92,
    }
    entity = CandidateEntity.from_dict(raw)
    assert entity.key == "rule:test"
    assert entity.confidence == 1.0
    assert entity.raw_scores.get("vector_score") == 0.92

    exported = entity.to_dict()
    assert exported["key"] == "rule:test"
    assert exported["confidence"] == 1.0


def test_scored_candidate_model() -> None:
    cand = CandidateEntity(
        key="rule:safe",
        value="zero trust",
        subject="rule:safe",
        object="zero trust",
    )
    features = CandidateFeatures(
        dense_score=0.85,
        bm25_score=0.75,
        graph_score=0.5,
    )
    scored = ScoredCandidate(
        candidate=cand,
        dense_score=0.85,
        bm25_score=0.75,
        graph_score=0.5,
        final_score=0.91,
        rank=1,
        features=features,
    )

    assert scored.final_score == 0.91
    assert scored.score == 0.91
    assert scored.rank == 1
    assert scored.dense_score == 0.85

    d = scored.to_dict()
    assert d["score"] == 0.91
    assert d["dense_score"] == 0.85
    assert d["rank"] == 1


# ---------------------------------------------------------------------------
# Strategies: DenseCosineStrategy
# ---------------------------------------------------------------------------

def test_dense_cosine_strategy_normalization() -> None:
    strategy = DenseCosineStrategy()
    assert strategy.normalize_dense(0.0) == 0.0
    assert strategy.normalize_dense(-0.5) == 0.0

    low = strategy.normalize_dense(0.3)
    mid = strategy.normalize_dense(0.5)
    high = strategy.normalize_dense(0.8)

    assert 0.0 < low < mid < high < 1.0
    assert strategy.normalize_dense(1.0) > 0.99


def test_dense_cosine_strategy_scoring() -> None:
    strategy = DenseCosineStrategy()
    cands = [
        {"key": "c1", "vector_score": 0.75},
        {"key": "c2", "vector_score": 0.25},
    ]
    scores = strategy.score("query", cands)
    assert scores["c1"] > scores["c2"]
    assert cands[0]["_vec_score"] == 0.75
    assert cands[0]["dense_score"] > 0.9


# ---------------------------------------------------------------------------
# Strategies: Bm25LexicalStrategy
# ---------------------------------------------------------------------------

def test_bm25_lexical_strategy_tokenization_and_scoring() -> None:
    strategy = Bm25LexicalStrategy()
    candidates = [
        {"key": "c1", "subject": "fact:python", "object": "Python is a programming language"},
        {"key": "c2", "subject": "fact:rust", "object": "Rust provides memory safety"},
    ]
    tokens = strategy.tokenize_candidates(candidates)
    assert len(tokens) == 2
    assert any("python" in t for t in tokens[0])
    assert any("rust" in t for t in tokens[1])

    scores = strategy.calculate_scores(["python"], tokens)
    assert scores[0] > 0.0
    assert scores[1] == 0.0


def test_bm25_lexical_strategy_phrase_boost() -> None:
    strategy = Bm25LexicalStrategy()
    candidates = [
        {"key": "c1", "subject": "fact:architecture", "object": "clean architecture in python"},
    ]
    scores = [1.5]
    strategy.apply_phrase_boost("clean architecture", candidates, scores)
    assert scores[0] == 2.5


# ---------------------------------------------------------------------------
# Strategies: GraphSynergyStrategy
# ---------------------------------------------------------------------------

def test_graph_synergy_strategy() -> None:
    strategy = GraphSynergyStrategy()
    candidates = [
        {"key": "c1", "graph_score": 0.8, "graph_hit": True},
        {"key": "c2", "graph_score": 0.0, "graph_hit": False},
    ]
    scores = strategy.score("test query", candidates)
    assert scores["c1"] == 0.8
    assert scores["c2"] == 0.0

    # Scoped entity check
    assert strategy.is_ungrounded_scoped_entity(
        s_cand_low="module:submodule:entity",
        q_stems=["other", "words"],
        matched_stems=0,
        ls=0.0,
        is_trusted=False,
        gs=0.0,
    ) is True


# ---------------------------------------------------------------------------
# Strategies: StaticAnchorDbsfFusion & RrfStrategy
# ---------------------------------------------------------------------------

def test_static_anchor_dbsf_fusion() -> None:
    fusion = StaticAnchorDbsfFusion()
    candidates = [
        {
            "key": "c1",
            "subject": "rule:editor",
            "object": "neovim is the primary editor",
            "_vec_score": 0.85,
            "_lex_score": 0.90,
            "_graph_score": 0.0,
            "_raw_bm25": 4.5,
            "_doc_tokens_set": {"neovim", "editor"},
            "confidence": 1.0,
        },
    ]
    fusion.score(query="neovim editor", candidates=candidates)
    assert candidates[0]["score"] > 0.8
    assert candidates[0]["relevance_reason"].startswith("dbsf(")


def test_rrf_strategy() -> None:
    rrf = RrfStrategy()
    candidates = [
        {
            "key": "c1",
            "subject": "fact:python",
            "object": "python tooling",
            "_vec_score": 0.70,
            "_lex_score": 0.80,
            "_graph_score": 0.0,
            "_doc_tokens_set": {"python", "tooling"},
            "confidence": 1.0,
        },
    ]
    rrf.score(query="python tooling", candidates=candidates)
    assert candidates[0]["score"] > 0.0
    assert candidates[0]["relevance_reason"].startswith("dense(")


# ---------------------------------------------------------------------------
# Strategies: EntropySalienceFilter
# ---------------------------------------------------------------------------

def test_entropy_salience_filter() -> None:
    filt = EntropySalienceFilter(threshold=0.1)
    ent_uniform = filt.compute_normalized_entropy([0.25, 0.25, 0.25, 0.25])
    assert ent_uniform == pytest.approx(1.0, abs=1e-3)

    ent_peaked = filt.compute_normalized_entropy([0.99, 0.005, 0.005])
    assert ent_peaked < ent_uniform

    cands = [{"key": "c1", "score": 0.9}, {"key": "c2", "score": 0.1}]
    filtered, eff_thresh = filt.filter(cands)
    assert len(filtered) == 2
    assert eff_thresh >= 0.1


# ---------------------------------------------------------------------------
# ScoringPipeline: Dependency Injection & End-to-End Orchestration
# ---------------------------------------------------------------------------

def test_scoring_pipeline_dependency_injection() -> None:
    dense = DenseCosineStrategy()
    bm25 = Bm25LexicalStrategy()
    graph = GraphSynergyStrategy()
    fusion = StaticAnchorDbsfFusion()

    pipeline = ScoringPipeline(
        strategies=[dense, bm25, graph],
        fusion=fusion,
        fusion_strategy="dbsf",
    )
    assert len(pipeline.strategies) == 3
    assert pipeline.fusion is fusion


def test_scoring_pipeline_with_candidate_entities() -> None:
    pipeline = ScoringPipeline()
    entities = [
        CandidateEntity(
            key="k1",
            value="Pixi environment configuration",
            subject="fact:pixi",
            object="Pixi environment configuration",
            confidence=1.0,
            source="kv",
            raw_scores={"vector_score": 0.85},
        ),
        CandidateEntity(
            key="k2",
            value="Unrelated recipe for soup",
            subject="fact:food",
            object="Unrelated recipe for soup",
            confidence=0.5,
            source="kv",
            raw_scores={"vector_score": 0.10},
        ),
    ]

    scored = pipeline.score("pixi environment", entities, as_models=True)
    assert len(scored) == 2
    assert all(isinstance(c, ScoredCandidate) for c in scored)
    assert scored[0].rank == 1
    assert scored[1].rank == 2
    assert scored[0].score > scored[1].score


def test_scoring_pipeline_dedup_authority_tiebreak() -> None:
    pipeline = ScoringPipeline()
    candidates = [
        {
            "key": "c1",
            "subject": "fact:low_auth",
            "object": "identical content line",
            "source": "kv",
            "vector_score": 0.60,
        },
        {
            "key": "c2",
            "subject": "rule:high_auth",
            "object": "identical content line",
            "source": "kv",
            "vector_score": 0.70,
        },
    ]

    results = pipeline.score("identical content", candidates, as_models=False)
    # Deduplication keeps only one because the object fingerprint collides
    assert len(results) == 1
    # Rule has higher authority than fact
    assert results[0]["subject"] == "rule:high_auth"
    assert results[0]["vector_score"] == 0.70


def test_score_candidates_facade_backward_compatibility() -> None:
    raw_candidates = [
        {
            "key": "fact:editor",
            "subject": "fact:editor",
            "object": "neovim is the chosen editor",
            "vector_score": 0.90,
            "confidence": 1.0,
        },
    ]

    results = score_candidates("neovim editor", raw_candidates)
    assert len(results) == 1
    assert isinstance(results[0], dict)
    assert "score" in results[0]
    assert results[0]["score"] > 0.8
