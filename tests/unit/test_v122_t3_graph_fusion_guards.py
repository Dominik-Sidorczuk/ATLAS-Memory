"""V122 (T3) — Graph Fusion Guards, Ungrounded Graph Damping, and Golden Master Parity.

Contracts tested:
  1. ungrounded_graph_only caps ungrounded graph-only hits at <= 0.18.
  2. Grounded graph hits (with dense >= 0.18 or lexical >= 0.22) receive DBSF graph synergy boost.
  3. Golden Master 30/30 harness and S3 canary hash (18396978e674f39d) pass bit-for-bit without drift.
"""
from __future__ import annotations

from atlas_memory.cognitive.retrieval.score import score_candidates
from tests.test_v117_golden_master import KNOWN_GOOD_HASH, SCORE_VECTORS, sha16


def test_t3_ungrounded_graph_only_damped_to_max_0p18() -> None:
    # Query with no lexical overlap with candidate text
    query = "konfiguracja bazy danych postgres"
    # Candidate with zero vector dense score and low/zero lexical score, but high graph score
    cand = {
        "key": "fact:ciasto:przepis",
        "subject": "ciasto czekoladowe",
        "predicate": "wymaga",
        "object": "maliny i czekolada",
        "source": "graph",
        "scores": {
            "dense": 0.0,
            "lex": 0.0,
            "graph": 0.95,
            "sim": 0.0,
        },
        "vector_score": 0.0,
        "graph_score": 0.95,
        "confidence": 0.9,
    }
    scored = score_candidates(query, [cand])
    assert len(scored) == 1
    res_score = scored[0]["score"]
    fused_score = scored[0]["rrf_score"]
    assert fused_score <= 0.18, f"Ungrounded graph fused score must be <= 0.18, got {fused_score}"
    assert res_score <= 0.20, f"Ungrounded graph score must be capped <= 0.20, got {res_score}"
    assert scored[0].get("_guard_damped") == "ungrounded_graph"


def test_t3_grounded_graph_receives_dbsf_boost() -> None:
    query = "konfiguracja CORS"
    # Candidate with dense grounding and graph hit
    cand_grounded = {
        "key": "fact:cors:config",
        "subject": "CORS",
        "predicate": "configured_on",
        "object": "Code OSS server production instance",
        "source": "vector_dense",
        "scores": {
            "dense": 0.70,
            "lex": 0.50,
            "graph": 0.85,
            "sim": 0.70,
        },
        "vector_score": 0.70,
        "graph_score": 0.85,
        "confidence": 0.9,
    }
    cand_no_graph = {
        "key": "fact:cors:config_no_g",
        "subject": "CORS",
        "predicate": "configured_on",
        "object": "Code OSS server fallback instance",
        "source": "vector_dense",
        "scores": {
            "dense": 0.70,
            "lex": 0.50,
            "graph": 0.0,
            "sim": 0.70,
        },
        "vector_score": 0.70,
        "graph_score": 0.0,
        "confidence": 0.9,
    }
    scored = score_candidates(query, [cand_grounded, cand_no_graph])
    scores_by_key = {c["key"]: c["score"] for c in scored}
    assert scores_by_key["fact:cors:config"] > scores_by_key["fact:cors:config_no_g"]


def test_t3_golden_master_s3_canary_hash_intact() -> None:
    q, cands = SCORE_VECTORS["S3_collision_fingerprints"]
    out = score_candidates(q, [dict(c) for c in cands])
    expected_hash, _ = KNOWN_GOOD_HASH["S3_collision_fingerprints"]
    assert expected_hash == "18396978e674f39d"
    actual_hash = sha16(out)
    assert actual_hash == expected_hash, f"Golden Master S3 hash drifted: {actual_hash} != {expected_hash}"
