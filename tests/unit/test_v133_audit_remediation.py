"""Unit tests covering remediation of the V133 Holistic Architectural Audit findings."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import networkx as nx
import pytest

from atlas_memory.cognitive.calibration_engine import IsotonicCalibrator, is_symbolic_or_identifier
from atlas_memory.cognitive.conflict_arbiter import ConflictArbiter
from atlas_memory.cognitive.lexicon import extract_exclusive_target
from atlas_memory.cognitive.models import ArbitrationResult
from atlas_memory.cognitive.predictive_coder import PredictiveCoder
from atlas_memory.cognitive.retrieval.candidates import (
    gather_candidates,
)
from atlas_memory.cognitive.retrieval.models import CandidateEntity
from atlas_memory.cognitive.retrieval.ppr import expand_associative_seeds_2hop
from atlas_memory.cognitive.retrieval.prefetch import PrefetchEngine
from atlas_memory.cognitive.retrieval.render import render_context_block
from atlas_memory.cognitive.retrieval.strategies.fusion import (
    RrfStrategy,
    compute_record_rank,
)
from atlas_memory.cognitive.retrieval.strategies.pipeline import ScoringPipeline
from atlas_memory.cognitive.write_path import EpistemicWritePath


def test_candidate_entity_preserves_source_channel_and_origin():
    """1. Zniszczenie tożsamości źródła danych (CandidateEntity.from_dict & to_dict)."""
    raw_data = {
        "key": "test:key",
        "value": "sample value",
        "source": "hot_buffer",
        "source_type": "user_explicit",
        "confidence": 0.95,
    }
    entity = CandidateEntity.from_dict(raw_data)
    assert entity.source == "hot_buffer", "Storage channel must be hot_buffer"
    assert entity.source_type == "user_explicit", "Origin ontology must be user_explicit"

    exported = entity.to_dict()
    assert exported["source"] == "hot_buffer"
    assert exported["source_type"] == "user_explicit"


def test_continuation_selection_filters_below_threshold():
    """2. Regresja ADR-114 i wyciek szumu w ścieżce Continuation."""
    pe = PrefetchEngine()
    gate = MagicMock()
    gate.reason = "continuation_prompt"
    gate.threshold = 0.25

    scored = [
        {"subject": "active:task", "source": "hot_buffer", "score": 0.30},
        {"subject": "noise:task", "source": "hot_buffer", "score": 0.05},
        {"subject": "rule:persisted", "source": "hot_buffer", "score": 0.90},  # Filtered by PERSISTENT_KEY_PREFIXES
    ]
    selected, skip = pe._assemble_selection("kontynuuj", scored, gate, force=False, limit=5, eff_cal=None)
    assert skip is None
    assert len(selected) == 1
    assert selected[0]["subject"] == "active:task"


@pytest.mark.asyncio
async def test_kv_exact_match_does_not_pollute_vector_score_and_allows_real_vector():
    """3. Fałszowanie dowodu wektorowego przez dopasowanie leksykalne KV."""
    mock_kv = MagicMock()
    mock_kv.get_arbitration_candidates_sync = MagicMock(return_value={})
    mock_kv.get_all_sync = MagicMock(return_value={
        "fact:query": {"value": "matched value", "metadata": {}},
    })
    mock_kv.get = MagicMock(return_value={"value": "matched value"})

    mock_vec = MagicMock()
    # Real Qdrant hit returns cosine 0.88
    mock_vec.search = MagicMock(return_value=[
        {"id": "fact:query", "text": "matched value", "metadata": {"kv_key": "fact:query"}, "score": 0.88}
    ])

    cands = await gather_candidates(
        query="fact:query",
        kv_store=mock_kv,
        vector_store=mock_vec,
        limit=5,
    )
    by_key = {c["key"]: c for c in cands}
    assert "fact:query" in by_key
    target = by_key["fact:query"]
    assert target.get("exact_kv_hit") is True
    assert target.get("vector_score") == pytest.approx(0.88, abs=0.01)


@pytest.mark.asyncio
async def test_kv_batch_verification_graceful_degradation():
    """4. Kaskadowa eliminacja trafień wektorowych przy niedostępności KV."""
    mock_vec = MagicMock()
    mock_vec.search = MagicMock(return_value=[
        {"id": "doc:1", "text": "important text", "metadata": {"kv_key": "doc:1"}, "score": 0.85}
    ])
    failing_kv = MagicMock()
    failing_kv.get_arbitration_candidates_sync = MagicMock(side_effect=RuntimeError("Database locked"))
    failing_kv.get_all_sync = MagicMock(side_effect=RuntimeError("Database locked"))
    failing_kv.get = MagicMock(side_effect=RuntimeError("Database locked"))

    cands = await gather_candidates(
        query="important text",
        kv_store=failing_kv,
        vector_store=mock_vec,
        limit=5,
    )
    assert len(cands) >= 1
    assert any(c["key"] == "doc:1" for c in cands), "Vector hits must not be discarded when KV store fails"


def test_trim_by_token_budget_knapsack():
    """5. Knapsack 0/1 packaging instead of brittle sequential break."""
    records = [
        {"subject": "s1", "object": "short fact 1", "score": 0.90},
        {"subject": "s2", "object": "very long text " * 10, "score": 0.80},
        {"subject": "s3", "object": "short fact 2", "score": 0.85},
    ]
    budgeted, tot = PrefetchEngine._trim_by_token_budget(records, budget=40)
    subjects = [r["subject"] for r in budgeted]
    assert "s1" in subjects
    assert "s3" in subjects
    assert "s2" not in subjects


def test_render_context_block_clean_truncation():
    """6. Martwy kod i destrukcyjne obcinanie w formatowaniu kontekstu."""
    records = [
        {"subject": "user:pref", "object": "Use Python 3.14 for all components", "source": "kv"},
        {"subject": "db:host", "object": "localhost:5432", "source": "kv"},
    ]
    block = render_context_block(records, max_total_chars=80)
    assert block.startswith("## ATLAS Memory Context")
    assert not block.endswith("[SUPERSEDED]")
    assert not block.endswith("[SUPERSEDED→")


def test_ppr_expand_associative_seeds_2hop_contains_traversal():
    """7. Logiczna sprzeczność w wieloskokowej propagacji asocjacyjnej."""
    g = nx.DiGraph()
    g.add_node("concept:auth", type="entity")
    g.add_node("concept:oauth", type="entity")
    g.add_node("rule:token_ttl", type="entity")

    g.add_edge("concept:auth", "concept:oauth", relation="SYNONYM", weight=1.0)
    g.add_edge("concept:oauth", "rule:token_ttl", relation="CONTAINS", weight=0.9)

    phrase_seeds = {"concept:auth": 1.0}
    expanded = expand_associative_seeds_2hop(g, phrase_seeds)
    assert "rule:token_ttl" in expanded, "Hop 2 must traverse CONTAINS relations"
    assert expanded["rule:token_ttl"] > 0.0


def test_isotonic_calibrator_predict_proba_dict_subcurves():
    """1. Błąd wykonania (Crash) przy predykcji w IsotonicCalibrator."""
    cal = IsotonicCalibrator()
    cal.by_type = {
        "prose": {
            "knots_x": [0.0, 0.4, 0.8, 1.0],
            "knots_y": [0.0, 0.3, 0.7, 1.0],
            "optimal_threshold": 0.35,
        }
    }
    prob = cal.predict_proba(0.6, query_type="prose")
    assert isinstance(prob, float)
    assert 0.3 <= prob <= 0.7


def test_dedup_candidates_preserves_distinct_keys():
    """2. Katastrofalna kolizja i utrata unikalnych faktów w dedup_candidates."""
    pipe = ScoringPipeline()
    candidates = [
        {"key": "db:port", "object": "5432", "source": "kv", "vector_score": 0.5},
        {"key": "proxy:port", "object": "5432", "source": "kv", "vector_score": 0.6},
        {"key": "db:port", "object": "5432", "source": "vector", "vector_score": 0.8},
    ]
    deduped = pipe.dedup_candidates(candidates)
    keys = {c["key"] for c in deduped}
    assert "db:port" in keys
    assert "proxy:port" in keys
    assert len(deduped) == 2, "db:port and proxy:port must coexist, while duplicate db:port is merged"


def test_epistemic_write_path_rollback_restores_old_state():
    """4. Destrukcyjny, niepełny rollback (Epistemic Amnesia)."""
    kv = MagicMock()
    existing = {
        "rule:timeout": {"value": "30s", "confidence": 1.0, "metadata": {"source_type": "user_explicit"}}
    }
    kv.get_arbitration_candidates_sync = MagicMock(return_value=existing)
    kv.set_sync = MagicMock()
    kv.delete_sync = MagicMock()

    failing_graph = MagicMock()
    failing_graph.add_relation = MagicMock(side_effect=RuntimeError("Graph storage full"))

    arbiter = ConflictArbiter()
    wp = EpistemicWritePath(kv_store=kv, graph_store=failing_graph, conflict_arbiter=arbiter)

    with pytest.raises(RuntimeError):
        wp.write(
            key="rule:timeout:new",
            value="60s",
            confidence=1.0,
            metadata={"predicate": "equals", "object": "60s", "supersedes": "rule:timeout"},
        )

    # Verify that old key was restored during rollback
    restore_calls = [
        call for call in kv.set_sync.call_args_list
        if call.args and call.args[0] == "rule:timeout" and call.kwargs.get("reason") == "rollback_restore"
    ]
    assert len(restore_calls) == 1, "Old superseded state must be restored on transaction failure"


def test_predictive_coder_zero_reference_discrepancy():
    """9. Dzielenie przez zero i eksplozja błędu predykcji wokół zera."""
    pc = PredictiveCoder()
    res = pc.compute_discrepancy(observed="0.01", expected="0.0")
    assert res["severity"] == "LOW"
    assert res["discrepancy_score"] <= 0.05


def test_compute_record_rank_time_decay_with_none_now():
    """6. Neutralizacja starzenia wiedzy (TTL / Time Decay) w compute_record_rank."""
    fresh_record = {"subject": "status:node", "timestamp": time.time(), "confidence": 1.0}
    old_record = {"subject": "status:node", "timestamp": time.time() - 86400 * 30, "confidence": 1.0}

    rank_fresh = compute_record_rank(fresh_record, now=None)
    rank_old = compute_record_rank(old_record, now=None)
    assert rank_fresh > rank_old, "Older records must have lower rank when now is None"


def test_extract_exclusive_target_stops_at_prepositions():
    """8. Skazanie na szum dopasowania wyłączności przez zachłanny Regex."""
    text1 = "Używaj tylko Pythona w tym projekcie"
    text2 = "Używaj tylko Rusta w tym projekcie"

    targets1 = extract_exclusive_target(text1)
    targets2 = extract_exclusive_target(text2)

    assert "python" in targets1 or "pythona" in targets1
    assert "projekcie" not in targets1
    assert "rust" in targets2 or "rusta" in targets2
    assert "projekcie" not in targets2
    assert len(targets1 & targets2) == 0, "Exclusivity targets must not overlap on prepositional context words"


def test_rrf_extract_features_preserves_public_signals():
    """10. Ekstrakcja cech w fuzji nie niszczy publicznych cech modelu."""
    cand = {"vector_score": 0.8, "bm25_score": 0.6, "graph_score": 0.4, "raw_bm25": 1.2}
    vs, ls, gs, raw_bm, doc_tokens, is_g = RrfStrategy._extract_features(cand)
    assert vs == 0.8
    assert "vector_score" in cand
    assert "bm25_score" in cand


def test_is_symbolic_or_identifier_prose_sentences():
    """7. Klasyfikacja wielowyrazowych zdań jako prose, a struktur kluczy jako symbolic."""
    assert is_symbolic_or_identifier("to jest zwykla baza danych") is False
    assert is_symbolic_or_identifier("jak dziala test pamieci w projekcie") is False
    assert is_symbolic_or_identifier("użytkownik pyta o stan systemu") is False
    assert is_symbolic_or_identifier("hermes:memory_md:db") is True
    assert is_symbolic_or_identifier("db_host") is True
    assert is_symbolic_or_identifier("port-27124") is True


def test_continuation_selection_10_records_exact_2_pass():
    """2b. Weryfikacja odcięcia szumu w sesji: 2 z wynikiem >= 0.18, 8 z wynikiem < 0.18."""
    pe = PrefetchEngine()
    gate = MagicMock()
    gate.reason = "continuation_prompt"
    gate.threshold = 0.18

    scored = [
        {"subject": f"valid:{i}", "source": "hot_buffer", "score": 0.20 + i * 0.05}
        for i in range(2)
    ] + [
        {"subject": f"noise:{i}", "source": "hot_buffer", "score": 0.05 + i * 0.01}
        for i in range(8)
    ]
    selected, skip = pe._assemble_selection("kontynuuj", scored, gate, force=False, limit=10, eff_cal=None)
    assert skip is None
    assert len(selected) == 2
    assert all(r["subject"].startswith("valid:") for r in selected)


def test_same_key_same_value_allows_reobservation():
    """8. Zapis istniejącego klucza tą samą wartością z pewnością 0.85 (przy bazie 1.0) zwraca akcję accept."""
    arbiter = ConflictArbiter()
    existing = {
        "pref:language": {
            "value": "polish",
            "confidence": 1.0,
            "metadata": {"source_type": "user_explicit"},
        }
    }
    res = arbiter.arbitrate(
        key="pref:language",
        val="polish",
        confidence=0.85,
        metadata={"source_type": "tool_output"},
        existing_items=existing,
    )
    assert res.is_conflict is False
    assert res.action == "accept"
    assert res.warning is None


def test_write_path_blocks_hard_rejected_conflict():
    """10. Ścieżka write() twardo odrzuca zapis przy decyzji reject_conflict."""
    kv = MagicMock()
    arbiter = MagicMock()
    arb_res = ArbitrationResult(
        is_conflict=True,
        is_belief_revision=False,
        action="reject_conflict",
        reason="exclusivity_collision",
        message="Hard exclusivity violation",
        conflict_details={"reason": "exclusivity_collision"},
    )
    arbiter.arbitrate = MagicMock(return_value=arb_res)
    wp = EpistemicWritePath(kv_store=kv, conflict_arbiter=arbiter)

    res = wp.write(key="tool:exclusive", value="rust", confidence=0.9)
    assert res["status"] == "rejected"
    assert res["persisted"] is False
    assert res["action"] == "reject_conflict"
    kv.set_sync.assert_not_called()


def test_ppr_cache_invalidation_on_deep_edge_mutation():
    """5. Modyfikacja krawędzi o indeksie > 100 unieważnia cache ig.Graph."""
    from atlas_memory.cognitive.retrieval.ppr import HippoRAG2PPR
    ppr = HippoRAG2PPR()
    g = nx.DiGraph()
    for i in range(150):
        g.add_edge(f"node_{i}", f"node_{i+1}", relation="REL", weight=1.0)

    # Initial build
    ppr._build_graph_structures(g)
    sig1 = ppr._cached_sig

    # Mutate edge 120 (far beyond index 64)
    g["node_120"]["node_121"]["weight"] = 2.5
    ppr._build_graph_structures(g)
    sig2 = ppr._cached_sig

    assert sig1 != sig2, "Cache signature must change when deep edge (>64) is mutated"
