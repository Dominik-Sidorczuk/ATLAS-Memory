"""V121 (T1) — Unit tests for score_candidates decomposition helpers.

Pokrywa edge case'y per-helper (pusty pool, ties, missing fields, monotoniczność).
Nie testuje Golden Master (to robi tests/test_v117_golden_master.py).
Kontrakt: helpery są prywatne (_-prefix), więc import przez tests jest akceptowalny
jako biały-box smoke po refaktorze monolitu CC=96.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from atlas_memory.cognitive.retrieval.score import score_candidates
from atlas_memory.cognitive.retrieval.strategies.fusion import (
    RrfStrategy,
    StaticAnchorDbsfFusion,
    _assemble_rrf_score,
    _write_dbsf_scores,
    compute_tiebreaks,
    write_metadata_mirror,
)
from atlas_memory.cognitive.retrieval.strategies.lexical import (
    LEX_EXACT_PHRASE_BONUS,
    Bm25LexicalStrategy,
)
from atlas_memory.cognitive.retrieval.strategies.pipeline import ScoringPipeline

_pipe = ScoringPipeline()
_dbsf = StaticAnchorDbsfFusion()
_dedup_candidates = _pipe.dedup_candidates
_dedup_rank_key = ScoringPipeline.dedup_rank_key
_attach_scoring_features = ScoringPipeline.attach_scoring_features
_build_rank_indices = ScoringPipeline.build_rank_indices
_compute_tiebreaks = compute_tiebreaks
_write_metadata_mirror = write_metadata_mirror
_apply_lexical_phrase_boost = Bm25LexicalStrategy.apply_phrase_boost
_apply_graph_synergy = StaticAnchorDbsfFusion._apply_graph_synergy
_combine_base_fused = StaticAnchorDbsfFusion._combine_base_fused
_compute_ttl_factor = StaticAnchorDbsfFusion._compute_ttl_factor
_dbsf_lex_signals = _dbsf._lex_signals
_dbsf_rule_tiebreak = StaticAnchorDbsfFusion._dbsf_rule_tiebreak
_is_ungrounded_scoped_entity = StaticAnchorDbsfFusion._is_ungrounded_scoped_entity
_extract_populated_features = RrfStrategy._extract_features
_write_zero_signal = RrfStrategy._write_zero_signal

# ─── _dedup_candidates / _dedup_rank_key ──────────────────────────────────────


def test_dedup_empty_input():
    assert _dedup_candidates([]) == []


def test_dedup_no_collisions_keeps_all():
    a = {"subject": "x:1", "object": "alpha", "key": "k1", "source": "kv"}
    b = {"subject": "x:2", "object": "beta", "key": "k2", "source": "kv"}
    out = _dedup_candidates([a, b])
    assert len(out) == 2


def test_dedup_key_in_c_no_source_appends_key_suffix():
    """'key' w c bez 'source' → fingerprint + key::c['key'] → unikalne FP per rekord."""
    a = {"object": "same object text here", "key": "kk1"}
    b = {"object": "same object text here", "key": "kk2"}
    out = _dedup_candidates([a, b])
    assert len(out) == 2  # różne key → różne fp


def test_dedup_higher_authority_wins():
    """Gdy cur_rank > prev_rank → cur wygrywa; merged_vec/graph to max z obu."""
    a = {
        "subject": "fact:low", "object": "same o",
        "key": "k", "source": "kv",
        "vector_score": 0.5, "graph_score": 0.0, "graph_hit": False,
    }
    b = {
        "subject": "rule:high", "object": "same o",
        "key": "k", "source": "kv",
        "vector_score": 0.7, "graph_score": 0.4, "graph_hit": True,
    }
    out = _dedup_candidates([a, b])
    assert len(out) == 1
    # winner ma authority rule → wyższy → b
    assert out[0]["subject"] == "rule:high"
    assert out[0]["vector_score"] == pytest.approx(0.7)
    assert out[0]["graph_score"] == pytest.approx(0.4)
    assert out[0]["graph_hit"] is True


def test_dedup_lower_authority_loses_but_merged_kept():
    """Gdy cur_rank <= prev_rank → prev wygrywa, ale merged_vec/graph to max z obu."""
    a = {
        "subject": "rule:high", "object": "same o",
        "key": "k", "source": "kv",
        "vector_score": 0.5, "graph_score": 0.0, "graph_hit": False,
    }
    b = {
        "subject": "fact:low", "object": "same o",
        "key": "k", "source": "kv",
        "vector_score": 0.9, "graph_score": 0.6, "graph_hit": True,
    }
    out = _dedup_candidates([a, b])
    assert len(out) == 1
    assert out[0]["subject"] == "rule:high"
    # merged max zostaje na prev
    assert out[0]["vector_score"] == pytest.approx(0.9)
    assert out[0]["graph_score"] == pytest.approx(0.6)
    assert out[0]["graph_hit"] is True


def test_dedup_rank_key_handles_missing_fields():
    """Brak graph_score/vector_score → None-or-fallback w _dedup_rank_key."""
    c = {"subject": "rule:x"}
    key = _dedup_rank_key(c)
    assert key == (110, 0.0, 0.0)


# ─── _apply_lexical_phrase_boost ──────────────────────────────────────────────


def test_phrase_boost_adds_when_present():
    cands = [{"subject": "what is atlas", "object": "memory system"}]
    scores = [1.0]
    _apply_lexical_phrase_boost("atlas", cands, scores)
    assert scores[0] == pytest.approx(1.0 + LEX_EXACT_PHRASE_BONUS)


def test_phrase_boost_no_match_no_change():
    cands = [{"subject": "hello", "object": "world"}]
    scores = [2.0]
    _apply_lexical_phrase_boost("atlas", cands, scores)
    assert scores[0] == pytest.approx(2.0)


def test_phrase_boost_empty_query_no_change():
    cands = [{"subject": "x", "object": "y"}]
    scores = [0.5]
    _apply_lexical_phrase_boost("", cands, scores)
    assert scores[0] == pytest.approx(0.5)


# ─── _attach_scoring_features ─────────────────────────────────────────────────


def test_attach_features_normalizes_lex():
    cands = [{"subject": "a", "object": "b", "vector_score": 0.4, "graph_score": 0.1}]
    _attach_scoring_features(cands, [2.0], 4.0, [["a", "b"]])
    assert cands[0]["_lex_score"] == pytest.approx(0.5, rel=1e-4)
    assert cands[0]["_raw_bm25"] == 2.0
    assert cands[0]["_vec_score"] == 0.4
    assert cands[0]["_graph_score"] == 0.1
    assert cands[0]["_doc_tokens_set"] == {"a", "b"}


def test_attach_features_zero_max_bm25_uses_zero_norm():
    cands = [{"subject": "a", "object": "b", "vector_score": 0.0, "graph_score": None}]
    _attach_scoring_features(cands, [1.0], 0.0, [[]])
    assert cands[0]["_lex_score"] == 0.0
    assert cands[0]["_graph_score"] == 0.0


def test_attach_features_graph_score_from_metadata_fallback():
    cands = [{"subject": "x", "object": "y", "metadata": {"graph_score": 0.42}}]
    _attach_scoring_features(cands, [0.0], 0.0, [[]])
    assert cands[0]["_graph_score"] == pytest.approx(0.42)


def test_attach_features_clamps_graph_score_to_unit_interval():
    cands = [{"subject": "x", "object": "y", "graph_score": 1.5}]
    _attach_scoring_features(cands, [0.0], 0.0, [[]])
    assert cands[0]["_graph_score"] == 1.0


# ─── _build_rank_indices ──────────────────────────────────────────────────────


def test_rank_indices_one_based_unique():
    a = {"_vec_score": 0.9, "_lex_score": 0.1}
    b = {"_vec_score": 0.5, "_lex_score": 0.7}
    c = {"_vec_score": 0.1, "_lex_score": 0.9}
    rv, rl = _build_rank_indices([a, b, c])
    assert rv[id(a)] == 1
    assert rv[id(c)] == 3
    assert rl[id(c)] == 1
    assert rl[id(a)] == 3


# ─── _extract_populated_features / _write_zero_signal ─────────────────────────


def test_extract_pops_temp_keys_and_bumps_graph_cand_gs():
    c = {
        "_vec_score": 0.5, "_lex_score": 0.3, "_graph_score": 0.0,
        "_raw_bm25": 1.5, "_doc_tokens_set": {"x"}, "graph_hit": True,
    }
    vs, ls, gs, raw_bm25, toks, is_graph = _extract_populated_features(c)
    assert vs == 0.5 and ls == 0.3 and raw_bm25 == 1.5
    assert toks == {"x"}
    assert is_graph is True
    assert gs == 0.50  # bumped
    assert "_vec_score" not in c
    assert "_lex_score" not in c
    assert "_graph_score" not in c
    assert "_raw_bm25" not in c
    assert "_doc_tokens_set" not in c


def test_extract_keeps_gs_when_already_positive():
    c = {"_vec_score": 0.0, "_lex_score": 0.0, "_graph_score": 0.7, "_raw_bm25": 0.0, "_doc_tokens_set": set()}
    vs, ls, gs, *_ = _extract_populated_features(c)
    assert gs == 0.7


def test_write_zero_signal_sets_relevance_and_zeros():
    c = {"subject": "x"}
    _write_zero_signal(c)
    assert c["score"] == 0.0
    assert c["rrf_score"] == 0.0
    assert c["graph_hit"] is False
    assert c["relevance_reason"] == "no_match(sim=0.0)"


# ─── _compute_tiebreaks ──────────────────────────────────────────────────────


def test_tiebreaks_user_explicit_high_authority():
    c = {"subject": "rule:x", "source_type": "user_explicit", "confidence": 1.0}
    auth, vw, auth_tb, conf_tb = _compute_tiebreaks(c)
    assert auth == pytest.approx(1.10, rel=1e-3)
    assert vw == 1.0
    assert auth_tb > 0.0
    assert conf_tb > 0.0


def test_tiebreaks_default_source_type_uses_fallback_vw():
    c = {"subject": "x", "source_type": "unknown_kind"}
    _, vw, _, _ = _compute_tiebreaks(c)
    assert vw == 0.5


# ─── _assemble_rrf_score ──────────────────────────────────────────────────────


def test_assemble_rrf_dense_and_lex_combined():
    c: Dict[str, Any] = {"subject": "rule:x"}
    _assemble_rrf_score(
        c, vs=0.8, ls=0.5, gs=0.0, is_trusted=True,
        rank_vec_id=1, rank_lex_id=2, auth=1.10,
        auth_tiebreak=0.01, conf_tiebreak=0.005,
    )
    assert c["score"] > 0.0
    assert c["dense_score"] == 0.8
    assert c["sparse_score"] == 0.5
    assert c["graph_score"] == 0.0
    assert "dense(rank=1" in c["relevance_reason"]


def test_assemble_rrf_only_lex_untrusted_damped():
    c: Dict[str, Any] = {"subject": "x"}
    _assemble_rrf_score(
        c, vs=0.0, ls=0.6, gs=0.0, is_trusted=False,
        rank_vec_id=10, rank_lex_id=1, auth=0.5,
        auth_tiebreak=0.005, conf_tiebreak=0.003,
    )
    # Untrusted single-modality: 0.15 × min(1, ls) × rb_norm — powinno być mniejsze niż trusted
    assert c["score"] >= 0.0


def test_assemble_rrf_graph_boost_added():
    c: Dict[str, Any] = {"subject": "x"}
    _assemble_rrf_score(
        c, vs=0.0, ls=0.5, gs=0.8, is_trusted=True,
        rank_vec_id=10, rank_lex_id=1, auth=0.7,
        auth_tiebreak=0.005, conf_tiebreak=0.003,
    )
    assert c["graph_score"] == pytest.approx(0.8)
    assert c["score"] > 0.0


# ─── _dbsf_lex_signals ───────────────────────────────────────────────────────


def test_dbsf_lex_signals_prose_uses_default_alpha():
    is_sym, alpha, *_ = _dbsf_lex_signals("tell me about Atlas memory", [], set(), 0.6, 0.0, 1.0)
    assert is_sym is False
    assert alpha == pytest.approx(0.35)


def test_dbsf_lex_signals_symbolic_uses_symbolic_alpha():
    is_sym, alpha, *_ = _dbsf_lex_signals("atlas:kalibracja:progi", [], set(), 0.6, 0.0, 1.0)
    assert is_sym is True
    assert alpha == pytest.approx(0.30)


def test_dbsf_lex_signals_matched_stems_and_query_cov():
    _, _, matched, cov, *_ = _dbsf_lex_signals("alpha beta", ["alpha", "beta"], {"alpha"}, 0.5, 0.0, 0.0)
    assert matched == 1
    assert cov == pytest.approx(0.5)


def test_dbsf_lex_signals_query_cov_default_one_for_empty_stems():
    _, _, matched, cov, *_ = _dbsf_lex_signals("anything", [], set(), 0.0, 0.0, 0.0)
    assert matched == 0
    assert cov == 1.0


# ─── _is_ungrounded_scoped_entity ─────────────────────────────────────────────


def test_ungrounded_scoped_entity_with_colon_and_no_match():
    assert _is_ungrounded_scoped_entity(
        "hermes:fakt:alpha", ["some", "other"], matched_stems=0, ls=0.0, is_trusted=False, gs=0.0,
    ) is True


def test_ungrounded_scoped_entity_ignored_when_trusted():
    assert _is_ungrounded_scoped_entity(
        "hermes:fakt:alpha", ["some"], matched_stems=0, ls=0.0, is_trusted=True, gs=0.0,
    ) is False


def test_ungrounded_scoped_entity_ignored_when_graph_present():
    assert _is_ungrounded_scoped_entity(
        "hermes:fakt:alpha", ["some"], matched_stems=0, ls=0.0, is_trusted=False, gs=0.5,
    ) is False


def test_ungrounded_scoped_entity_ignored_when_no_colon():
    assert _is_ungrounded_scoped_entity(
        "somerandomkey", ["some"], matched_stems=0, ls=0.0, is_trusted=False, gs=0.0,
    ) is False


# ─── _combine_base_fused ──────────────────────────────────────────────────────


def test_base_fused_dense_and_lex_weighted():
    fused = _combine_base_fused(
        vs=0.8, ls=0.4, n_dense=0.7, n_lex=0.3, eff_alpha=0.35,
        is_trusted=False, ungrounded_scoped_entity=False, matched_stems=1, q_stems=["x", "y"],
    )
    # 0.35*0.7 + 0.65*0.3 = 0.245 + 0.195 = 0.44
    assert fused == pytest.approx(0.44, rel=1e-3)


def test_base_fused_trusted_short_circuit_above_078():
    fused = _combine_base_fused(
        vs=0.79, ls=0.1, n_dense=0.9, n_lex=0.1, eff_alpha=0.35,
        is_trusted=True, ungrounded_scoped_entity=False, matched_stems=1, q_stems=["x"],
    )
    assert fused == pytest.approx(0.9, rel=1e-3)


def test_base_fused_single_modality_lex_only_trusted():
    fused = _combine_base_fused(
        vs=0.0, ls=0.7, n_dense=0.0, n_lex=0.7, eff_alpha=0.35,
        is_trusted=True, ungrounded_scoped_entity=False, matched_stems=0, q_stems=["x"],
    )
    assert fused == pytest.approx(0.7, rel=1e-3)


def test_base_fused_single_modality_untrusted_damped():
    fused = _combine_base_fused(
        vs=0.0, ls=0.7, n_dense=0.0, n_lex=0.7, eff_alpha=0.35,
        is_trusted=False, ungrounded_scoped_entity=False, matched_stems=0, q_stems=["x"],
    )
    assert fused == pytest.approx(0.65 * 0.7, rel=1e-3)


def test_base_fused_ungrounded_scoped_entity_damp_to_015():
    fused = _combine_base_fused(
        vs=0.5, ls=0.0, n_dense=0.5, n_lex=0.0, eff_alpha=0.35,
        is_trusted=False, ungrounded_scoped_entity=True, matched_stems=0, q_stems=["x", "y"],
    )
    assert fused == pytest.approx(0.075, rel=1e-3)


# ─── _apply_graph_synergy ─────────────────────────────────────────────────────


def test_graph_synergy_skipped_when_no_graph():
    fused = _apply_graph_synergy(
        c={}, fused=0.5, vs=0.7, ls=0.0, gs=0.0,
        n_dense=0.5, n_lex=0.0, matched_stems=1, q_stems=["x"],
        is_sym=False, is_trusted=False, eff_alpha=0.35,
    )
    assert fused == 0.5


def test_graph_synergy_ungrounded_graph_only_damped_to_018():
    c: Dict[str, Any] = {}
    fused = _apply_graph_synergy(
        c=c, fused=0.7, vs=0.5, ls=0.0, gs=0.9,
        n_dense=0.1, n_lex=0.0, matched_stems=0, q_stems=["a", "b", "c"],
        is_sym=False, is_trusted=False, eff_alpha=0.35,
    )
    assert fused <= 0.18
    assert c.get("_guard_damped") == "ungrounded_graph"


def test_graph_synergy_only_graph_no_vector_or_lex_capped_022():
    # n_dense >= 0.18 → nie wchodzi w ungrounded_graph_only; ścieżka vs<=0 && ls<=0
    fused = _apply_graph_synergy(
        c={}, fused=0.0, vs=0.0, ls=0.0, gs=0.5,
        n_dense=0.20, n_lex=0.0, matched_stems=0, q_stems=["x"],
        is_sym=False, is_trusted=False, eff_alpha=0.35,
    )
    assert fused == pytest.approx(min(0.22, 0.30 * 0.5), rel=1e-3)


def test_graph_synergy_dense_and_graph_adds_small_boost():
    fused = _apply_graph_synergy(
        c={}, fused=0.5, vs=0.6, ls=0.0, gs=0.6,
        n_dense=0.5, n_lex=0.0, matched_stems=1, q_stems=["x"],
        is_sym=False, is_trusted=False, eff_alpha=0.35,
    )
    # 0.9*0.5 + 0.1*0.6 + 0.02*0.6 = 0.45 + 0.06 + 0.012 = 0.522
    assert fused == pytest.approx(0.522, rel=1e-3)


# ─── _dbsf_rule_tiebreak ──────────────────────────────────────────────────────


def test_rule_tiebreak_zero_for_ungrounded_scoped():
    assert _dbsf_rule_tiebreak(1.1, 0.8, 0.5, 0.6, True, ungrounded_scoped_entity=True) == 0.0


def test_rule_tiebreak_high_auth_and_strong_anchor():
    assert _dbsf_rule_tiebreak(1.0, 0.7, 0.0, 0.0, False, ungrounded_scoped_entity=False) == 0.035


def test_rule_tiebreak_no_match_for_low_auth():
    assert _dbsf_rule_tiebreak(0.5, 0.7, 0.0, 0.0, False, ungrounded_scoped_entity=False) == 0.0


# ─── _compute_ttl_factor ──────────────────────────────────────────────────────


def test_ttl_factor_one_for_non_episodic_subject():
    assert _compute_ttl_factor(
        "fact:kalibracja:prog", c={},
        matched_stems=2, q_stems=["x", "y"], n_dense=0.5, gs=0.0,
    ) == 1.0


def test_ttl_factor_damped_for_old_episodic():
    c = {"timestamp": 1.0}  # < 1e8 → age_days=0 → ttl_factor stays 1.0
    factor = _compute_ttl_factor(
        "fact:episodic_event", c=c,
        matched_stems=2, q_stems=["x", "y"], n_dense=0.5, gs=0.0,
    )
    # Brak ts > 1e8 → age_days nie rośnie → factor 1.0
    assert factor == 1.0


def test_ttl_factor_single_word_coincidence_damped_when_no_graph():
    factor = _compute_ttl_factor(
        "fact:kalibracja:prog", c={},
        matched_stems=1, q_stems=["a", "b", "c", "d"], n_dense=0.1, gs=0.0,
    )
    # len>=3 and matched<=1 and n_dense<0.28 and gs<=0 → *= 0.55
    assert factor == pytest.approx(0.55, rel=1e-3)


# ─── _write_dbsf_scores / _write_metadata_mirror ─────────────────────────────


def test_write_dbsf_scores_caps_at_one():
    c: Dict[str, Any] = {}
    _write_dbsf_scores(c, fused=2.0, n_dense=1.0, n_lex=1.0, gs=1.0, vs=1.0,
                       is_graph_cand=True, auth_tiebreak=0.01, conf_tiebreak=0.005, rule_tiebreak=0.035)
    assert c["score"] == 1.0
    assert "dbsf(dense=" in c["relevance_reason"]


def test_write_metadata_mirror_overlays_existing_keys():
    c = {"subject": "x", "metadata": {"score": 99.0, "extra": "kept"}}
    c["score"] = 0.5
    c["relevance_reason"] = "test"
    c["dense_score"] = 0.4
    c["sparse_score"] = 0.3
    c["graph_score"] = 0.2
    c["graph_hit"] = True
    _write_metadata_mirror(c)
    assert c["metadata"]["score"] == 0.5
    assert c["metadata"]["extra"] == "kept"
    assert c["metadata"]["graph_hit"] is True


# ─── score_candidates: end-to-end contracts ───────────────────────────────────


def test_score_candidates_empty_input_returns_empty():
    assert score_candidates("anything", []) == []


def test_score_candidates_dedups_and_sorts_descending():
    cands = [
        {"subject": "fact:low", "object": "alpha content here", "key": "k", "source": "kv", "vector_score": 0.3},
        {"subject": "rule:high", "object": "alpha content here", "key": "k", "source": "kv", "vector_score": 0.3},
        {"subject": "hermes:fakt:alone", "object": "beta", "key": "u", "source": "kv", "vector_score": 0.6},
    ]
    out = score_candidates("alpha content", cands)
    assert len(out) == 2  # dedup
    scores = [c["score"] for c in out]
    assert scores == sorted(scores, reverse=True)
    assert all("score" in c and "relevance_reason" in c for c in out)
