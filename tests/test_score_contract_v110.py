"""V110 (P0) — Kontrakt normalizacji score ∈ [0, 1] dla wszystkich kandydatów.

Asercje przez publiczne API ``score_candidates`` (strategie RRF i DBSF),
na fixture'ach identycznych jak w tests/unit/test_audit_redactions_and_provenance.py.

Regresja (audit v93): ``c["score"] = round(rrf + ..., 4)`` mogło przekroczyć 1.0
(sum: rrf 1.02 + auth 0.01 + conf 0.005 + rule 0.035; analogicznie DBSF fused +
tie-breaki). Po naprawie KAŻDA ścieżka zapisu ``c["score"]`` przechodzi przez
``min(1.0, .)``.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from atlas_memory.cognitive.retrieval.score import score_candidates

STRATEGIES = ["rrf", "dbsf"]


def _fixture_candidates() -> List[Dict[str, Any]]:
    """Fixture spójna z tests/unit/test_audit_redactions_and_provenance.py."""
    subjects = [
        "rule:preferred_editor", "rule:git:branch_strategy", "fact:user_role", "fact:project_stack",
        "preference:language", "constraint:no_force_push", "rule:test_first", "fact:daemon_socket",
    ]
    objects = [
        "nvim jest preferowanym edytorem uzytkownika", "branch strategy: zawsze worktree, nigdy main",
        "architect", "python pixi ruff pytest", "polski w rozmowie", "nigdy force push na main",
        "testy przed kodem RED GREEN", "~/.hermes/atlas.sock unix socket",
    ]
    cands = []
    for i, (s, o) in enumerate(zip(subjects, objects, strict=True)):
        cands.append({
            "subject": s,
            "predicate": "stated_memory",
            "object": o,
            "confidence": 1.0,
            "source_type": "user_explicit",
            "timestamp": 1_700_000_000.0,
            "vector_score": round(0.9 - 0.07 * i, 4),
        })
    return cands


def _assert_bounds(ranked: List[Dict[str, Any]], label: str) -> None:
    assert ranked, f"{label}: score_candidates zwrocil pusta liste"
    for c in ranked:
        s = float(c["score"])
        assert 0.0 <= s <= 1.0, (
            f"{label}: kandydat {c.get('subject', c.get('key', '?'))} ma score={s!r} poza [0.0, 1.0]"
        )


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_v110_score_contract_prose_query(strategy: str) -> None:
    """P0: score ∈ [0,1] kazdego kandydata — zapytanie prose, strategia rrf i dbsf."""
    cands = _fixture_candidates()
    ranked = score_candidates("preferowany edytor nvim", [dict(c) for c in cands], fusion_strategy=strategy)
    _assert_bounds(ranked, f"prose/{strategy}")


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_v110_score_contract_symbolic_query(strategy: str) -> None:
    """P0: score ∈ [0,1] — zapytanie symboliczne/identifier (sciezka DBSF_ALPHA_SYMBOLIC)."""
    cands = _fixture_candidates()
    ranked = score_candidates("BRANCH_STRATEGY branch strategy worktree", [dict(c) for c in cands], fusion_strategy=strategy)
    _assert_bounds(ranked, f"symbolic/{strategy}")


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_v110_saturated_candidate_capped_at_one(strategy: str) -> None:
    """P0: kandydat nasycony (vs=1.0, exact phrase, auth rule, conf 1.0) nie przekracza 1.0.

    To dokladna sciezka regresji z audytu: rrf = 0.5*(1*1 + 1*1) + 0.02 = 1.02
    + auth_tiebreak + conf_tiebreak + rule_tiebreak ~ 1.069 > 1.0 przed naprawa.
    """
    cands = [
        {
            "subject": "rule:preferred_editor",
            "predicate": "stated_memory",
            "object": "preferowany edytor nvim",  # exact phrase query == object
            "confidence": 1.0,
            "source_type": "user_explicit",
            "timestamp": 1_700_000_000.0,
            "vector_score": 1.0,
        },
        {
            "subject": "fact:noise",
            "predicate": "stated_memory",
            "object": "zupelnie inny temat",
            "confidence": 0.4,
            "source_type": "agent_inference",
            "vector_score": 0.05,
        },
    ]
    ranked = score_candidates("preferowany edytor nvim", cands, fusion_strategy=strategy)
    _assert_bounds(ranked, f"saturated/{strategy}")
    top = ranked[0]
    assert float(top["score"]) == 1.0, (
        f"saturated/{strategy}: oczekiwano sufitu 1.0 na nasyconym kandydacie, dostano {top['score']!r}"
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_v110_score_never_negative(strategy: str) -> None:
    """P0: kandydaci bez zadnego trafienia maja score 0.0 (nigdy ujemny)."""
    cands = _fixture_candidates()
    cands.append({
        "subject": "fact:unrelated_topic",
        "predicate": "stated_memory",
        "object": "całkowicie niepasujący temat xyzzy",
        "confidence": 0.5,
        "source_type": "agent_inference",
        "vector_score": 0.0,
    })
    ranked = score_candidates("preferowany edytor nvim", cands, fusion_strategy=strategy)
    for c in ranked:
        assert 0.0 <= float(c["score"]) <= 1.0, (
            f"{strategy}: score poza zakresem: {c['score']!r}"
        )


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_v110_ranking_order_preserved_below_ceiling(strategy: str) -> None:
    """P0: kolejnosc malejaca zachowana (sufit nie miesza relative-order < 1.0)."""
    cands = _fixture_candidates()
    ranked = score_candidates("python pytest ruff", [dict(c) for c in cands], fusion_strategy=strategy)
    scores = [float(c["score"]) for c in ranked]
    assert scores == sorted(scores, reverse=True), f"{strategy}: ranking nie jest malejacy: {scores}"
