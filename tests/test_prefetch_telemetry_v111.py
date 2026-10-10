"""V111 F-028 — uczciwa telemetria prefetch: empty_pool + no_results.

Kontrakty:
 T1: niepusty query + pula po scoringu PUSTA -> evaluate_gate musi ODRZUCIC
     (should_retrieve=False, reason='empty_pool'), nie udawać 'accepted'.
 T1b:端到端 — prefetch z pustą pulą zwraca skipped=True, reason='empty_pool'.
 T2: bramka accepted, ale entropy filter wycina wszystko -> count==0 ORAZ
     reason=='no_results' (NIE 'accepted'), skipped=False (bramka przepuściła).
 T3: regresja — normalny accepted z count>0 -> reason='accepted' bez zmian.

Unit/offline: żadnego demona, żadnego socketa, żadnych sieci. Monkeypatch
`score_candidates`/`gather_candidates` w module prefetch (wzór mocków jak
w tests/unit/test_retrieval_policy_and_prefetch.py).
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from atlas_memory.cognitive.retrieval.gate import evaluate_gate
from atlas_memory.cognitive.retrieval.prefetch import PrefetchEngine

PROSE_QUERY = "jak działa proces kalibracji w systemie"


# ---------------------------------------------------------------------------
# T1 — bramka: pusta pula przy niepustym query to ODRZUCENIE, nie sukces
# ---------------------------------------------------------------------------

def test_t1_empty_pool_gate_rejects_with_empty_pool_reason():
    gate = evaluate_gate(PROSE_QUERY, [])
    assert gate.should_retrieve is False, (
        "pusta pula kandydatów nie może być maskowana jako accepted (F-028)"
    )
    assert gate.reason == "empty_pool"
    # próg/fallback selekcji nie zmieniają się dla tej ścieżki
    assert gate.threshold == 0.25
    assert gate.threshold_selection_reason == "no_candidate_fallback"


def test_t1_empty_query_semantics_unchanged():
    """empty_query/continuation/system_notification wracają wcześniejszymi
    gałęziami — gałąź empty_pool wolno ruszać TYLKO ścieżki pełnego query."""
    g_empty = evaluate_gate("", [])
    assert g_empty.reason == "empty_query"
    assert g_empty.should_retrieve is False

    g_notif = evaluate_gate("[system: noop]", [])
    assert g_notif.should_retrieve is False
    assert g_notif.reason == "system_notification"


def test_t1_below_threshold_still_wins_over_empty_pool():
    """Niepusta pula pod progiem => below_threshold (bez zmian)."""
    pool = [{"subject": "fact:noise", "score": 0.10, "object": "x", "content": "x"}]
    gate = evaluate_gate(PROSE_QUERY, pool)
    assert gate.should_retrieve is False
    assert gate.reason == "below_threshold"


# ---------------------------------------------------------------------------
# helpers — mocki zależności prefetch (offline)
# ---------------------------------------------------------------------------

def _raw_candidate() -> Dict[str, Any]:
    return {
        "subject": "fact:kalibracja:opis",
        "predicate": "stated_memory",
        "object": "Kalibracja progowa działa na pulach kandydatów.",
        "content": "Kalibracja progowa działa na pulach kandydatów.",
        "source": "kv",
        "score": 0.26,
    }


def _scored_ungrounded_graph_tail() -> List[Dict[str, Any]]:
    """score >= 0.25 (gate accepted), ale g>0, d<0.12, l<0.08, s<0.28
    -> realny apply_entropy_salience_filter wycina wszystko (selected=[])."""
    return [{
        "subject": "fact:siatka:asocjacja",
        "predicate": "stated_memory",
        "object": "Słaba asocjacja grafowa bez kotwicy dense/sparse.",
        "content": "Słaba asocjacja grafowa bez kotwicy dense/sparse.",
        "source": "graph",
        "score": 0.26,
        "dense_score": 0.05,
        "sparse_score": 0.02,
        "graph_score": 0.30,
        "confidence": 1.0,
    }]


def _scored_grounded() -> List[Dict[str, Any]]:
    """Kandydat kotwiczony (g=0, wysokie dense/sparse) — przechodzi entropy
    filter i budget -> selected niepuste."""
    return [{
        "subject": "fact:kalibracja:opis",
        "predicate": "stated_memory",
        "object": "Kalibracja progowa działa na pulach kandydatów w ATLAS.",
        "content": "Kalibracja progowa działa na pulach kandydatów w ATLAS.",
        "source": "kv",
        "score": 0.45,
        "dense_score": 0.50,
        "sparse_score": 0.30,
        "graph_score": 0.0,
        "confidence": 1.0,
    }]


def _install_mocks(
    monkeypatch: pytest.MonkeyPatch,
    scored_pool: List[Dict[str, Any]],
) -> None:
    async def fake_gather(**kwargs: Any) -> List[Dict[str, Any]]:
        return [_raw_candidate()] if scored_pool else []

    def fake_score(query: str, candidates: List[Dict[str, Any]], fusion_strategy: Any = None, **kwargs: Any):
        return [dict(c) for c in scored_pool]

    monkeypatch.setattr(
        "atlas_memory.cognitive.retrieval.prefetch.gather_candidates", fake_gather
    )
    monkeypatch.setattr(
        "atlas_memory.cognitive.retrieval.prefetch.score_candidates", fake_score
    )


# ---------------------------------------------------------------------------
# T1b — end-to-end: pusta pula przez prefetch => skipped + empty_pool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t1b_prefetch_empty_pool_reports_empty_pool_not_accepted(monkeypatch: pytest.MonkeyPatch):
    _install_mocks(monkeypatch, scored_pool=[])
    engine = PrefetchEngine()
    res = await engine.execute({"query": PROSE_QUERY, "session_id": "v111-telemetry"})
    assert res["skipped"] is True
    assert res["reason"] == "empty_pool", (
        f"pusta pula nie może zwracać reason='accepted'; dostałem {res['reason']!r}"
    )
    assert res["count"] == 0


# ---------------------------------------------------------------------------
# T2 — accepted + selected=[] => count==0 ORAZ reason=='no_results'
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t2_gate_accepted_but_entropy_cuts_all_yields_no_results(monkeypatch: pytest.MonkeyPatch):
    _install_mocks(monkeypatch, scored_pool=_scored_ungrounded_graph_tail())
    engine = PrefetchEngine()
    res = await engine.execute({"query": PROSE_QUERY, "session_id": "v111-telemetry"})
    # bramka realnie przepuściła: skipped=False zostaje
    assert res["skipped"] is False
    assert res["count"] == 0
    assert res["reason"] == "no_results", (
        f"count=0 przy przepuszczonej bramce musi być 'no_results', nie 'accepted'; "
        f"dostałem {res['reason']!r}"
    )
    assert res["records"] == []


# ---------------------------------------------------------------------------
# T3 — regresja behawioralna: normalny accepted z wynikami bez zmian
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t3_normal_accepted_with_results_keeps_accepted_reason(monkeypatch: pytest.MonkeyPatch):
    _install_mocks(monkeypatch, scored_pool=_scored_grounded())
    engine = PrefetchEngine()
    res = await engine.execute({"query": PROSE_QUERY, "session_id": "v111-telemetry"})
    assert res["skipped"] is False
    assert res["count"] == 1
    assert res["reason"] == "accepted"
    assert res["records"][0]["subject"] == "fact:kalibracja:opis"
