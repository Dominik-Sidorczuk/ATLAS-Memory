"""V114 F-114 — mierzalny próg kontynuacji w `evaluate_gate`.

Defekt v93 (udokumentowany w ADR-114): ścieżka continuation zwracała
`GateDecision(., threshold=0.0)`, więc prefetch serwował KAŻDEGO gorącego
kandydata (1244/1500 tokenów szumu na gorącej sesji). Raport kalibracyjny
v93: mediana score odrzuceń `below_threshold` w gałęzi continuation leżała
w paśmie [0.10, 0.18]; ustawienie `CONTINUATION_MIN_SCORE = 0.18` odcina
ten szum, zachowując gorące dowody.

Kontrakty testowane:
 (a) gorący kandydat ze score >= 0.18 → accepted, threshold=0.18,
     reason='continuation' (próg mierzalny, brak szumu);
 (b) gorący kandydat ze score < 0.18 → can_retrieve=False,
     reason='continuation_below_threshold', threshold=0.18;
 (c) regresja: continuation prompt BEZ kandydatów (czysta pompa 'dalej')
     → reason='continuation_prompt', threshold=0.0 (bez zmian);
 (d) publiczny symbol CONTINUATION_MIN_SCORE = 0.18 + importowalny.
"""
from __future__ import annotations

from atlas_memory.cognitive.retrieval.gate import (
    CONTINUATION_MIN_SCORE,
    evaluate_gate,
)


# (d) stała publiczna + zakres tolerancji z ADR (ściśle 0.18; bez driftu).
def test_d_continuation_min_score_constant_is_018_with_adr_justification():
    assert CONTINUATION_MIN_SCORE == 0.18, (
        "CONTINUATION_MIN_SCORE jest częścią kontraktu V114; zmiana wymaga ADR"
    )
    assert isinstance(CONTINUATION_MIN_SCORE, float)


def _hot(score: float, subject: str = "x") -> dict:
    return {"source": "hot_buffer", "subject": subject, "score": score}


def test_a_hot_candidate_above_threshold_accepted_with_measurable_threshold():
    """(a) score=0.25 >= 0.18 → accepted, threshold=CONTINUATION_MIN_SCORE."""
    decision = evaluate_gate("Kontynuuj działanie", [_hot(0.25, "session_turn_7")])
    assert decision.should_retrieve is True
    assert decision.reason == "continuation"
    assert decision.threshold == CONTINUATION_MIN_SCORE


def test_b_hot_candidate_below_threshold_rejected_below_threshold_reason():
    """(b) score=0.05 < 0.18 → False + 'continuation_below_threshold'."""
    decision = evaluate_gate("Kontynuuj działanie", [_hot(0.05, "session_turn_7")])
    assert decision.should_retrieve is False
    assert decision.reason == "continuation_below_threshold"
    assert decision.threshold == CONTINUATION_MIN_SCORE


def test_b_boundary_exactly_at_threshold_accepted():
    """Dokładnie 0.18 = accepted (>=). Granica jest częścią kontraktu."""
    decision = evaluate_gate("Kontynuuj działanie", [_hot(0.18, "session_turn_7")])
    assert decision.should_retrieve is True
    assert decision.reason == "continuation"


def test_c_continuation_prompt_without_candidates_keeps_legacy_reason():
    """(c) regresja: brak scored_candidates → continuation_prompt, threshold=0.0."""
    decision = evaluate_gate("Kontynuuj działanie", [])
    assert decision.should_retrieve is False
    assert decision.reason == "continuation_prompt"
    assert decision.threshold == 0.0


def test_c_continuation_prompt_with_only_non_hot_candidates_unchanged():
    """(c) regresja: continuation z kandydatami spoza hot_buffer → continuation_prompt."""
    decision = evaluate_gate(
        "Kontynuuj działanie",
        [{"source": "vector", "subject": "x", "score": 0.99}],
    )
    assert decision.should_retrieve is False
    assert decision.reason == "continuation_prompt"
    assert decision.threshold == 0.0
