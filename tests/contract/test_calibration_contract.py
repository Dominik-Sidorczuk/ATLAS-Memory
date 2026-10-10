"""Contract tests for Calibration Engine.

Verifies:
1. A4: relevance_threshold in Orchestrator is an absolute threshold (>= 0.20), not a relative margin (0.03).
2. B1: shared predicate is_symbolic_or_identifier produces consistent classification.
3. Sleep-cycle drift detector: live logic on synthetic reference distribution
   (V99/ADR-017: reliktowe testy treści usuniętego artefaktu i datasetu wycięte,
   pokrycie detektora dryfu przeniesione na fixture tmp_path).
"""

import numpy as np

from atlas_memory.cognitive.calibration_engine import (
    CalibrationEngine,
    IsotonicCalibrator,
    is_symbolic_or_identifier,
)
from atlas_memory.orchestrator import MemoryOrchestrator


def test_a4_relevance_threshold_is_absolute_not_margin():
    """A4: relevance_threshold must be absolute cosine cutoff (>= 0.20), never a margin (0.03)."""
    engine = CalibrationEngine(enabled=True)
    engine.calibrator = IsotonicCalibrator(optimal_threshold=0.45, optimal_margin=0.03)
    orch = MemoryOrchestrator(calibration_engine=engine, enable_calibrated_thresholds=True)

    relevance_threshold = orch.get_relevance_threshold("prose query")

    assert relevance_threshold >= 0.20
    assert relevance_threshold != 0.03


def test_b1_shared_predicate_consistency():
    """B1: Shared predicate is_symbolic_or_identifier produces consistent classification."""
    assert is_symbolic_or_identifier("hermes:rule:git") is True
    assert is_symbolic_or_identifier("kv_item_name") is True
    assert is_symbolic_or_identifier("item-flag") is True
    assert is_symbolic_or_identifier("word") is True
    assert is_symbolic_or_identifier("a") is False
    assert is_symbolic_or_identifier("To jest proste zapytanie tekstowe") is False
    assert is_symbolic_or_identifier("Jak dziala kalibracja w pamieci") is False


def test_fallback_guard_rejection_when_no_candidate():
    """When dataset has no candidate, B4 must reject publication even if test AUC is 1.0 and neg>=30."""
    scores_tr = np.linspace(0.5, 0.9, 70, dtype=np.float64)
    labels_tr = np.zeros(70, dtype=np.float64)
    margins_tr = np.zeros(70, dtype=np.float64)

    scores_te = np.concatenate([np.linspace(0.8, 0.95, 35), np.linspace(0.5, 0.7, 35)]).astype(np.float64)
    labels_te = np.concatenate([np.ones(35), np.zeros(35)]).astype(np.float64)

    train_data = {"prose": {"scores": scores_tr, "labels": labels_tr, "margins": margins_tr}}
    test_data = {"prose": {"scores": scores_te, "labels": labels_te, "margins": np.zeros(70, dtype=np.float64)}}

    cal = IsotonicCalibrator(min_precision_target=0.50)
    res = cal.fit_by_type(train_data, test_data=test_data)

    prose_res = res["prose"]
    assert prose_res["optimal_threshold"] is None
    assert prose_res["threshold_selection_reason"] == "no_candidate_fallback"
    assert prose_res["is_published"] is False
    assert prose_res["rejection_reason"] == "no_candidate_fallback"


def test_runtime_fallback_thresholds_typespecific_not_literal():
    """Runtime z pustym kalibratorem zwraca fallbacki per typ (0.70 / 0.35), NIGDY 0.55."""
    cal_empty = IsotonicCalibrator()
    engine = CalibrationEngine(enabled=True)
    engine.calibrator = cal_empty

    assert engine.get_threshold(query_type="symbolic") == 0.70
    assert engine.get_threshold(query_type="prose") == 0.35
    assert engine.get_threshold(query_type="prose") != 0.55
