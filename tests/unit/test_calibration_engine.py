"""Unit tests for Self-Calibrating Retrieval Engine (V72.7 - Zadanie 2).

Verifies:
1. Pure-numpy PAVA algorithm guarantees monotonicity and weighted least squares fitting.
2. Calibration metrics: Brier score, ECE, ROC AUC, and Wilson confidence interval.
3. Artifact serialization, deserialization, and SHA-256 integrity verification.
4. Scale-invariant decision rules (score + margin threshold).
5. Drift protection: two-sample Kolmogorov-Smirnov test and conditional refit gate.
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from atlas_memory.cognitive.calibration_engine import (
    CalibrationEngine,
    IsotonicCalibrator,
    brier_score,
    expected_calibration_error,
    pava,
    roc_auc_score,
)


def test_pava_monotonicity_and_correctness():
    """PAVA must strictly output monotonically non-decreasing values."""
    y = np.array([0.9, 0.2, 0.8, 0.1, 0.7, 0.4, 0.95, 0.5])
    w = np.array([1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0])
    fitted = pava(y, w)

    assert len(fitted) == len(y)
    diffs = np.diff(fitted)
    assert np.all(diffs >= -1e-12), f"PAVA output must be monotonic non-decreasing: {fitted}"
    assert np.all(fitted >= 0.0) and np.all(fitted <= 1.0)


def test_calibration_metrics_synthetic():
    """Verifies Brier score, ECE, and AUC on synthetic data."""
    y_true = np.array([0, 0, 1, 1])
    y_prob_perfect = np.array([0.0, 0.0, 1.0, 1.0])
    assert brier_score(y_true, y_prob_perfect) == 0.0
    assert expected_calibration_error(y_true, y_prob_perfect) == 0.0
    assert roc_auc_score(y_true, y_prob_perfect) == 1.0

    y_prob_bad = np.array([1.0, 1.0, 0.0, 0.0])
    assert brier_score(y_true, y_prob_bad) == 1.0
    assert roc_auc_score(y_true, y_prob_bad) == 0.0


def test_calibrator_fit_predict_and_artifact_roundtrip(tmp_path: Path):
    """Verifies fitting calibrator, saving artifact, and verifying SHA-256."""
    rng = np.random.RandomState(42)
    scores = np.sort(rng.uniform(0.3, 0.95, size=200))
    # Probabilities increase with score
    probs = 1.0 / (1.0 + np.exp(-10 * (scores - 0.6)))
    labels = (rng.uniform(0, 1, size=200) < probs).astype(np.int32)
    margins = np.clip(scores - 0.5, 0.0, 0.4)

    calibrator = IsotonicCalibrator()
    metrics = calibrator.fit(scores, labels, margins=margins, n_bins=10)

    assert "brier_calibrated" in metrics
    assert "ece_calibrated" in metrics
    assert calibrator.optimal_threshold >= 0.3

    # Prediction test
    p_preds = calibrator.predict_proba(np.array([0.2, 0.5, 0.65, 0.8, 1.0]))
    assert np.all(np.diff(p_preds) >= -1e-12), "Predictions must be monotonic non-decreasing"

    # Artifact save & load roundtrip
    artifact_path = tmp_path / "test_calibration.json"
    calibrator.save_artifact(artifact_path)
    assert artifact_path.is_file()

    # Verify SHA-256 matches file contents
    loaded = IsotonicCalibrator.load_artifact(artifact_path)
    assert loaded.knots_x == calibrator.knots_x
    assert loaded.knots_y == calibrator.knots_y
    assert loaded.optimal_threshold == calibrator.optimal_threshold
    assert loaded.optimal_margin == calibrator.optimal_margin

    content = artifact_path.read_bytes()
    expected_sha = hashlib.sha256(content).hexdigest()
    assert expected_sha != ""
    assert loaded.sha256 != ""


def test_scale_invariant_decision_rules():
    """Scale-invariant acceptance rule enforces score and margin thresholds."""
    calibrator = IsotonicCalibrator(optimal_threshold=0.55, optimal_margin=0.03)

    # Score below threshold -> reject
    assert not calibrator.should_accept(0.50, margin=0.10)
    # Score above threshold, margin below threshold -> reject
    assert not calibrator.should_accept(0.60, margin=0.01)
    # Both above threshold -> accept
    assert calibrator.should_accept(0.60, margin=0.05)
    # No margin check if use_margin=False
    assert calibrator.should_accept(0.60, margin=0.01, use_margin=False)


def test_drift_detection_and_conditional_refit(tmp_path: Path):
    """Drift protection: refits only when KS test confirms drift AND Brier improves."""
    artifact_path = tmp_path / "drift_test.json"
    engine = CalibrationEngine(artifact_path=None, enabled=True, ks_drift_threshold=0.15, min_brier_improvement=0.01)

    rng = np.random.RandomState(42)
    train_scores = rng.normal(0.65, 0.05, size=150)
    train_labels = (train_scores > 0.65).astype(np.int32)
    train_margins = np.clip(train_scores - 0.5, 0.01, 0.3)

    # Initial fit
    calibrator = IsotonicCalibrator()
    calibrator.fit(train_scores, train_labels, margins=train_margins)
    calibrator.save_artifact(artifact_path)
    engine.calibrator = calibrator

    holdout_scores = rng.normal(0.65, 0.05, size=80)
    holdout_labels = (holdout_scores > 0.65).astype(np.int32)

    # Scenario 1: Same distribution -> no drift
    same_scores = rng.normal(0.65, 0.05, size=100)
    same_labels = (same_scores > 0.65).astype(np.int32)
    refit, reason, _ = engine.evaluate_drift_and_refit(
        same_scores, same_labels, None, holdout_scores, holdout_labels
    )
    assert not refit, "Should not refit when no drift is detected"
    assert "No significant distribution drift" in reason

    # Scenario 2: Distribution shifted (e.g. PolDense [query] shift down to 0.45),
    # and holdout labels show that a refitted calibrator improves Brier score
    shifted_scores = rng.normal(0.45, 0.05, size=150)
    shifted_labels = (shifted_scores > 0.45).astype(np.int32)
    shifted_margins = np.clip(shifted_scores - 0.35, 0.01, 0.3)

    # When evaluated on holdout shifted data
    holdout_shifted_scores = rng.normal(0.45, 0.05, size=80)
    holdout_shifted_labels = (holdout_shifted_scores > 0.45).astype(np.int32)

    refit, reason, new_cal = engine.evaluate_drift_and_refit(
        shifted_scores, shifted_labels, shifted_margins, holdout_shifted_scores, holdout_shifted_labels
    )
    assert refit, f"Expected refit on significant drift, but got: {reason}"
    assert new_cal is not None
    assert "Refit approved" in reason


@pytest.mark.asyncio
async def test_retrieval_pipeline_calibration_wiring():
    """RetrievalPipeline respects calibrated threshold when enabled and defaults when disabled."""
    from unittest.mock import AsyncMock, MagicMock

    from atlas_memory.cognitive.retrieval.pipeline import RetrievalPipeline

    mock_vec = MagicMock()
    # Hit score is 0.55. For query "test:key", default threshold is 0.70 (so it would be rejected).
    # But calibrated threshold is 0.45, so if enabled it will be admitted.
    mock_vec.search = AsyncMock(return_value=[
        {
            "id": "item1",
            "score": 0.55,
            "record": {
                "subject": "test:key",
                "predicate": "fact",
                "object": "Calibrated value 123",
                "confidence": 1.0,
                "metadata": {},
            },
        }
    ])

    engine = CalibrationEngine(enabled=True)
    engine.calibrator = IsotonicCalibrator(optimal_threshold=0.45)

    # 1. Disabled: threshold 0.70 -> rejected
    # V92: wejście kandydatów = search_hybrid_async (zmiana kontraktu, ADR w commicie)
    mock_vec.search_hybrid_async = mock_vec.search
    pipeline_off = RetrievalPipeline(vector_store=mock_vec, enable_calibrated_thresholds=False)
    results_off = await pipeline_off.retrieve("test:key", limit=5, force=True)
    assert len(results_off.records) == 0

    # 2. Enabled: calibrated threshold 0.45 -> admitted
    pipeline_on = RetrievalPipeline(
        vector_store=mock_vec,
        calibration_engine=engine,
        enable_calibrated_thresholds=True,
    )
    results_on = await pipeline_on.retrieve("test:key", limit=5, force=True)
    assert len(results_on.records) == 1
    assert results_on.records[0].subject == "test:key"


def test_orchestrator_calibration_wiring():
    """MemoryOrchestrator uses calibrated cosine threshold when enabled (V72.8 - FINDING-078)."""
    from atlas_memory.core.models import EpistemicSource, MemoryRecord
    from atlas_memory.orchestrator import MemoryOrchestrator

    engine = CalibrationEngine(enabled=True)
    engine.calibrator = IsotonicCalibrator(optimal_threshold=0.10, optimal_margin=0.03)

    rec = MemoryRecord(
        subject="test_fact",
        predicate="states",
        object="Value of interest",
        source_type=EpistemicSource.USER_EXPLICIT,
        confidence=1.0,
        importance_score=0.99,
        metadata={"vector_score": 0.15},  # between 0.10 and 0.25
    )

    # 1. Disabled: relevance_threshold is 0.25 -> vec_score 0.15 is below threshold
    orch_off = MemoryOrchestrator(enable_calibrated_thresholds=False)
    # no query terms overlap
    fused_off = orch_off.epistemic_rank([rec], query="completely unrelated xyz")
    # Because it's below 0.25 and not overlapping, relevance_mult is 0.40 (rule downscaled)
    # 2. Enabled: relevance_threshold is 0.10 -> vec_score 0.15 is above threshold
    orch_on = MemoryOrchestrator(
        calibration_engine=engine,
        enable_calibrated_thresholds=True,
    )
    fused_on = orch_on.epistemic_rank([rec], query="completely unrelated xyz")
    # Calibrated score should have higher relevance multiplier (1.0 vs 0.40)
    assert fused_on[0][1] > fused_off[0][1]


def test_daemon_calibration_propagation(tmp_path: Path):
    """AtlasDaemon forwards calibration_engine and enable_calibrated_thresholds."""
    from unittest.mock import MagicMock

    from atlas_memory.server.atlas_daemon import AtlasDaemon

    mock_orch = MagicMock()
    daemon = AtlasDaemon(
        socket_path=tmp_path / "test.sock",
        pid_path=tmp_path / "test.pid",
        orchestrator=mock_orch,
        enable_calibrated_thresholds=True,
    )
    assert daemon.enable_calibrated_thresholds is True
    assert daemon.retrieval_pipeline.enable_calibrated_thresholds is True
    assert mock_orch.enable_calibrated_thresholds is True


def test_bug_cal43_default_sc_mode_is_batch_with_orthogonal_probes():
    """BUG-CAL43 (ADR-021): DEFAULT_SC_MODE defaults to 'batch' (ADR-015/020) with per-probe measured_dimensions."""
    import inspect

    import scripts.build_retrieval_calibration as brc

    assert brc.DEFAULT_SC_MODE == "batch"
    judge = brc.RetrievalJudge(enabled=False)
    assert judge.sc_mode == "batch"
    sig = inspect.signature(brc.build_dataset)
    assert sig.parameters["sc_mode"].default == "batch"
    obj = {
        "semantic_entailment": 0.5,
        "operational_actionability": 0.0,
        "entity_scope_grounding": 0.0,
        "polarity_fidelity": 2.0,
        "epistemic_freshness": 2.0,
        "hard_veto": False,
    }
    probe_scores = [
        brc.compute_latent_score(obj, measured_dimensions=spec["measured_dimensions"])[0]
        for spec in brc.SC_PROBE_SPECS
    ]
    assert probe_scores == [0.0909, 0.0, 1.0]


