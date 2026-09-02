from __future__ import annotations

import time

from atlas_memory.extensions.compactor import ContextCompactor
from atlas_memory.extensions.epistemic import EpistemicCalibrator
from atlas_memory.models import EpistemicSource, MemoryRecord


def test_epistemic_calibration_sources():
    calibrator = EpistemicCalibrator()

    user_rec = MemoryRecord(
        subject="server", predicate="has_ip", object="10.0.0.1", confidence=1.0,
        source_type=EpistemicSource.USER_EXPLICIT
    )
    inferred_rec = MemoryRecord(
        subject="server", predicate="has_ip", object="10.0.0.2", confidence=1.0,
        source_type=EpistemicSource.AGENT_INFERENCE
    )

    calibrated_user = calibrator.calibrate(user_rec)
    calibrated_inferred = calibrator.calibrate(inferred_rec)

    assert calibrated_user.confidence == 1.0
    assert calibrated_inferred.confidence == 0.60


def test_arbitration_user_explicit_overrides_agent_inference():
    calibrator = EpistemicCalibrator()
    t0 = time.time()

    existing = MemoryRecord(
        subject="db_port", predicate="is", object="5432", timestamp=t0,
        source_type=EpistemicSource.AGENT_INFERENCE, confidence=0.9
    )
    incoming = MemoryRecord(
        subject="db_port", predicate="is", object="5433", timestamp=t0 - 100,  # even if older
        source_type=EpistemicSource.USER_EXPLICIT, confidence=0.8
    )

    wins, reason = calibrator.arbitrate_conflict(existing, incoming)
    assert wins is True
    assert "epistemic_override" in reason


def test_arbitration_agent_inference_cannot_override_user_explicit():
    calibrator = EpistemicCalibrator()
    t0 = time.time()

    existing = MemoryRecord(
        subject="api_key", predicate="is", object="secret_a", timestamp=t0,
        source_type=EpistemicSource.USER_EXPLICIT, confidence=1.0
    )
    incoming = MemoryRecord(
        subject="api_key", predicate="is", object="secret_b", timestamp=t0 + 1000,  # newer
        source_type=EpistemicSource.AGENT_INFERENCE, confidence=0.9
    )

    wins, reason = calibrator.arbitrate_conflict(existing, incoming)
    assert wins is False
    assert "epistemic_rejected" in reason


def test_arbitration_equal_rank_newer_timestamp_wins():
    calibrator = EpistemicCalibrator()
    t0 = time.time()

    existing = MemoryRecord(
        subject="status", predicate="is", object="idle", timestamp=t0,
        source_type=EpistemicSource.TOOL_OUTPUT, confidence=0.9
    )
    incoming = MemoryRecord(
        subject="status", predicate="is", object="busy", timestamp=t0 + 10,
        source_type=EpistemicSource.TOOL_OUTPUT, confidence=0.9
    )

    wins, reason = calibrator.arbitrate_conflict(existing, incoming)
    assert wins is True
    assert "timestamp_override" in reason


def test_arbitration_equal_rank_and_time_higher_confidence_wins():
    calibrator = EpistemicCalibrator()
    t0 = 1000.0

    existing = MemoryRecord(
        subject="flag", predicate="is", object="0", timestamp=t0,
        source_type=EpistemicSource.EXTERNAL_DOC, confidence=0.6
    )
    incoming = MemoryRecord(
        subject="flag", predicate="is", object="1", timestamp=t0,
        source_type=EpistemicSource.EXTERNAL_DOC, confidence=0.85
    )

    wins, reason = calibrator.arbitrate_conflict(existing, incoming)
    assert wins is True
    assert "confidence_override" in reason


def test_context_compactor_turn_buffering_and_compaction():
    compactor = ContextCompactor(session_window_size=4)

    needs_compaction = False
    for i in range(4):
        needs_compaction = compactor.add_interaction_turn(
            role="user" if i % 2 == 0 else "assistant",
            content=f"Krok {i}: ustaw port: 808{i}",
        )

    assert needs_compaction is True
    level = compactor.compact_working_window("test_ep")
    assert level.source_items_count == 4
    assert len(level.extracted_facts) > 0
    assert "808" in level.compressed_text
