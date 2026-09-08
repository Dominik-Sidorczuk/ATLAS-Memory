from __future__ import annotations

import time
from pathlib import Path

import pytest

from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.extensions.compactor import ContextCompactor
from atlas_memory.extensions.epistemic import EpistemicCalibrator
from atlas_memory.models import EpistemicSource, MemoryRecord
from atlas_memory.server.atlas_daemon import AtlasDaemon


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


@pytest.mark.asyncio
async def test_f1_epistemic_guard(tmp_path: Path):
    """Finding F-1: Epistemic guard in _handle_set detects conflicts and warns."""
    db_path = str(tmp_path / "test_atlas.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    daemon = AtlasDaemon(engine=engine)

    res1 = await daemon._handle_set({
        "key": "system:arch",
        "value": "ATLAS uses SQLite KV store",
        "confidence": 1.0,
        "metadata": {"source_type": "user_explicit"},
    })
    assert res1["status"] == "ok"
    assert res1["persisted"] is True
    assert "warning" not in res1

    res2 = await daemon._handle_set({
        "key": "test:conflict:20260905",
        "value": "ATLAS prefetch uses vector retrieval for semantic matching",
        "confidence": 0.8,
    })
    assert res2["status"] == "ok"
    assert res2["persisted"] is True
    assert res2["warning"] == "epistemic_conflict"
    assert res2["conflict_details"]["reason"] == "explicit_conflict_marker"

    state2 = await daemon._handle_get({"key": "test:conflict:20260905"})
    assert state2["found"] is True
    assert state2["state"]["metadata"].get("conflict_warning") is True

    res3 = await daemon._handle_set({
        "key": "fact:arch_dispute",
        "value": "ATLAS nie używa SQLite KV store w architekturze",
        "confidence": 0.9,
    })
    assert res3["status"] == "ok"
    assert res3["persisted"] is True
    assert res3["warning"] == "epistemic_conflict"
    assert res3["conflict_details"]["reason"] == "semantic_contradiction"


@pytest.mark.asyncio
async def test_f1_polish_exclusivity_and_prohibition_conflict(tmp_path: Path):
    """Finding F-1 (AGI-5): Polish polarity & exclusivity collision detection in epistemic guard."""
    db_path = str(tmp_path / "test_atlas.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    daemon = AtlasDaemon(engine=engine)

    res_canonical = await daemon._handle_set({
        "key": "rule:obsidian:vault_write",
        "value": "EDYCJA vault: WYŁĄCZNIE przez MCP (nigdy bezpośrednio na dysku)",
        "confidence": 1.0,
        "metadata": {"source_type": "user_explicit"},
    })
    assert res_canonical["status"] == "ok"
    assert "warning" not in res_canonical

    res_conflict = await daemon._handle_set({
        "key": "rule:obsidian:edit_method",
        "value": "edycję robimy wyłącznie przez terminal",
        "confidence": 0.95,
    })
    assert res_conflict["status"] == "ok"
    assert res_conflict.get("warning") == "epistemic_conflict"
    assert res_conflict["conflict_details"]["reason"] in ("exclusivity_collision", "exclusivity_violation", "semantic_contradiction")
    assert res_conflict["conflict_details"]["conflicting_key"] == "rule:obsidian:vault_write"

    res_prohib = await daemon._handle_set({
        "key": "rule:obsidian:disk_edit",
        "value": "zezwól na edycję bezpośrednio na dysku",
        "confidence": 0.8,
    })
    assert res_prohib["status"] == "ok"
    assert res_prohib.get("warning") == "epistemic_conflict"

    res_allowed = await daemon._handle_set({
        "key": "rule:obsidian:read_search",
        "value": "wyszukiwanie notatek przez MCP",
        "confidence": 1.0,
    })
    assert res_allowed["status"] == "ok"
    assert "warning" not in res_allowed


@pytest.mark.asyncio
async def test_benign_corpus_zero_false_positives(tmp_path: Path):
    """
    Finding F-1 Benign Corpus Gate:
    Verifies that conversational churn and generic words do not cause false-positive
    epistemic conflicts on ordinary benign knowledge assertions.
    """
    db_path = str(tmp_path / "test_atlas.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    daemon = AtlasDaemon(engine=engine)

    await daemon._handle_set({
        "key": "rule:obsidian:vault_write",
        "value": "EDYCJA vault: WYŁĄCZNIE przez MCP (nigdy bezpośrednio na dysku)",
        "confidence": 1.0,
        "metadata": {"source_type": "user_explicit"},
    })

    await daemon._handle_set({
        "key": "mnemosyne:mnemosyne_working_memory:[USER] No i ten:working_fact",
        "value": "Tylko pokazuje że o 760 znaków przekroczono limit i nie działa",
        "confidence": 0.5,
    })

    benign_corpus = [
        ("test:tecza", "Tęcza ma siedem barw"),
        ("test:barszcz", "Przepis na tradycyjny barszcz ukraiński"),
        ("test:benchmark", "Benchmark przepustowości sieci TCP i UDP"),
        ("test:python", "Python 3.12 wprowadza ulepszenia wydajności"),
        ("test:kawa", "Kawa po arabsku parzona jest z kardamonem"),
        ("test:kawa_z_mlekiem", "Kawa z mlekiem owsianym jest pyszna"),
        ("test:kuchnia", "Kuchnia włoska słynie z makaronów i pizzy"),
        ("test:astronomia", "Mars jest czwartą planetą od Słońca"),
        ("test:fizyka", "Prędkość światła w próżni wynosi około 300 000 km/s"),
        ("test:sport", "Mecz zakończył się remisem"),
        ("test:muzyka", "Fortepian ma 88 klawiszy"),
        ("test:chemia", "Woda składa się z dwóch atomów wodoru i jednego tlenu"),
    ]

    for key, val in benign_corpus:
        res = await daemon._handle_set({
            "key": key,
            "value": val,
            "confidence": 0.95,
        })
        assert res["status"] == "ok", f"Expected ok for {key}, got {res}"
        assert "warning" not in res or res["warning"] != "epistemic_conflict", (
            f"False positive epistemic conflict on benign assertion '{key}': {res.get('conflict_details')}"
        )


@pytest.mark.asyncio
async def test_epistemic_attribution_and_compatible_facts(tmp_path: Path):
    """Verifies exclusivity collisions correctly attribute to the competing rule rather than unrelated facts, and compatible facts do not collide."""
    db_path = str(tmp_path / "test_atlas.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    daemon = AtlasDaemon(engine=engine)

    await daemon._handle_set({
        "key": "rule:git:branch_strategy",
        "value": "Wdrażanie kodu WYŁĄCZNIE przez Pull Requesty (zakaz bezpośredniego push do main)",
        "confidence": 1.0,
        "metadata": {"source_type": "user_explicit"},
    })

    await daemon._handle_set({
        "key": "fact:git_push_delegated_agent",
        "value": "zakaz git push dla Delegated Agent",
        "confidence": 0.9,
    })

    res_conflict = await daemon._handle_set({
        "key": "rule:git:deploy_method",
        "value": "wdrażanie kodu wyłącznie przez bezpośredni commit do gałęzi main",
        "confidence": 0.95,
    })

    assert res_conflict["status"] == "ok"
    assert res_conflict.get("warning") == "epistemic_conflict"
    assert res_conflict["conflict_details"]["conflicting_key"] == "rule:git:branch_strategy"
    assert res_conflict["conflict_details"]["reason"] == "exclusivity_collision"

    res_neutral = await daemon._handle_set({
        "key": "fact:github_repo_main",
        "value": "Repozytorium na githubie używa brancha main i śledzi historię git",
        "confidence": 0.9,
    })
    assert res_neutral["status"] == "ok"
    assert "warning" not in res_neutral or res_neutral["warning"] != "epistemic_conflict"

