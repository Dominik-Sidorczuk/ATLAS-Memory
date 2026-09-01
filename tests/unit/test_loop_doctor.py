"""
Unit tests for LoopDoctor diagnostic tool (Loop Engineering V23).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.loop_doctor import REPO_ROOT, LoopDoctor

# docs/baselines/ nie idzie na GitHub — te testy tylko lokalnie
_BASELINES_DIR = Path(REPO_ROOT) / "docs" / "baselines"
_NEED_BASELINES = any(_BASELINES_DIR.glob("benchmark_*.json"))


def test_loop_doctor_instantiation():
    """Verify LoopDoctor initializes cleanly."""
    doctor = LoopDoctor(root=REPO_ROOT, auto_fix=True)
    assert doctor.root == REPO_ROOT
    assert doctor.score_points == 100


def test_loop_doctor_baseline_json_check():
    """Verify LoopDoctor validates baseline JSON files in docs/baselines/."""
    if not _NEED_BASELINES:
        pytest.skip("docs/baselines/ not in public repo")
    doctor = LoopDoctor(root=REPO_ROOT)
    result = doctor.check_baseline_json_integrity()
    assert result is True
    assert any("baseline JSON" in p for p in doctor.passes)


def test_loop_doctor_doc_sync_check():
    """Verify LoopDoctor checks STATE.md and README.md synchronization."""
    doctor = LoopDoctor(root=REPO_ROOT)
    result = doctor.check_doc_sync_passport()
    assert result is True


def test_loop_doctor_code_anti_patterns():
    """Verify LoopDoctor detects zero unhandled pass stubs in production Python files."""
    doctor = LoopDoctor(root=REPO_ROOT)
    result = doctor.check_code_anti_patterns()
    assert result is True
    assert any("Zero unhandled 'pass'" in p for p in doctor.passes)


def test_loop_doctor_import_integrity():
    """Verify LoopDoctor verifies 100% clean import of all submodules."""
    doctor = LoopDoctor(root=REPO_ROOT)
    result = doctor.check_import_integrity()
    assert result is True
    assert any("import cleanly" in p for p in doctor.passes)


def test_loop_doctor_diagnose_score(monkeypatch):
    """Verify full diagnosis returns valid score and maturity tier."""
    if not _NEED_BASELINES:
        pytest.skip("docs/baselines/ not in public repo (score < 90 without them)")
    doctor = LoopDoctor(root=REPO_ROOT, auto_fix=True)
    monkeypatch.setattr(doctor, "check_git_status", lambda: True)
    report = doctor.diagnose()
    assert "loop_ready_score" in report
    assert report["loop_ready_score"] >= 90
    assert "L2+" in report["maturity_tier"]


