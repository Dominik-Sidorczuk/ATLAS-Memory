"""V124 (F-040) T2 — session_tracker directly invokes genuine skill compiler with full error protection.

Verifies:
- prefix_guard re-exports compile_sop_to_skill from l3_procedural.skill_compiler;
- session_tracker invokes compile_sop_to_skill directly during auto-propose;
- session_tracker on_session_end path survives any compiler failure cleanly.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest

import atlas_memory.hermes.prefix_guard as prefix_guard
import atlas_memory.l1_working.session_tracker as session_tracker
from atlas_memory.core.models import ConsolidationStats
from atlas_memory.l1_working.session_tracker import HermesSessionHook


def test_prefix_guard_reexports_compiler() -> None:
    """prefix_guard re-exports genuine compile_sop_to_skill."""
    from atlas_memory.l3_procedural.skill_compiler import compile_sop_to_skill as genuine_compile
    assert prefix_guard.compile_sop_to_skill is genuine_compile


@pytest.mark.asyncio
async def test_session_tracker_invokes_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    """session_tracker auto-propose invokes compile_sop_to_skill directly."""
    calls: List[dict] = []

    def _fake_compile(sop: Any) -> object:
        calls.append({"id": sop.procedure_id})
        return Path("/tmp/fake_skill_dir")

    monkeypatch.setattr(session_tracker, "compile_sop_to_skill", _fake_compile)

    baked = {
        "sop_v124": SimpleNamespace(
            procedure_id="sop_v124",
            name="synthetic sop",
            success_rate=1.0,
            invocations_count=6,
        )
    }

    class _Baker:
        pass

    baker = _Baker()
    baker.baked_sops = baked

    engine = SimpleNamespace(
        write_path=None,
        hot_buffer=None,
        auditor=SimpleNamespace(sleep_baker=baker, run_sleep_cycle_consolidation=None),
        sleep_baker=baker,
        kv=None,
    )

    hook = HermesSessionHook(engine)
    hook.active_sessions["sess_v124"] = {"session_id": "sess_v124", "start_time": 0.0}
    hook._proposed_sop_ids.clear()

    stats = await hook.on_session_end("sess_v124", trigger_sleep_cycle=True)

    assert calls == [{"id": "sop_v124"}]
    assert isinstance(stats, ConsolidationStats)
    assert "sop_v124" in getattr(stats, "proposed_skills", [])


@pytest.mark.asyncio
async def test_session_tracker_end_path_survives_compiler_failure(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """session_tracker auto-propose tolerates compiler errors and completes cleanly."""
    baked = {
        "sop_v124": SimpleNamespace(
            procedure_id="sop_v124",
            name="synthetic sop",
            success_rate=1.0,
            invocations_count=6,
        )
    }

    class _Baker:
        pass

    baker = _Baker()
    baker.baked_sops = baked

    engine = SimpleNamespace(
        write_path=None,
        hot_buffer=None,
        auditor=SimpleNamespace(sleep_baker=baker, run_sleep_cycle_consolidation=None),
        sleep_baker=baker,
        kv=None,
    )

    def _boom(sop: Any) -> object:
        raise RuntimeError("simulated skill compiler failure")

    monkeypatch.setattr(session_tracker, "compile_sop_to_skill", _boom)

    hook = HermesSessionHook(engine)
    hook.active_sessions["sess_v124"] = {"session_id": "sess_v124", "start_time": 0.0}
    hook._proposed_sop_ids.clear()

    stats = await hook.on_session_end("sess_v124", trigger_sleep_cycle=True)

    assert isinstance(stats, ConsolidationStats)
    assert "sop_v124" in getattr(stats, "proposed_skills", [])
    assert any("Failed to compile skill for SOP sop_v124" in r.message for r in caplog.records)
