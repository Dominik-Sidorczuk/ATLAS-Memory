"""
Domain Test Suite: Procedural Sleep Consolidation & Trajectory Extraction.

Validates:
1. Real Trajectory Extraction:
   - Objective success extraction from transcript (tool_result content, exit codes)
   - Parameters and tool_name propagation
2. Sleep Watermark & State Transitions:
   - Stale-epoch watermark reset after daemon restart
   - Normal progression & empty buffer handling
3. Procedural Sleep Baker:
   - Dynamic population threshold for SOPs
   - Override preservation
   - Step extraction and tool_name resolution
"""
from __future__ import annotations

import pytest

from atlas_memory.hermes.trajectory_extractor import extract_trajectory
from atlas_memory.l1_working.trajectory_buffer import ArrowTrajectoryBuffer
from atlas_memory.l3_procedural.sleep_baker import (
    SleepBaker,
    compute_effective_min_frequency,
    sop_vector_entry,
)
from atlas_memory.server.atlas_daemon import _resolve_sleep_watermark

# ── Helpers ────────────────────────────────────────────────────────────────────

def _msg(role: str, **kw) -> dict:
    return {"role": role, **kw}


def _tool_result(content: str, call_id: str = "call_1") -> dict:
    return _msg("tool", tool_call_id=call_id, content=content)


def _assistant_with_calls(*calls) -> dict:
    return _msg("assistant", tool_calls=list(calls))


def _call(call_id: str, name: str, args: str = "{}") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TRAJECTORY EXTRACTION & OBJECTIVE SUCCESS
# ═══════════════════════════════════════════════════════════════════════════════

def test_extract_trajectory_success_step():
    messages = [
        _assistant_with_calls(_call("c1", "terminal_run", '{"cmd": "pwd"}')),
        _tool_result("stdout: /home/user\n", "c1"),
    ]
    steps = extract_trajectory(messages, session_id="s1")
    assert len(steps) == 1
    assert steps[0]["tool_name"] == "terminal_run"
    assert steps[0]["parameters"] == {"cmd": "pwd"}
    assert steps[0]["success"] is True
    assert steps[0]["error"] is None


def test_extract_trajectory_failure_from_exit_code():
    messages = [
        _assistant_with_calls(_call("c1", "terminal_run", '{"cmd": "false"}')),
        _tool_result("exit code 1\n", "c1"),
    ]
    steps = extract_trajectory(messages, session_id="s1")
    assert len(steps) == 1
    assert steps[0]["success"] is False
    assert "exit_code" in (steps[0]["error"] or "")


def test_extract_trajectory_multiple_steps_order_preserved():
    messages = [
        _assistant_with_calls(_call("c1", "file_read", '{"path": "a.txt"}')),
        _tool_result("hello\n", "c1"),
        _assistant_with_calls(_call("c2", "file_write", '{"path": "b.txt", "content": "hi"}')),
        _tool_result("ok\n", "c2"),
    ]
    steps = extract_trajectory(messages, session_id="s1")
    assert len(steps) == 2
    assert steps[0]["tool_name"] == "file_read"
    assert steps[1]["tool_name"] == "file_write"
    assert all(s["success"] for s in steps)


def test_trajectory_buffer_records_real_trajectory():
    buf = ArrowTrajectoryBuffer(state_dim=8)
    real_steps = [
        {"tool_name": "terminal_run", "parameters": {"cmd": "echo 1"}, "success": True},
        {"tool_name": "terminal_run", "parameters": {"cmd": "echo 2"}, "success": True},
    ]
    buf.record_real_trajectory(real_steps, session_id="test_sess")
    trajs = buf.get_session_trajectories("test_sess")
    assert len(trajs) == 1
    assert len(trajs[0]["steps"]) == 2
    assert trajs[0]["steps"][0]["tool_name"] == "terminal_run"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SLEEP WATERMARK & RESTART IDEMPOTENCE
# ═══════════════════════════════════════════════════════════════════════════════

def test_resolve_sleep_watermark_stale_epoch_resets():
    # restart: KV pamięta 100, świeży bufor RAM ma max step_id=2 -> stary znak, reset
    assert _resolve_sleep_watermark(buffer_max=2, watermark=100) == 0


def test_resolve_sleep_watermark_same_epoch_passthrough():
    # ta sama epoka: watermark <= buffer_max -> bez zmian
    assert _resolve_sleep_watermark(buffer_max=10, watermark=10) == 10
    assert _resolve_sleep_watermark(buffer_max=10, watermark=4) == 4


def test_resolve_sleep_watermark_empty_buffer_with_watermark_resets():
    assert _resolve_sleep_watermark(buffer_max=0, watermark=5) == 0
    assert _resolve_sleep_watermark(buffer_max=0, watermark=0) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PROCEDURAL BAKER & SOP POPULATION THRESHOLD
# ═══════════════════════════════════════════════════════════════════════════════

def test_dynamic_threshold_boundaries():
    assert compute_effective_min_frequency(1) == 2
    assert compute_effective_min_frequency(3) == 2
    assert compute_effective_min_frequency(4) == 2
    assert compute_effective_min_frequency(5) == 3
    assert compute_effective_min_frequency(8) == 3
    assert compute_effective_min_frequency(9) == 4
    assert compute_effective_min_frequency(17) == 5


def test_dynamic_threshold_override_exact_value():
    assert compute_effective_min_frequency(50, override=1) == 1
    assert compute_effective_min_frequency(2, override=3) == 3


def test_sop_vector_entry_structure():
    sop_dict = {
        "sop_id": "sop_123",
        "signature": "git_pull->pytest",
        "description": "Auto SOP",
        "success_rate": 0.95,
        "invocations_count": 5,
        "steps": [{"tool_name": "git_pull"}, {"tool_name": "pytest"}],
    }
    rec, embed_text = sop_vector_entry(sop_dict)
    assert rec.subject == "sop::sop_123"
    assert "git_pull" in embed_text
    assert rec.object == "git_pull->pytest"
    assert rec.metadata["type"] == "baked_sop"


def test_sleep_baker_step_extraction():
    baker = SleepBaker(min_frequency=1)
    raw_steps = [{"tool": "atlas_recall", "action": "atlas_recall"}]
    sig, steps = baker._extract_steps_and_sig(raw_steps)
    assert sig == "atlas_recall"
    assert steps[0].tool_name == "atlas_recall"


def test_weight_baker_procedure_distillation():
    from atlas_memory.l3_procedural.weight_baker import WeightBaker
    baker = WeightBaker(min_occurrence_threshold=2, min_success_rate=0.60)

    baker.log_episode_trajectory(["git_pull", "run_tests", "deploy"], success=True, context_tag="ci_cd")
    baker.log_episode_trajectory(["git_pull", "run_tests", "deploy"], success=True, context_tag="ci_cd")
    baker.log_episode_trajectory(["git_pull", "run_tests", "deploy"], success=False, context_tag="ci_cd")

    baked = baker.bake_procedures()
    assert len(baked) == 1
    assert baked[0].frequency == 3
    assert baked[0].success_rate == pytest.approx(2.0 / 3.0, rel=0.1)


@pytest.mark.asyncio
async def test_sleep_baker_bake_into_sop_and_kv():
    from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
    from atlas_memory.l3_procedural.sleep_baker import StandardProcedure, Step

    baker = SleepBaker()
    kv = VerifiedKVStore(db_path=":memory:")

    proc = StandardProcedure(
        procedure_id="sop_test_01",
        name="Standard Fix Procedure",
        signature="read_file->patch->pytest",
        steps=[
            Step(tool_name="read_file", params_pattern={"path": str}),
            Step(tool_name="patch", params_pattern={"path": str, "diff": str}),
            Step(tool_name="pytest", params_pattern={"args": list}),
        ],
        success_rate=0.95,
        invocations_count=10,
    )

    baked_data = await baker.bake_into_sop(proc, kv_store=kv)
    assert "sop_hash" in baked_data
    assert len(baked_data["sop_hash"]) == 64
    assert baked_data["sop_id"] == "sop_test_01"

    stored = await kv.get_state("sop::sop_test_01")
    assert stored is not None
    assert stored["value"]["sop_id"] == "sop_test_01"
    assert stored["confidence"] == 0.95

    await kv.close()


def test_sleep_baker_generic_tool_signature_disambiguation():
    """Verifies that generic tools (bash/terminal) disambiguate signature by command binary instead of collapsing."""
    baker = SleepBaker(min_frequency=2)
    sig_ls, _ = baker._extract_steps_and_sig([
        {"tool": "bash", "params": {"command": "ls -la /tmp"}},
        {"tool": "read_file", "params": {"path": "a.txt"}},
    ])
    sig_rm, _ = baker._extract_steps_and_sig([
        {"tool": "bash", "params": {"command": "rm -rf /tmp/x"}},
        {"tool": "read_file", "params": {"path": "c.txt"}},
    ])
    assert sig_ls != sig_rm
    assert "ls" in sig_ls and "rm" in sig_rm


