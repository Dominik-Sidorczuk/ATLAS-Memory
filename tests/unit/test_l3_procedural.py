"""
Unit Tests for Layer 3: Procedural Memory & Skill Compilation.
"""
from __future__ import annotations

import time

import pytest

from atlas_memory.l2_semantic import SymbolicGraphStore, VerifiedKVStore
from atlas_memory.l3_procedural.auditor import MemoryAuditor
from atlas_memory.l3_procedural.sleep_baker import SleepBaker, StandardProcedure, Step
from atlas_memory.l3_procedural.weight_baker import WeightBaker
from atlas_memory.models import EpistemicSource, MemoryRecord


@pytest.mark.asyncio
async def test_auditor_temporal_conflict_resolution():
    graph = SymbolicGraphStore()
    kv = VerifiedKVStore(db_path=":memory:")
    auditor = MemoryAuditor(graph, kv)

    t0 = time.time()
    t1 = t0 + 10.0

    # 1. Zapis starszego stanu z source_type=USER_EXPLICIT
    await kv.set_state("server_status", "running", metadata={"source_type": "user_explicit"})
    # Próba zapisu jeszcze starszego rekordu o tym samym priorytecie
    rec_stale = MemoryRecord(
        subject="server_status",
        predicate="state_value",
        object="stopped",
        timestamp=t0 - 100.0,
        source_type=EpistemicSource.USER_EXPLICIT,
        is_state_variable=True,
    )

    conflict, should_save = await auditor.check_and_resolve_conflict(rec_stale)
    assert conflict is not None
    assert "stale" in conflict.resolution_strategy
    assert should_save is False

    # 2. Zapis nowszego stanu
    rec_newer = MemoryRecord(
        subject="server_status",
        predicate="state_value",
        object="maintenance",
        timestamp=t1 + 100.0,
        source_type=EpistemicSource.USER_EXPLICIT,
        is_state_variable=True,
    )
    conflict2, should_save2 = await auditor.check_and_resolve_conflict(rec_newer)
    assert conflict2 is not None
    assert "timestamp" in conflict2.resolution_strategy
    assert should_save2 is True

    await kv.close()


@pytest.mark.asyncio
async def test_auditor_sleep_cycle_consolidation():
    graph = SymbolicGraphStore()
    kv = VerifiedKVStore(db_path=":memory:")
    auditor = MemoryAuditor(graph, kv)

    # Dodaj węzły, z których część zostanie osierocona
    await graph.add_record(MemoryRecord(subject="NodeA", predicate="links_to", object="NodeB"))
    await graph.remove_edge("NodeA", "links_to", "NodeB")

    stats = await auditor.run_sleep_cycle_consolidation()
    assert stats.orphaned_nodes_gc >= 2
    assert stats.duration_ms > 0.0

    await kv.close()


def test_weight_baker_procedure_distillation():
    baker = WeightBaker(min_occurrence_threshold=2, min_success_rate=0.60)

    # Rejestruj udane trajektorie (2 sukcesy, 1 porażka -> 66.7% >= 0.60)
    baker.log_episode_trajectory(["git_pull", "run_tests", "deploy"], success=True, context_tag="ci_cd")
    baker.log_episode_trajectory(["git_pull", "run_tests", "deploy"], success=True, context_tag="ci_cd")
    baker.log_episode_trajectory(["git_pull", "run_tests", "deploy"], success=False, context_tag="ci_cd")

    baked = baker.bake_procedures()
    assert len(baked) == 1
    assert baked[0].frequency == 3
    assert baked[0].success_rate == pytest.approx(2.0 / 3.0, rel=0.1)
    assert "ready_for_weight_baking" in baked[0].distilled_weights_meta["status"]
import pytest


def test_weight_baker_peft_lora_distillation():
    baker = WeightBaker(min_occurrence_threshold=2, min_success_rate=0.60, default_lora_rank=8)

    # Rejestruj udane trajektorie (2 sukcesy, 1 porażka -> 66.7% success rate >= 0.60)
    baker.log_episode_trajectory(["git_pull", "run_tests", "deploy"], success=True, context_tag="ci_cd")
    baker.log_episode_trajectory(["git_pull", "run_tests", "deploy"], success=True, context_tag="ci_cd")
    baker.log_episode_trajectory(["git_pull", "run_tests", "deploy"], success=False, context_tag="ci_cd")

    baked = baker.bake_procedures()
    assert len(baked) == 1
    proc = baked[0]
    assert proc.frequency == 3
    assert proc.success_rate == pytest.approx(2.0 / 3.0, rel=0.1)
    assert proc.lora_config["r"] == 8
    assert "target_modules" in proc.lora_config
    assert proc.distilled_weights_meta["status"] == "ready_for_weight_baking"

@pytest.mark.asyncio

async def test_sleep_baker_konsolidacja_sesji():
    """Test 1: Konsolidacja trajektorii i grupowanie powtarzalnych sekwencji akcji."""
    baker = SleepBaker(min_frequency=2, min_success_rate=0.7)

    trajektorie = [
        {
            "context": "bugfix",
            "success": True,
            "steps": [
                {"tool_name": "read_file", "params": {"path": "src/main.py"}},
                {"tool_name": "patch", "params": {"path": "src/main.py"}},
                {"tool_name": "pytest", "params": {"target": "tests/"}},
            ],
        },
        {
            "context": "bugfix",
            "success": True,
            "steps": [
                {"tool_name": "read_file", "params": {"path": "src/utils.py"}},
                {"tool_name": "patch", "params": {"path": "src/utils.py"}},
                {"tool_name": "pytest", "params": {"target": "tests/"}},
            ],
        },
        {
            "context": "refactor",
            "success": True,
            "steps": [
                {"tool_name": "search_files", "params": {"pattern": "TODO"}},
                {"tool_name": "read_file", "params": {"path": "src/todo.py"}},
            ],
        },
    ]

    sops = baker.konsolidacja_sesji(trajektorie)
    assert len(sops) == 1

    sop = sops[0]
    assert isinstance(sop, StandardProcedure)
    assert sop.signature == "read_file->patch->pytest"
    assert sop.invocations_count == 2
    assert sop.success_rate == 1.0
    assert len(sop.steps) == 3
    assert sop.steps[0].tool_name == "read_file"
    assert sop.steps[1].tool_name == "patch"
    assert sop.steps[2].tool_name == "pytest"


@pytest.mark.asyncio
async def test_sleep_baker_threshold_filtering():
    """Test 2: Filtrowanie trajektorii poniżej min_frequency oraz min_success_rate."""
    baker = SleepBaker(min_frequency=3, min_success_rate=0.8)

    trajektorie = [
        # Sekwencja A: 3x, ale 1 fail -> 2/3 = 0.66 < 0.8
        {"success": True, "steps": ["fetch", "parse"]},
        {"success": False, "steps": ["fetch", "parse"]},
        {"success": True, "steps": ["fetch", "parse"]},
        # Sekwencja B: 2x -> < min_frequency (3)
        {"success": True, "steps": ["compile", "run"]},
        {"success": True, "steps": ["compile", "run"]},
    ]

    sops = baker.konsolidacja_sesji(trajektorie)
    assert len(sops) == 0


@pytest.mark.asyncio
async def test_sleep_baker_bake_into_sop_and_kv():
    """Test 3: Wypiekanie do SOP z hashem i persistencja w VerifiedKVStore."""
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

    # Weryfikacja w KVStore
    stored = await kv.get_state("sop::sop_test_01")
    assert stored is not None
    assert stored["value"]["sop_id"] == "sop_test_01"
    assert stored["value"]["sop_hash"] == baked_data["sop_hash"]
    assert stored["confidence"] == 0.95

    # Weryfikacja integralności hash-chain po zapisie do KV
    is_valid, broken_seq = await kv.verify_chain_integrity()
    assert is_valid is True
    assert broken_seq == 0

    await kv.close()
import ast
from pathlib import Path

import pytest

from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.hermes.prefix_guard import HermesSessionHook
from atlas_memory.l3_procedural.skill_compiler import (
    ASTSafetyScanner,
    SafetyViolationError,
    compile_sop_to_skill,
)


def sample_sop() -> StandardProcedure:
    return StandardProcedure(
        procedure_id="sop_git_workflow",
        name="SOP: Git Commit Workflow",
        steps=[
            Step(tool_name="git_status", params_pattern={"path": "."}, expected_outcome="clean or dirty status"),
            Step(tool_name="git_add", params_pattern={"files": ["src/"]}, expected_outcome="staged files"),
            Step(tool_name="git_commit", params_pattern={"message": "feat: update"}, expected_outcome="commit hash"),
        ],
        success_rate=0.95,
        invocations_count=8,
        signature="git_status->git_add->git_commit",
    )


def test_compile_sop_creates_directory_structure(tmp_path: Path):
    sop = sample_sop()
    target_dir = tmp_path / "skill_git"
    res_path = compile_sop_to_skill(sop, target_dir=target_dir)

    assert res_path == target_dir
    assert target_dir.is_dir()
    assert (target_dir / "SKILL.md").is_file()
    assert (target_dir / "scripts").is_dir()
    assert (target_dir / "scripts" / "handler.py").is_file()


def test_skill_md_has_yaml_frontmatter(tmp_path: Path):
    sop = sample_sop()
    target_dir = tmp_path / "skill_frontmatter"
    compile_sop_to_skill(sop, target_dir=target_dir)

    content = (target_dir / "SKILL.md").read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert "name: sop_git_workflow\n" in content
    assert "description: SOP: Git Commit Workflow\n" in content
    assert "version: 1.0.0\n" in content
    assert "\n---\n" in content
    assert "## Steps" in content
    assert "## Usage" in content


def test_handler_py_syntax_valid(tmp_path: Path):
    sop = sample_sop()
    target_dir = tmp_path / "skill_syntax"
    compile_sop_to_skill(sop, target_dir=target_dir)

    code = (target_dir / "scripts" / "handler.py").read_text(encoding="utf-8")
    parsed = ast.parse(code)
    assert isinstance(parsed, ast.Module)


def test_handler_py_implements_steps(tmp_path: Path):
    sop = sample_sop()
    target_dir = tmp_path / "skill_methods"
    compile_sop_to_skill(sop, target_dir=target_dir)

    code = (target_dir / "scripts" / "handler.py").read_text(encoding="utf-8")
    assert "def step_1(" in code
    assert "def step_2(" in code
    assert "def step_3(" in code
    assert "def run(" in code


def test_ast_scanner_detects_eval():
    scanner = ASTSafetyScanner()
    violations = scanner.scan_code("eval('1+1')")
    assert len(violations) >= 1
    assert any("eval" in v for v in violations)


def test_ast_scanner_detects_os_system():
    scanner = ASTSafetyScanner()
    violations = scanner.scan_code("import os; os.system('ls')")
    assert len(violations) >= 1
    assert any("os.system" in v for v in violations)


def test_ast_scanner_detects_subprocess():
    scanner = ASTSafetyScanner()
    violations = scanner.scan_code("import subprocess; subprocess.run(['ls'])")
    assert len(violations) >= 1
    assert any("subprocess" in v for v in violations)


def test_ast_scanner_clean_code():
    scanner = ASTSafetyScanner()
    violations = scanner.scan_code("def foo(): return 42")
    assert violations == []


def test_compile_rejects_unsafe_code(tmp_path: Path):
    unsafe_sop = StandardProcedure(
        procedure_id="sop_malicious",
        name="SOP: Dangerous",
        steps=[
            Step(tool_name="eval", params_pattern={"code": "1+1"}),
        ],
        success_rate=0.9,
        invocations_count=5,
    )
    with pytest.raises(SafetyViolationError) as exc_info:
        compile_sop_to_skill(unsafe_sop, target_dir=tmp_path / "unsafe")

    assert len(exc_info.value.violations) > 0


def test_compile_with_target_dir(tmp_path: Path):
    sop = sample_sop()
    custom_target = tmp_path / "custom_location"
    out = compile_sop_to_skill(sop, target_dir=custom_target)
    assert out == custom_target
    assert (custom_target / "SKILL.md").exists()
    assert (custom_target / "scripts" / "handler.py").exists()


@pytest.mark.asyncio
async def test_prefix_guard_auto_propose_skills():
    engine = HybridMemoryEngine.create_default()
    hook = HermesSessionHook(memory_engine=engine)
    baker = SleepBaker()
    # Inject baked SOP into baker
    sop1 = StandardProcedure(
        procedure_id="sop_qualified",
        name="SOP: High Quality",
        steps=[Step(tool_name="test_tool", params_pattern={})],
        success_rate=0.95,
        invocations_count=6,
    )
    sop2 = StandardProcedure(
        procedure_id="sop_unqualified",
        name="SOP: Low Count",
        steps=[Step(tool_name="test_tool", params_pattern={})],
        success_rate=0.95,
        invocations_count=2,
    )
    baker.baked_sops["sop_qualified"] = sop1
    baker.baked_sops["sop_unqualified"] = sop2
    engine.sleep_baker = baker  # type: ignore

    stats = await hook.on_session_end(session_id="test_sess", trigger_sleep_cycle=True)
    assert "sop_qualified" in stats.proposed_skills
    assert "sop_unqualified" not in stats.proposed_skills
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas_memory.models import ConsolidationStats


@pytest.mark.asyncio
async def test_auto_propose_calls_compile():
    mock_engine = MagicMock()
    mock_engine.process_all_pending = AsyncMock()
    mock_engine.auditor.run_sleep_cycle_consolidation = AsyncMock(return_value=ConsolidationStats())

    qualifying_sop = StandardProcedure(
        procedure_id="sop_deploy_app",
        name="Deploy Application",
        steps=[Step(tool_name="terminal", params_pattern={"command": "build"})],
        success_rate=0.95,
        invocations_count=10,
    )
    unqualifying_sop = StandardProcedure(
        procedure_id="sop_flake",
        name="Flaky Procedure",
        steps=[Step(tool_name="terminal", params_pattern={"command": "test"})],
        success_rate=0.5,
        invocations_count=2,
    )

    mock_sleep_baker = MagicMock()
    mock_sleep_baker.baked_sops = {
        "sop_deploy_app": qualifying_sop,
        "sop_flake": unqualifying_sop,
    }
    mock_engine.sleep_baker = mock_sleep_baker

    hook = HermesSessionHook(memory_engine=mock_engine)

    with patch("atlas_memory.hermes.prefix_guard.compile_sop_to_skill") as mock_compile:
        stats = await hook.on_session_end(session_id="session_123", trigger_sleep_cycle=True)

        assert "sop_deploy_app" in stats.proposed_skills
        assert "sop_flake" not in stats.proposed_skills
        mock_compile.assert_called_once_with(qualifying_sop)


@pytest.mark.asyncio
async def test_auto_propose_skips_unsafe_sop(caplog):
    mock_engine = MagicMock()
    mock_engine.process_all_pending = AsyncMock()
    mock_engine.auditor.run_sleep_cycle_consolidation = AsyncMock(return_value=ConsolidationStats())

    unsafe_sop = StandardProcedure(
        procedure_id="sop_rm_rf",
        name="Dangerous Cleanup",
        steps=[Step(tool_name="terminal", params_pattern={"command": "rm -rf /"})],
        success_rate=0.99,
        invocations_count=8,
    )

    mock_sleep_baker = MagicMock()
    mock_sleep_baker.baked_sops = {
        "sop_rm_rf": unsafe_sop,
    }
    mock_engine.sleep_baker = mock_sleep_baker

    hook = HermesSessionHook(memory_engine=mock_engine)

    with patch("atlas_memory.hermes.prefix_guard.compile_sop_to_skill") as mock_compile:
        mock_compile.side_effect = SafetyViolationError(["Dangerous call detected"])
        stats = await hook.on_session_end(session_id="session_123", trigger_sleep_cycle=True)

        assert "sop_rm_rf" in stats.proposed_skills
        mock_compile.assert_called_once_with(unsafe_sop)
        # Verify it didn't crash and logged warning
        assert any("Unsafe SOP" in record.message for record in caplog.records)
