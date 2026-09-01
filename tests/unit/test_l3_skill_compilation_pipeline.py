import json
from pathlib import Path

import pytest

from atlas_memory.l3_procedural.skill_compiler import (
    SafetyViolationError,
    compile_and_register_sop,
    register_in_hermes_environment,
)
from atlas_memory.l3_procedural.sleep_baker import StandardProcedure, Step


def test_compile_and_register_sop_full_lifecycle(tmp_path: Path):
    sop = StandardProcedure(
        procedure_id="atlas_deploy_cluster",
        name="Deploy Kubernetes Cluster",
        steps=[
            Step(tool_name="validate_config", params_pattern={"env": "production"}),
            Step(tool_name="apply_manifests", params_pattern={"replicas": 3}),
        ],
        success_rate=0.95,
        invocations_count=10,
    )

    build_dir = tmp_path / "build"
    hermes_root = tmp_path / "hermes_skills"

    res = compile_and_register_sop(
        sop=sop,
        target_dir=build_dir,
        hermes_skills_root=hermes_root,
    )

    assert res["registered"] is True
    assert res["procedure_id"] == "atlas_deploy_cluster"
    assert res["steps_count"] == 2

    reg_path = Path(res["registered_path"])
    assert reg_path.exists()
    assert (reg_path / "SKILL.md").exists()
    assert (reg_path / "scripts" / "handler.py").exists()

    manifest_path = hermes_root / "atlas_procedural" / "skills_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert build_dir.name in manifest
    assert manifest[build_dir.name]["status"] == "active"


def test_register_in_hermes_environment_missing_files_raises(tmp_path: Path):
    empty_skill = tmp_path / "empty_skill"
    empty_skill.mkdir()
    hermes_root = tmp_path / "hermes"

    with pytest.raises(FileNotFoundError, match="SKILL.md missing"):
        register_in_hermes_environment(empty_skill, hermes_skills_root=hermes_root)

    (empty_skill / "SKILL.md").write_text("# Test", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="scripts/handler.py missing"):
        register_in_hermes_environment(empty_skill, hermes_skills_root=hermes_root)


def test_register_in_hermes_environment_blocks_ast_safety_violation(tmp_path: Path):
    unsafe_skill = tmp_path / "unsafe_skill"
    unsafe_scripts = unsafe_skill / "scripts"
    unsafe_scripts.mkdir(parents=True)

    (unsafe_skill / "SKILL.md").write_text("# Unsafe", encoding="utf-8")
    (unsafe_scripts / "handler.py").write_text(
        "import os\ndef run():\n    os.system('rm -rf /')\n",
        encoding="utf-8",
    )

    hermes_root = tmp_path / "hermes"
    with pytest.raises(SafetyViolationError) as exc_info:
        register_in_hermes_environment(unsafe_skill, hermes_skills_root=hermes_root)

    assert any("os.system" in v for v in exc_info.value.violations)


def test_skill_manifest_updates_on_recompilation(tmp_path: Path):
    sop = StandardProcedure(
        procedure_id="atlas_cache_warmup",
        name="Warm Up Redis Cache",
        steps=[Step(tool_name="ping_redis", params_pattern={})],
        success_rate=0.88,
        invocations_count=5,
    )
    hermes_root = tmp_path / "hermes"

    res1 = compile_and_register_sop(sop, hermes_skills_root=hermes_root)
    assert res1["registered"] is True

    # Recompile with new step
    sop.steps.append(Step(tool_name="preload_keys", params_pattern={"limit": 500}))
    sop.success_rate = 0.99
    res2 = compile_and_register_sop(sop, hermes_skills_root=hermes_root)
    assert res2["registered"] is True

    manifest_path = hermes_root / "atlas_procedural" / "skills_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert sop.procedure_id in manifest
