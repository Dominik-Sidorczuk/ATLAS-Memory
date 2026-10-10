"""V153 Omniverse AGI L3 Procedural Memory Observability & Architecture Tests.

Verifies:
1. Structured AGI Logger methods for L3:
   - [COG:PROCEDURAL] procedural distillation of trajectory to SOP.
   - [COG:SKILL] autonomous skill compilation with AST safety scan.
   - [COG:SAGA] reversible state monad transactions and compensatory rollbacks.
2. 10-phase Phased Boot Sequence:
   - [SYS:BOOT] [4/10] [L3] MemoryAuditor (Saga)
   - [SYS:BOOT] [5/10] [L3] Procedural Engine
3. Subsystem Initialization Matrix with L0, L1, L2, L3 completeness.
4. RPC procedural/status handler and get_stats telemetry integration.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from atlas_memory.core.agi_logger import (
    log_procedural_bake,
    log_saga_transaction,
    log_skill_compilation,
    log_subsystem_status_matrix,
)
from atlas_memory.server.atlas_daemon import AtlasDaemon, SubsystemLifecycleManager


def test_agi_logger_l3_methods(caplog: pytest.LogCaptureFixture) -> None:
    """Verifies that L3 procedural, skill, and saga logging emit canonical formatted records."""
    logger = logging.getLogger("test_l3_logger")
    logger.setLevel(logging.INFO)

    with caplog.at_level(logging.INFO, logger="test_l3_logger"):
        log_procedural_bake(
            logger,
            sop_id="sop:code_refactor_pipeline",
            steps_count=5,
            trigger="refactor_target_module",
            skill_path="~/.hermes/skills/baked-skills/sop_refactor",
            latency_ms=42.5,
        )
        log_skill_compilation(
            logger,
            skill_name="code_refactor_pipeline",
            ast_status="passed",
            violations_count=0,
            target_path="~/.hermes/skills/baked-skills/sop_refactor/SKILL.md",
        )
        log_saga_transaction(
            logger,
            tx_id="tx_178942000",
            action="atomic_write",
            steps_applied=["intent", "graph", "kv", "vector"],
            status="committed",
        )
        log_saga_transaction(
            logger,
            tx_id="tx_178942001",
            action="compensatory_rollback",
            steps_applied=["vector", "kv", "graph"],
            status="rolled_back",
            reason="qdrant_timeout",
        )

    output = "\n".join(rec.message for rec in caplog.records)
    assert "[COG:PROCEDURAL] baked SOP 'sop:code_refactor_pipeline' (5 steps, trigger='refactor_target_module') -> skill: ~/.hermes/skills/baked-skills/sop_refactor in 42.5ms" in output
    assert "[COG:SKILL] compiled skill 'code_refactor_pipeline' | AST safety: passed (violations=0) -> ~/.hermes/skills/baked-skills/sop_refactor/SKILL.md" in output
    assert "[COG:SAGA] tx='tx_178942000' atomic_write status=committed | steps: [intent,graph,kv,vector]" in output
    assert "[COG:SAGA] tx='tx_178942001' compensatory_rollback status=rolled_back | steps: [vector,kv,graph] (reason='qdrant_timeout')" in output


def test_subsystem_status_matrix_with_l3(caplog: pytest.LogCaptureFixture) -> None:
    """Verifies that Subsystem Initialization Matrix renders L0, L1, L2, L3 completely."""
    logger = logging.getLogger("test_matrix_l3")
    logger.setLevel(logging.INFO)

    rows = [
        ("[L0] TTT Engine", "READY", "Steps: 55 restored"),
        ("[L1] HotBuffer (Working)", "READY", "Seeded: 1 records"),
        ("[L2] Kùzu Multigraph", "READY", "Nodes: 1832 | Edges: 2121"),
        ("[L2] Qdrant Vector Engine", "READY", "Dim: 512 (onnx68 / poldense68)"),
        ("[L3] Memory Auditor", "ACTIVE", "Saga Monad: ACTIVE | Rollback: verified | Decay: enabled"),
        ("[L3] Procedural Consol", "READY", "Stream: Arrow/Parquet | SOPs: 2 | AST Guard: strict"),
        ("[CR] HippoRAG 2 PPR", "WARMED", "Warmup completed (damping=0.85)"),
        ("[EG] Epistemic Guard", "ACTIVE", "Conflict Policy: STRICT"),
        ("[MB] Mnemosyne Bridge", "READY", "Sync: active"),
        ("[SH] OmniRoute Shadow", "DISPATCHING", "Queue max: 1000 | Model: auto/best-fast"),
        ("[IPC] UDS Socket Server", "LISTENING", "Path: ~/.hermes/atlas.sock | MaxMsg: 16MB"),
    ]

    with caplog.at_level(logging.INFO, logger="test_matrix_l3"):
        log_subsystem_status_matrix(
            log=logger,
            rows=rows,
            total_modules=10,
            active_count=10,
            optional_disabled=0,
        )

    output = "\n".join(rec.message for rec in caplog.records)
    assert "[L3] Memory Auditor       | ACTIVE      | Saga Monad: ACTIVE | Rollback: verified | Decay: enabled" in output
    assert "[L3] Procedural Consol    | READY       | Stream: Arrow/Parquet | SOPs: 2 | AST Guard: strict" in output
    assert "CORE SYSTEM STATUS: ALL MANDATORY SUBSYSTEMS GREEN (10/10 ONLINE)" in output


def test_lifecycle_10_phases_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Verifies that SubsystemLifecycleManager transitions through 10 phases with L3 tags."""
    logger = logging.getLogger("test_boot_lifecycle")
    logger.setLevel(logging.INFO)
    lifecycle = SubsystemLifecycleManager(logger)

    with caplog.at_level(logging.INFO, logger="test_boot_lifecycle"):
        lifecycle.transition(1, 10, "[L2]", "VerifiedKVStore", "READY", "SHA-256 verified", duration_ms=2.0)
        lifecycle.transition(2, 10, "[L2]", "Kùzu Multigraph", "READY", "Nodes: 1832", duration_ms=10.0)
        lifecycle.transition(3, 10, "[L2]", "Qdrant Vector Engine", "READY", "Dim: 512", duration_ms=5.0)
        lifecycle.transition(4, 10, "[L3]", "MemoryAuditor (Saga)", "ACTIVE", "Monad: reversible", duration_ms=1.5)
        lifecycle.transition(5, 10, "[L3]", "Procedural Engine", "READY", "Stream: Arrow/Parquet | SOPs: 2", duration_ms=3.2)
        lifecycle.transition(6, 10, "[CR]", "HippoRAG 2 PPR", "WARMED", "Warmup completed", duration_ms=7.0)
        lifecycle.transition(7, 10, "[L1]", "HotBuffer (Working)", "READY", "Seeded: 1", duration_ms=0.5)
        lifecycle.transition(8, 10, "[EG]", "Epistemic Guard", "ACTIVE", "Conflict: STRICT", duration_ms=0.2)
        lifecycle.transition(9, 10, "[MB]", "Mnemosyne Bridge", "READY", "Sync: active", duration_ms=15.0)
        lifecycle.transition(10, 10, "[SH]", "OmniRoute Shadow", "DISPATCHING", "Queue max: 1000", duration_ms=0.1)

    output = "\n".join(rec.message for rec in caplog.records)
    assert "[SYS:BOOT] [1/10] [L2] VerifiedKVStore" in output
    assert "[SYS:BOOT] [4/10] [L3] MemoryAuditor (Saga): INITIALIZING -> HEALTHCHECK -> ACTIVE [Monad: reversible] (2ms)" in output
    assert "[SYS:BOOT] [5/10] [L3] Procedural Engine: INITIALIZING -> HEALTHCHECK -> READY [Stream: Arrow/Parquet | SOPs: 2] (3ms)" in output
    assert "[SYS:BOOT] [10/10] [SH] OmniRoute Shadow" in output


@pytest.mark.asyncio
async def test_procedural_status_rpc_handler(tmp_path: Path) -> None:
    """Verifies that procedural/status RPC returns full L3 diagnostic state."""
    daemon = AtlasDaemon(
        socket_path=tmp_path / "test_proc.sock",
        pid_path=tmp_path / "test_proc.pid",
    )
    mock_engine = MagicMock()
    mock_auditor = MagicMock()
    mock_auditor.decay_engine = MagicMock()
    mock_engine.auditor = mock_auditor
    mock_traj_buf = MagicMock()
    mock_traj_buf.trajectories = [{"sop_id": "test_1"}, {"sop_id": "test_2"}]
    mock_engine.trajectory_buffer = mock_traj_buf
    mock_kv = MagicMock()
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = [("sop:code_refactor", {}), ("sop:data_pipeline", {})]
    mock_kv.ensure_connection.return_value = mock_conn
    mock_engine.kv = mock_kv

    daemon.engine = mock_engine

    res = await daemon._handle_procedural_status({})
    assert res["status"] == "ok"
    assert res["layer"] == "L3"
    assert res["sops_count"] == 2
    assert "sop:code_refactor" in res["sops"]
    assert res["trajectories_buffered"] == 2
    assert res["saga_monad"] == "active"
    assert res["decay_engine"] == "active"
    assert res["ast_safety_guard"] == "strict"
