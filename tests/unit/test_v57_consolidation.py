"""
Unit tests for ATLAS V57 Architectural Consolidation.
- F1: clamp_conf causal semantics (negative, >1, None, string, NaN for annealer & energy_module)
- F2: Single clamp_conf definition across codebase (zero dead duplicate bodies)
- F3: Singleton flock concurrency race (two parallel launcher starts on tmp_path, 2nd exit != 0)
- F4: Daemon set and observation record_mental_transition (Arrow buffer & L0 step increments)
"""
from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from atlas_memory.causal.annealer import CausalAnnealer
from atlas_memory.causal.energy_module import EnergyModule
from atlas_memory.causal.models import CausalEdge
from atlas_memory.cognitive.models import clamp_conf
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.hermes.atlas_provider import clamp_conf as provider_clamp_conf
from atlas_memory.models import clamp_conf as re_exported_clamp_conf
from atlas_memory.server.atlas_daemon import AtlasDaemon


# ---------------------------------------------------------------------------
# F1: clamp_conf Causal Semantics (annealer & energy_module)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_f1_clamp_conf_causal_semantics() -> None:
    """Verifies clamp_conf behavior on boundary/pathological inputs in causal components."""
    # 1. Base clamp_conf semantics
    assert clamp_conf(-0.5) == 0.0
    assert clamp_conf(1.5) == 1.0
    assert clamp_conf(None) == 1.0
    assert clamp_conf("0.85") == pytest.approx(0.85)
    assert clamp_conf("invalid_str") == 1.0
    assert clamp_conf(float("nan")) == 1.0

    # 2. EnergyModule intrinsic_cost with out-of-bounds confidence
    energy = EnergyModule()

    cost_neg = await energy.intrinsic_cost("srv_a", "read", current_confidence=-0.5)
    assert 0.0 <= cost_neg <= 1.0
    assert math.isfinite(cost_neg)

    cost_high = await energy.intrinsic_cost("srv_a", "read", current_confidence=2.5)
    assert 0.0 <= cost_high <= 1.0
    assert math.isfinite(cost_high)

    cost_none = await energy.intrinsic_cost("srv_a", "read", current_confidence=None)
    assert 0.0 <= cost_none <= 1.0
    assert math.isfinite(cost_none)

    cost_nan = await energy.intrinsic_cost("srv_a", "read", current_confidence=float("nan"))
    assert 0.0 <= cost_nan <= 1.0
    assert math.isfinite(cost_nan)

    # 3. EnergyModule compute_edge_energy_and_gradient
    # model_construct bypasses Pydantic model validation to verify clamp_conf in compute_edge_energy_and_gradient
    edge_out_of_bounds = CausalEdge.model_construct(
        source="NodeX",
        predicate="depends_on",
        target="NodeY",
        confidence=1.5,
    )
    e_val, grad_val = await energy.compute_edge_energy_and_gradient(edge_out_of_bounds)
    assert math.isfinite(e_val)
    assert math.isfinite(grad_val)

    # Edge with negative confidence via model_construct
    edge_neg = CausalEdge.model_construct(
        source="NodeX",
        predicate="depends_on",
        target="NodeY",
        confidence=-0.5,
    )
    e_neg, grad_neg = await energy.compute_edge_energy_and_gradient(edge_neg)
    assert math.isfinite(e_neg)
    assert math.isfinite(grad_neg)

    # 4. CausalAnnealer with annealing step
    annealer = CausalAnnealer(energy_module=energy)
    test_edge = CausalEdge(
        source="Alpha",
        predicate="depends_on",
        target="Beta",
        confidence=0.8,
    )
    res = await annealer.run([test_edge], max_iter=2, T_init=0.5)
    assert len(res.final_edges) == 1
    assert 0.0 <= res.final_edges[0].confidence <= 1.0


# ---------------------------------------------------------------------------
# F2: Single clamp_conf Definition across Codebase
# ---------------------------------------------------------------------------
def test_f2_single_clamp_conf_definition() -> None:
    """Verifies that clamp_conf is defined in exactly one place and reused everywhere."""
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"

    pattern = "def clamp_conf"
    matches: list[str] = []

    for py_file in src_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if pattern in text:
            for line_no, line in enumerate(text.splitlines(), start=1):
                if pattern in line:
                    matches.append(f"{py_file.relative_to(repo_root)}:{line_no}")

    assert len(matches) == 1, f"Expected exactly 1 definition of clamp_conf, found: {matches}"
    assert "src/atlas_memory/cognitive/models.py" in matches[0]

    # In-memory identity checks: provider and models re-export must reference the identical function
    assert provider_clamp_conf is clamp_conf
    assert re_exported_clamp_conf is clamp_conf


# ---------------------------------------------------------------------------
# F3: Singleton Flock Concurrency Race
# ---------------------------------------------------------------------------
def test_f3_singleton_flock_concurrency_race(tmp_path: Path) -> None:
    """Verifies atomic mutual exclusion via flock when starting two daemon launchers concurrently."""
    repo_root = Path(__file__).resolve().parents[2]
    launcher_script = repo_root / "scripts" / "atlas_daemon_launcher.py"

    iso_sock = tmp_path / "iso_atlas.sock"
    iso_pid = tmp_path / "iso_atlas.pid"
    iso_lock = tmp_path / "iso_atlas.lock"

    env = dict(os.environ)
    env["ATLAS_SOCKET_PATH"] = str(iso_sock)
    env["ATLAS_PID_PATH"] = str(iso_pid)
    env["ATLAS_LOCK_PATH"] = str(iso_lock)
    env["PYTHONPATH"] = str(repo_root / "src")

    # Start first launcher process in background
    proc1 = subprocess.Popen(
        [sys.executable, str(launcher_script)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Wait up to 5s for proc1 to acquire flock and create lock/pid file
        start_t = time.time()
        running = False
        while time.time() - start_t < 5.0:
            if iso_lock.exists() and iso_lock.stat().st_size > 0:
                running = True
                break
            time.sleep(0.05)

        assert running, "Process 1 failed to acquire flock in time"

        # Attempt to launch second process with identical lock/socket environment
        res2 = subprocess.run(
            [sys.executable, str(launcher_script)],
            env=env,
            capture_output=True,
            text=True,
            timeout=5.0,
        )

        # Process 2 must immediately fail with non-zero code and message
        assert res2.returncode != 0, f"Expected non-zero exit code for proc2, got {res2.returncode}"
        combined_output = (res2.stdout + res2.stderr).lower()
        assert "daemon already running" in combined_output

        # Verify live socket ~/.hermes/atlas.sock was not touched
        home_sock = Path.home() / ".hermes" / "atlas.sock"
        if home_sock.exists():
            assert iso_sock.resolve() != home_sock.resolve()

    finally:
        # Cleanly shut down Process 1
        proc1.send_signal(signal.SIGINT)
        try:
            proc1.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc1.kill()
            proc1.wait(timeout=2.0)


# ---------------------------------------------------------------------------
# F4: Daemon Set & Observation record_mental_transition
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_f4_set_and_observation_record_mental_transition(tmp_path: Path) -> None:
    """Verifies that _handle_set and commit_observation record mental transitions to Arrow & L0."""
    db_path = str(tmp_path / "test.db")
    qdrant_path = str(tmp_path / "qdrant")
    kuzu_path = str(tmp_path / "kuzu")
    sock_path = tmp_path / "daemon.sock"
    pid_path = tmp_path / "daemon.pid"

    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=qdrant_path,
        kuzu_path=kuzu_path,
    )

    daemon = AtlasDaemon(
        socket_path=sock_path,
        pid_path=pid_path,
        engine=engine,
        graph_client=engine.graph,
        latent_buffer=engine.latent,
    )

    initial_arrow_count = len(engine.trajectory_buffer._steps)
    initial_ttt_steps = getattr(engine.ttt, "step_count", 0) if engine.ttt is not None else 0

    # 1. Test _handle_set (powers atlas_remember)
    set_params: dict[str, Any] = {
        "key": "user_backend_preference",
        "value": "use_v57_consolidation",
        "confidence": 0.95,
        "session_id": "test_session_v57",
    }
    set_res = await daemon._handle_set(set_params)
    assert set_res.get("status") == "ok" or set_res.get("success") is True or "key" in set_res

    # Verify Arrow trajectory increment
    assert len(engine.trajectory_buffer._steps) == initial_arrow_count + 1
    assert engine.trajectory_buffer._action_names[-1] == "remember"

    if engine.ttt is not None:
        assert getattr(engine.ttt, "step_count", 0) == initial_ttt_steps + 1

    # 2. Test commit_observation
    obs_params: dict[str, Any] = {
        "subject": "SystemCache",
        "predicate": "status_is",
        "object": "warmed_up",
        "session_id": "test_session_v57",
    }
    obs_res = await daemon._handle_commit_observation(obs_params)
    assert obs_res.get("status") == "ok"

    # Verify Arrow trajectory increment
    assert len(engine.trajectory_buffer._steps) == initial_arrow_count + 2
    assert engine.trajectory_buffer._action_names[-1] == "observe:status_is"

    if engine.ttt is not None:
        assert getattr(engine.ttt, "step_count", 0) == initial_ttt_steps + 2

    # Verify Arrow buffer export
    table = engine.trajectory_buffer.to_arrow_table()
    if table is not None:
        assert len(table) >= 2

    # Clean up engine worker
    await engine.stop_worker()
