"""V123 (T1) — Closed-Loop Active Sensing & Predictive Coder Integration.

Contracts tested:
  1. Expectation resolution from registered `PredictionCheck` when `expected_value` / `tolerance` is omitted.
  2. `PredictionError` records are automatically appended to `ActiveSensingEngine.error_history`.
  3. Closed-loop writeback routes through `EpistemicWritePath.write_async` when available.
  4. RPC `_handle_active_sensing` supports registering expectations via `action="register"` or `register_expectation=True`.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from atlas_memory.active.prediction_error import ActiveSensingEngine, PredictionCheck
from atlas_memory.cognitive.predictive_coder import PredictiveCoder
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.server.atlas_daemon import AtlasDaemon


@pytest.mark.asyncio
async def test_t1_expectation_resolution_from_active_sensing():
    ase = ActiveSensingEngine()
    ase.register_expectation(
        PredictionCheck(
            check_id="chk_gpu_temp",
            target_entity="gpu_temp_celsius",
            expected_predicate="temperature",
            expected_value="65.0",
            tolerance=5.0,
        )
    )

    coder = PredictiveCoder(active_sensing=ase)

    # 1. Observation within tolerance: 68.0 vs registered 65.0 (tol=5.0) -> no error
    res_ok = await coder.execute_sensing({
        "target_entity": "gpu_temp_celsius",
        "observed_value": "68.0",
    })
    assert res_ok["status"] == "ok"
    assert res_ok["has_error"] is False
    assert res_ok["expected_value"] == "65.0"
    assert res_ok["tolerance"] == 5.0

    # 2. Observation outside tolerance: 85.0 vs registered 65.0 -> prediction error
    res_err = await coder.execute_sensing({
        "target_entity": "gpu_temp_celsius",
        "observed_value": "85.0",
    })
    assert res_err["status"] == "ok"
    assert res_err["has_error"] is True
    assert res_err["severity"] in ("MODERATE", "CRITICAL")
    assert res_err["expected_value"] == "65.0"

    # 3. Check error_history in ActiveSensingEngine
    assert len(ase.error_history) == 1
    last_err = ase.error_history[-1]
    assert last_err.target_entity == "gpu_temp_celsius"
    assert last_err.observed_value == "85.0"
    assert last_err.expected_value == "65.0"


@pytest.mark.asyncio
async def test_t1_writeback_through_epistemic_write_path():
    ase = ActiveSensingEngine()
    mock_engine = MagicMock()
    mock_write_path = MagicMock()
    mock_write_path.write_async = AsyncMock(return_value={"status": "ok", "key": "probe:db_pool:last_observed"})
    mock_engine.write_path = mock_write_path
    mock_engine.kv = MagicMock()

    coder = PredictiveCoder(engine=mock_engine, active_sensing=ase)

    res = await coder.execute_sensing({
        "target_entity": "db_pool",
        "observed_value": "exhausted",
        "expected_value": "healthy",
    })

    assert res["status"] == "ok"
    assert res["has_error"] is True
    assert res["world_model_updated"] is True
    assert mock_write_path.write_async.await_count == 1
    call_kwargs = mock_write_path.write_async.call_args.kwargs
    assert call_kwargs["key"] == "probe:db_pool:last_observed"
    assert call_kwargs["value"] == "exhausted"
    assert call_kwargs["reason"] == "active_sensing_writeback"


@pytest.mark.asyncio
async def test_t1_daemon_active_sensing_registration_rpc(tmp_path: Path):
    db_path = str(tmp_path / "test_atlas.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    daemon = AtlasDaemon(engine=engine)

    # Register via RPC
    reg_res = await daemon._handle_active_sensing({
        "action": "register",
        "target_entity": "api_latency",
        "expected_value": "45.0",
        "tolerance": 10.0,
    })
    assert reg_res["status"] == "ok"
    assert reg_res["registered"] is True
    assert reg_res["target_entity"] == "api_latency"

    # Probe without expected_value -> should auto-resolve from registration
    probe_res = await daemon._handle_active_sensing({
        "target_entity": "api_latency",
        "observed_value": "50.0",
    })
    assert probe_res["status"] == "ok"
    assert probe_res["has_error"] is False
    assert probe_res["expected_value"] == "45.0"
