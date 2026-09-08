"""
Unit Tests for Active Sensing & Predictive Coding.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from atlas_memory.active.prediction_error import ActiveSensingEngine, PredictionCheck
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.server.atlas_daemon import AtlasDaemon


@pytest.mark.asyncio
async def test_expectation_registration_and_listing():
    engine = ActiveSensingEngine()
    check = PredictionCheck(
        check_id="check_nas_ping",
        target_entity="entity_nas_01",
        expected_predicate="ping_latency_ms",
        expected_value="3.5",
        tolerance=2.0,
    )
    engine.register_expectation(check)

    checks = engine.expectation_checks()
    assert len(checks) == 1
    assert checks[0].check_id == "check_nas_ping"


@pytest.mark.asyncio
async def test_numeric_tolerance_within_and_outside():
    engine = ActiveSensingEngine()
    engine.register_expectation(PredictionCheck(
        check_id="ping_check",
        target_entity="entity_nas_01",
        expected_predicate="ping_ms",
        expected_value="4.0",
        tolerance=2.0,
    ))

    # 1. 5.5 ms -> w granicach tolerancji (4.0 +/- 2.0) -> brak błędu
    err_ok = engine.detect_discrepancy("entity_nas_01", "ping_ms", "5.5")
    assert err_ok is None

    # 2. 12.0 ms -> przekroczenie tolerancji -> błąd predykcji
    err_fail = engine.detect_discrepancy("entity_nas_01", "ping_ms", "12.0")
    assert err_fail is not None
    assert err_fail.observed_value == "12.0"
    assert err_fail.expected_value == "4.0"
    assert err_fail.discrepancy_score > 0.0


@pytest.mark.asyncio
async def test_categorical_exact_mismatch():
    engine = ActiveSensingEngine()
    engine.register_expectation(PredictionCheck(
        check_id="service_status",
        target_entity="PostgreSQL",
        expected_predicate="status",
        expected_value="running",
    ))

    # Zgodne
    assert engine.detect_discrepancy("PostgreSQL", "status", "running") is None

    # Rozbieżne (awaria)
    err = engine.detect_discrepancy("PostgreSQL", "status", "stopped")
    assert err is not None
    assert err.severity == "CRITICAL"
    assert err.discrepancy_score == 1.0


@pytest.mark.asyncio
async def test_process_observation_updates_triple_supersede_zero_llm():
    engine = ActiveSensingEngine()
    engine.register_expectation(PredictionCheck(
        check_id="open_issues",
        target_entity="repo_loop",
        expected_predicate="open_issues_count",
        expected_value="5",
        tolerance=1.0,
    ))

    updated_triples = []

    async def mock_triple_add(subject, predicate, object_, confidence, source, supersede):
        updated_triples.append({
            "subject": subject,
            "predicate": predicate,
            "object": object_,
            "confidence": confidence,
            "source": source,
            "supersede": supersede,
        })

    # Obserwacja: 15 otwartych issue (duża anomalia)
    err = await engine.process_observation(
        observed_entity="repo_loop",
        observed_predicate="open_issues_count",
        observed_value=15,
        mnemosyne_triple_add_fn=mock_triple_add,
    )

    assert err is not None
    assert err.world_model_updated is True
    assert len(updated_triples) == 1
    assert updated_triples[0]["subject"] == "repo_loop"
    assert updated_triples[0]["object"] == "15"
    assert updated_triples[0]["supersede"] is True
    assert updated_triples[0]["source"] == "active_sensing_tool"

import pytest

from atlas_memory.active.shadow_worker import OmniRouteShadowWorker
from atlas_memory.orchestrator import MemoryOrchestrator


@pytest.mark.asyncio
async def test_successful_shadow_extraction_and_commit():
    orchestrator = MemoryOrchestrator()

    def mock_http(url, payload):
        return {
            "choices": [{
                "message": {
                    "content": '```json\n[{"subject": "TrueNAS", "predicate": "ip_address", "object": "192.168.1.50", "source_type": "user_explicit", "confidence": 1.0, "is_state_variable": true}]\n```'
                }
            }]
        }

    worker = OmniRouteShadowWorker(orchestrator=orchestrator, http_client_fn=mock_http)
    await worker.enqueue_turn("Mój serwer to TrueNAS o IP 192.168.1.50", "Zrozumiałem.")
    processed = await worker.process_all_pending()

    assert processed == 1
    assert orchestrator.stats["shadow_facts_extracted"] >= 1


@pytest.mark.asyncio
async def test_malformed_json_fallback_handling():
    orchestrator = MemoryOrchestrator()

    def mock_broken_http(url, payload):
        return {
            "choices": [{
                "message": {
                    "content": "To jest odpowiedź bez formatu JSON, ale serwer to 10.0.0.1"
                }
            }]
        }

    worker = OmniRouteShadowWorker(orchestrator=orchestrator, http_client_fn=mock_broken_http)
    await worker.enqueue_turn("Mój serwer to 10.0.0.1", "Ok")
    processed = await worker.process_all_pending()

    assert processed == 1
    # Powinno zadziałać dzięki fallbackowi do reguł


@pytest.mark.asyncio
async def test_retry_logic_on_http_failure():
    orchestrator = MemoryOrchestrator()
    call_count = 0

    def mock_flaky_http(url, payload):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionResetError("OmniRoute temp connection drop")
        return {
            "choices": [{
                "message": {
                    "content": '[{"subject": "ServiceGateway", "predicate": "port", "object": "8080", "source_type": "user_explicit", "confidence": 1.0, "is_state_variable": true}]'
                }
            }]
        }

    worker = OmniRouteShadowWorker(orchestrator=orchestrator, http_client_fn=mock_flaky_http, max_retries=3)
    await worker.enqueue_turn("Gateway port to 8080", "Zapisane")
    processed = await worker.process_all_pending()

    assert processed == 1
    assert call_count == 3


@pytest.mark.asyncio
async def test_timeout_fallback_graceful():
    orchestrator = MemoryOrchestrator()

    def mock_timeout_http(url, payload):
        raise TimeoutError("HTTP timeout 5.0s")

    worker = OmniRouteShadowWorker(orchestrator=orchestrator, http_client_fn=mock_timeout_http, max_retries=1)
    await worker.enqueue_turn("Mój NAS to 192.168.1.99", "OK")
    processed = await worker.process_all_pending()

    assert processed == 1


@pytest.mark.asyncio
async def test_queue_background_worker_start_stop():
    orchestrator = MemoryOrchestrator()

    def mock_fast_http(url, payload):
        return {"choices": [{"message": {"content": "[]"}}]}

    worker = OmniRouteShadowWorker(orchestrator=orchestrator, http_client_fn=mock_fast_http)
    task = worker.start()
    assert worker._is_running is True

    await worker.enqueue_turn("test 1", "resp 1")
    await worker.enqueue_turn("test 2", "resp 2")
    await worker.stop()

    assert worker._is_running is False
    assert task.done()


@pytest.mark.asyncio
async def test_v37_active_sensing_triggers_auto_anneal():
    """Test AtlasDaemon _handle_active_sensing spawns background causal graph annealing upon discrepancy."""
    mock_active = MagicMock()
    mock_active.register_expectation = MagicMock()

    from atlas_memory.active.prediction_error import PredictionError
    err = PredictionError(
        check_id="chk_srv_01",
        target_entity="srv_01",
        predicate="status",
        expected_value="running",
        observed_value="crashed",
        discrepancy_score=0.9,
        severity="CRITICAL",
    )
    mock_active.detect_discrepancy = MagicMock(return_value=err)

    mock_causal = MagicMock()
    mock_causal.recalibrate_graph_with_annealer = AsyncMock(return_value={"status": "ok", "updated_edges": 1})

    daemon = AtlasDaemon(
        active_sensing=mock_active,
        causal_engine=mock_causal,
    )

    res = await daemon._handle_active_sensing({
        "target_entity": "srv_01",
        "observed_predicate": "status",
        "observed_value": "crashed",
        "expected_value": "running",
    })

    assert res["status"] == "ok"
    assert res["has_error"] is True
    assert res["anneal_triggered"] is True

    await asyncio.sleep(0.01)
    mock_causal.recalibrate_graph_with_annealer.assert_called_once_with(target_entity="srv_01")


def test_f3_continuous_active_sensing_severity():
    """Finding F-3: Numeric discrepancies scale continuously instead of binary 1.0/CRITICAL."""
    ase = ActiveSensingEngine()
    check = PredictionCheck(
        check_id="chk_stats",
        target_entity="stats:kv_records",
        expected_predicate="count",
        expected_value="770",
    )
    ase.register_expectation(check)

    err = ase.detect_discrepancy("stats:kv_records", "count", "773")
    assert err is not None
    assert err.discrepancy_score < 0.01
    assert err.severity == "LOW"

    check2 = PredictionCheck(
        check_id="chk_large",
        target_entity="large_entity",
        expected_predicate="metric",
        expected_value="100",
    )
    ase.register_expectation(check2)
    err2 = ase.detect_discrepancy("large_entity", "metric", "10")
    assert err2 is not None
    assert err2.discrepancy_score >= 0.50
    assert err2.severity == "CRITICAL"


def test_f3_exact_match_returns_none():
    """Finding F-3 (AGI-3): Numeric exact match must return None and avoid false LOW discrepancy."""
    ase = ActiveSensingEngine()
    ase.register_expectation(
        PredictionCheck(
            check_id="pid_check_1",
            target_entity="daemon_pid",
            expected_predicate="is_running",
            expected_value="52313",
        )
    )

    res_exact = ase.detect_discrepancy("daemon_pid", "is_running", "52313")
    assert res_exact is None

    res_float = ase.detect_discrepancy("daemon_pid", "is_running", 52313.0)
    assert res_float is None

    res_diff = ase.detect_discrepancy("daemon_pid", "is_running", "52314")
    assert res_diff is not None
    assert res_diff.discrepancy_score > 0.0

    ase.register_expectation(
        PredictionCheck(
            check_id="tol_check_1",
            target_entity="latency_ms",
            expected_predicate="response_time",
            expected_value="100.0",
            tolerance=5.0,
        )
    )
    assert ase.detect_discrepancy("latency_ms", "response_time", 103.0) is None
    assert ase.detect_discrepancy("latency_ms", "response_time", 105.0) is None
    err = ase.detect_discrepancy("latency_ms", "response_time", 112.0)
    assert err is not None
    assert err.severity in ("WARNING", "CRITICAL")


@pytest.mark.asyncio
async def test_active_sensing_tolerance_and_gated_anneal(tmp_path: Path):
    """Verifies tolerance suppresses errors and sensory noise does not trigger auto-annealing."""
    db_path = str(tmp_path / "test_atlas.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    daemon = AtlasDaemon(engine=engine)

    res_tol = await daemon._handle_active_sensing({
        "target_entity": "agent_speed",
        "observed_value": "1480",
        "expected_value": "1500",
        "tolerance": 50.0,
    })
    assert res_tol["status"] == "ok"
    assert res_tol["has_error"] is False
    assert res_tol["anneal_triggered"] is False

    res_noise = await daemon._handle_active_sensing({
        "target_entity": "agent_speed",
        "observed_value": "1480",
        "expected_value": "1500",
    })
    assert res_noise["status"] == "ok"
    assert res_noise["has_error"] is True
    assert res_noise["prediction_error"]["severity"] == "LOW"
    assert res_noise["anneal_triggered"] is False

    res_anomaly = await daemon._handle_active_sensing({
        "target_entity": "agent_speed",
        "observed_value": "500",
        "expected_value": "1500",
    })
    assert res_anomaly["status"] == "ok"
    assert res_anomaly["has_error"] is True
    assert res_anomaly["prediction_error"]["severity"] == "CRITICAL"
    assert res_anomaly["anneal_triggered"] is True


