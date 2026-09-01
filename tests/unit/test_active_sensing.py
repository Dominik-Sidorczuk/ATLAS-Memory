"""
Unit Tests for Active Sensing & Predictive Coding.
"""
from __future__ import annotations

import pytest

from atlas_memory.active.prediction_error import ActiveSensingEngine, PredictionCheck


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

