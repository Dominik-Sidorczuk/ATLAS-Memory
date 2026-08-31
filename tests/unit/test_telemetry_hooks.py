"""
Unit Tests for Telemetry, CacheHitMonitor & AtlasPluginHooks.
"""
from __future__ import annotations

from atlas_memory.telemetry.cache_monitor import CacheHitMonitor


def test_cache_hit_rate_calculation_100_turns():
    monitor = CacheHitMonitor()

    # Symulacja 100 tur: 90 trafień (hits), 10 pudeł (misses)
    prefix_h = "hash_stable_prefix_abc123"
    for _ in range(90):
        monitor.record_turn(model="deepseek-chat", prefix_hash=prefix_h, cached=True, tokens_in_prefix=1500)
    for _ in range(10):
        monitor.record_turn(model="deepseek-chat", prefix_hash=prefix_h, cached=False, tokens_in_prefix=1500)


    assert monitor.get_hit_rate("deepseek-chat") == 0.90
    assert monitor.get_hit_rate() == 0.90


def test_monthly_report_cost_and_token_savings():
    monitor = CacheHitMonitor()

    # 100 tur na deepseek-chat (90 hits)
    for _ in range(90):
        monitor.report_cache_hit(model="deepseek-chat", prefix_hash="p1", hit=True, tokens_saved=2000, prompt_tokens=2500)
    for _ in range(10):
        monitor.report_cache_hit(model="deepseek-chat", prefix_hash="p1", hit=False, tokens_saved=2000, prompt_tokens=2500)

    report = monitor.monthly_report()

    assert report["total_monitored_turns"] == 100
    assert report["overall_cache_hit_rate_pct"] == 90.0
    assert report["total_tokens_saved"] == 90 * 2000  # 180,000 tokenów
    assert report["total_usd_saved_estimate"] > 0.0
    assert "deepseek-chat" in report["by_model"]


def test_cache_monitor_reset():
    monitor = CacheHitMonitor()
    monitor.report_cache_hit(model="qwen-2.5-72b", prefix_hash="p2", hit=True, tokens_saved=1000)
    assert monitor.get_hit_rate() == 1.0

    monitor.reset()
    assert monitor.get_hit_rate() == 0.0
    assert monitor.monthly_report()["total_monitored_turns"] == 0

import logging
from unittest.mock import AsyncMock, MagicMock, patch

from atlas_memory.hermes.plugin_hooks import AtlasPluginHooks, create_plugin_hooks
from atlas_memory.l3_procedural.sleep_baker import StandardProcedure, Step
from atlas_memory.models import ConsolidationStats


def test_plugin_hooks_on_session_end_returns_none():
    mock_provider = MagicMock()
    mock_hook = MagicMock()
    mock_stats = ConsolidationStats(proposed_skills=["test_skill_1"])
    mock_hook.on_session_end = AsyncMock(return_value=mock_stats)

    hooks = AtlasPluginHooks(provider=mock_provider, session_hook=mock_hook)
    messages = [{"session_id": "sess_123"}, {}, {}]
    result = hooks.on_session_end(messages=messages)

    assert result is None
    mock_hook.on_session_end.assert_called_once_with(session_id="sess_123", trigger_sleep_cycle=True)


def test_plugin_hooks_catches_exceptions(caplog):
    mock_provider = MagicMock()
    mock_hook = MagicMock()
    mock_hook.on_session_end = AsyncMock(side_effect=RuntimeError("Database lock error"))

    hooks = AtlasPluginHooks(provider=mock_provider, session_hook=mock_hook)
    messages = [{"session_id": "sess_err"}]
    with caplog.at_level(logging.ERROR):
        result = hooks.on_session_end(messages=messages)

    assert result is None
    assert "Database lock error" in caplog.text


def test_plugin_hooks_proposed_skills_populated(caplog):
    # Setup mock SleepBaker with SOP having success_rate=0.95 and invocations=10
    sop = StandardProcedure(
        procedure_id="sop_code_deploy",
        name="Code Deployment Procedure",
        steps=[Step(tool_name="git pull"), Step(tool_name="pytest"), Step(tool_name="deploy")],
        success_rate=0.95,
        invocations_count=10,
    )
    
    mock_engine = MagicMock()
    mock_engine.process_all_pending = AsyncMock()
    mock_engine.auditor.run_sleep_cycle_consolidation = AsyncMock(return_value=ConsolidationStats())
    
    mock_sleep_baker = MagicMock()
    mock_sleep_baker.baked_sops = {"sop_code_deploy": sop}
    mock_engine.sleep_baker = mock_sleep_baker

    with patch("atlas_memory.hermes.prefix_guard.compile_sop_to_skill") as mock_compile:
        from atlas_memory.hermes.prefix_guard import HermesSessionHook
        session_hook = HermesSessionHook(mock_engine)
        
        hooks = AtlasPluginHooks(provider=MagicMock(), session_hook=session_hook)
        messages = [{"session_id": "sess_sops"}] + [{}] * 9
        with caplog.at_level(logging.INFO):
            result = hooks.on_session_end(messages=messages)

        assert result is None
        assert "sop_code_deploy" in caplog.text
        mock_compile.assert_called_once_with(sop)


def test_plugin_hooks_sleep_cycle_disabled():
    mock_provider = MagicMock()
    mock_hook = MagicMock()
    mock_stats = ConsolidationStats()
    mock_hook.on_session_end = AsyncMock(return_value=mock_stats)

    hooks = AtlasPluginHooks(provider=mock_provider, session_hook=mock_hook)
    messages = [{"session_id": "sess_no_sleep"}, {}]
    result = hooks.on_session_end(messages=messages, trigger_sleep_cycle=False)

    assert result is None
    mock_hook.on_session_end.assert_called_once_with(session_id="sess_no_sleep", trigger_sleep_cycle=False)


def test_plugin_hooks_turn_count_logged(caplog):
    mock_provider = MagicMock()
    hooks = create_plugin_hooks(mock_provider)

    messages = [{"session_id": "sess_log"}, {}, {}, {}, {}]
    with caplog.at_level(logging.INFO):
        hooks.on_session_end(messages=messages)

    assert "turn_count=5" in caplog.text
