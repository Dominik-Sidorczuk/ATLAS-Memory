"""
Unit Tests for Telemetry Hooks (AtlasPluginHooks).

(V89: testy CacheHitMonitor usunięte razem z martwym pakietem telemetry.)
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

from atlas_memory.core.models import ConsolidationStats
from atlas_memory.hermes.plugin_hooks import AtlasPluginHooks, create_plugin_hooks
from atlas_memory.l3_procedural.sleep_baker import StandardProcedure, Step


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

    with patch("atlas_memory.l1_working.session_tracker.compile_sop_to_skill") as mock_compile:
        from atlas_memory.l1_working.session_tracker import HermesSessionHook
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
