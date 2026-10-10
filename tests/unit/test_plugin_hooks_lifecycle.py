"""V106 ETAP 4 (FIX-C): Unit tests for plugin_hooks.py watchdog lifecycle (on_session_start / on_session_end)."""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from atlas_memory.hermes.atlas_provider import AtlasMemoryProvider
from atlas_memory.hermes.plugin_hooks import create_plugin_hooks, on_session_end, on_session_start


def test_a_on_session_start_spawns_daemon_when_socket_absent(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Scenario (a): Socket does not exist -> on_session_start calls provider._ensure_daemon_running(timeout_s=3.0)."""
    sock = tmp_path / "atlas.sock"
    provider = AtlasMemoryProvider(socket_path=str(sock))
    hooks = create_plugin_hooks(provider)

    with (
        caplog.at_level(logging.INFO),
        patch.object(provider, "_ensure_daemon_running", return_value=True) as mock_ensure,
    ):
        res = hooks.on_session_start("sess-106")
        assert res is True
        mock_ensure.assert_called_once_with(timeout_s=3.0)
        assert any("daemon auto-bootstrapped" in r.message for r in caplog.records)


def test_b_on_session_start_noop_when_socket_exists(tmp_path: Path) -> None:
    """Scenario (b): Socket exists -> on_session_start returns True without spawning."""
    sock = tmp_path / "atlas.sock"
    sock.write_text("alive", encoding="utf-8")
    provider = AtlasMemoryProvider(socket_path=str(sock))
    hooks = create_plugin_hooks(provider)

    with patch.object(provider, "_ensure_daemon_running") as mock_ensure:
        res = hooks.on_session_start("sess-106")
        assert res is True
        assert mock_ensure.call_count == 0


def test_c_on_session_start_fake_provider_logs_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Scenario (c): Fake Hermes-side provider lacking _ensure_daemon_running -> logs WARNING and returns False."""
    sock = tmp_path / "missing.sock"
    fake_provider = MagicMock(spec=["_socket_path"])
    fake_provider._socket_path = str(sock)

    with caplog.at_level(logging.WARNING):
        res = on_session_start(fake_provider, session_id="fake-sess")
        assert res is False
        assert any("lacks _ensure_daemon_running" in r.message for r in caplog.records)


def test_d_on_session_end_does_not_sigterm_daemon(tmp_path: Path) -> None:
    """Scenario (d): on_session_end cleans up session without sending SIGTERM to daemon."""
    sock = tmp_path / "atlas.sock"
    sock.write_text("alive", encoding="utf-8")
    provider = AtlasMemoryProvider(socket_path=str(sock))
    provider.on_session_end = MagicMock()  # type: ignore[method-assign]

    with patch("os.kill") as mock_kill:
        on_session_end(provider, messages=[{"session_id": "sess-106", "role": "user", "content": "hi"}])
        assert mock_kill.call_count == 0
        provider.on_session_end.assert_called_once()
        assert sock.exists()
