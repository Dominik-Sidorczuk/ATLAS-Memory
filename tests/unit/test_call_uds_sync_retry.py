"""V106 ETAP 3 (FIX-B): Unit tests for _call_uds_sync auto-spawn and 1-retry backoff."""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from atlas_memory.hermes.atlas_provider import AtlasMemoryProvider, AtlasTransportError


def test_a_missing_socket_triggers_auto_spawn_and_succeeds(tmp_path: Path) -> None:
    """Scenario (a): Path(sock).exists() == False -> calls _ensure_daemon_running() and succeeds."""
    sock = tmp_path / "atlas.sock"
    provider = AtlasMemoryProvider(socket_path=str(sock))

    def fake_ensure(*args, **kwargs) -> bool:
        sock.write_text("spawned", encoding="utf-8")
        return True

    with (
        patch.object(provider, "_ensure_daemon_running", side_effect=fake_ensure) as mock_ensure,
        patch(
            "atlas_memory.hermes.atlas_provider.send_uds_request_sync",
            return_value={"status": "ok", "value": "remembered_fact"},
        ) as mock_send,
    ):
        res = provider._call_uds_sync("get", {"key": "k1"})
        assert mock_ensure.call_count == 1
        assert res == {"status": "ok", "value": "remembered_fact"}
        mock_send.assert_called_once_with(str(sock), "get", params={"key": "k1"})


def test_b_transport_error_retries_once_after_respawn(tmp_path: Path) -> None:
    """Scenario (b): First send_uds_request_sync raises AtlasTransportError -> respawns, waits 1s, retries and succeeds."""
    sock = tmp_path / "atlas.sock"
    sock.write_text("alive", encoding="utf-8")
    provider = AtlasMemoryProvider(socket_path=str(sock))

    attempts = {"count": 0}

    def flaky_send(sock_str: str, method: str, params: dict | None = None) -> dict:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise AtlasTransportError("broken pipe on first call")
        return {"status": "ok", "retried": True}

    with (
        patch.object(provider, "_ensure_daemon_running", return_value=True) as mock_ensure,
        patch("atlas_memory.hermes.atlas_provider.send_uds_request_sync", side_effect=flaky_send),
        patch("time.sleep") as mock_sleep,
    ):
        res = provider._call_uds_sync("prefetch", {"query": "test"})
        assert attempts["count"] == 2
        assert mock_ensure.call_count == 1
        mock_sleep.assert_called_once_with(1.0)
        assert res == {"status": "ok", "retried": True}


def test_c_double_transport_error_logs_warning_and_returns_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Scenario (c): Both initial call and retry raise AtlasTransportError -> WARNING logged + returns None."""
    sock = tmp_path / "atlas.sock"
    sock.write_text("alive", encoding="utf-8")
    provider = AtlasMemoryProvider(socket_path=str(sock))

    with (
        caplog.at_level(logging.WARNING),
        patch.object(provider, "_ensure_daemon_running", return_value=True) as mock_ensure,
        patch(
            "atlas_memory.hermes.atlas_provider.send_uds_request_sync",
            side_effect=AtlasTransportError("connection refused twice"),
        ),
        patch("time.sleep"),
    ):
        res = provider._call_uds_sync("set", {"key": "k", "value": "v"})
        assert res is None
        assert mock_ensure.call_count == 1
        assert any("falling back to in-process" in rec.message for rec in caplog.records)
