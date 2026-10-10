"""V106 ETAP 2 (FIX-A): Unit tests for AtlasMemoryProvider auto-spawn in initialize()."""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from atlas_memory.hermes.atlas_provider import AtlasMemoryProvider


def test_a_healthy_daemon_does_not_spawn(tmp_path: Path) -> None:
    """Scenario (a): daemon running & healthy -> initialize() does NOT call subprocess.Popen."""
    sock = tmp_path / "atlas.sock"
    sock.write_text("active", encoding="utf-8")
    provider = AtlasMemoryProvider(socket_path=str(sock))

    with (
        patch("subprocess.Popen") as mock_popen,
        patch(
            "atlas_memory.hermes.atlas_provider.send_uds_request_sync",
            return_value={"status": "ok", "dual_node_coverage": 1.0},
        ) as mock_rpc,
    ):
        provider.initialize("session-a")
        assert mock_popen.call_count == 0
        mock_rpc.assert_called_once_with(str(sock), "get_stats", params={})


def test_b_stale_socket_removed_and_respawned(tmp_path: Path) -> None:
    """Scenario (b): socket exists but daemon dead (get_stats returns None) -> unlink stale socket & spawn."""
    sock = tmp_path / "atlas.sock"
    sock.write_text("stale", encoding="utf-8")
    provider = AtlasMemoryProvider(socket_path=str(sock))

    call_idx = {"n": 0}

    def fake_send_uds(sock_str: str, method: str, params: dict | None = None) -> dict | None:
        call_idx["n"] += 1
        if call_idx["n"] == 1:
            return None  # Stale socket timeout on initial check
        return {"status": "ok", "dual_node_coverage": 1.0}

    def fake_popen(*args, **kwargs):
        # Simulate newly spawned daemon creating the socket file
        sock.write_text("new_daemon", encoding="utf-8")
        return MagicMock()

    with (
        patch("subprocess.Popen", side_effect=fake_popen) as mock_popen,
        patch("atlas_memory.hermes.atlas_provider.send_uds_request_sync", side_effect=fake_send_uds),
    ):
        provider.initialize("session-b")
        assert mock_popen.call_count == 1
        assert sock.exists()
        assert sock.read_text(encoding="utf-8") == "new_daemon"


def test_c_missing_socket_spawns_and_waits_for_ready(tmp_path: Path) -> None:
    """Scenario (c): socket does not exist -> spawns new daemon, polls until socket + get_stats ready."""
    sock = tmp_path / "atlas.sock"
    assert not sock.exists()
    provider = AtlasMemoryProvider(socket_path=str(sock))

    def fake_popen(*args, **kwargs):
        sock.write_text("spawned", encoding="utf-8")
        return MagicMock()

    with (
        patch("subprocess.Popen", side_effect=fake_popen) as mock_popen,
        patch(
            "atlas_memory.hermes.atlas_provider.send_uds_request_sync",
            return_value={"status": "ok", "kv_records": 805},
        ),
    ):
        provider.initialize("session-c")
        assert mock_popen.call_count == 1
        assert sock.exists()


def test_d_spawn_oserror_logs_warning_and_falls_back(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Scenario (d): subprocess.Popen raises OSError -> WARNING logged + in-process fallback."""
    sock = tmp_path / "atlas.sock"
    provider = AtlasMemoryProvider(socket_path=str(sock))

    with (
        caplog.at_level(logging.WARNING),
        patch("subprocess.Popen", side_effect=OSError("spawn permission denied")) as mock_popen,
    ):
        provider.initialize("session-d")
        assert mock_popen.call_count == 1
        assert any("falling back to in-process" in rec.message for rec in caplog.records)


def test_e_spawn_timeout_logs_warning_and_falls_back(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Scenario (e): socket never appears within timeout -> WARNING logged + in-process fallback."""
    sock = tmp_path / "atlas.sock"
    provider = AtlasMemoryProvider(socket_path=str(sock))

    with (
        caplog.at_level(logging.WARNING),
        patch("subprocess.Popen", return_value=MagicMock()) as mock_popen,
        patch("time.sleep", return_value=None),
    ):
        ok = provider._ensure_daemon_running(timeout_s=0.05)
        assert ok is False
        assert mock_popen.call_count == 1
        assert any("daemon spawn timeout, falling back to in-process" in rec.message for rec in caplog.records)


def test_provider_loads_without_igraph_or_networkx_in_python311(monkeypatch):
    """Ensure AtlasMemoryProvider imports cleanly in Hermes Python 3.11 where igraph/networkx are absent."""
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "igraph", None)
    mod = importlib.reload(importlib.import_module("atlas_memory.hermes.atlas_provider"))
    provider = mod.AtlasMemoryProvider()
    assert provider is not None
