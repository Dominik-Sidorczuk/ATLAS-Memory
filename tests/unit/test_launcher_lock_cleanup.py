"""V106 ETAP 5 (FIX-D): Unit tests for scripts/atlas_daemon_launcher.py lock cleanup, stale socket detection, and log rotation."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.atlas_daemon_launcher import (  # noqa: E402
    acquire_process_lock,
    check_singleton,
    rotate_daemon_log,
)


def test_a_dead_socket_connection_refused_removed_and_starts(tmp_path: Path) -> None:
    """Scenario (a): socket dead (ConnectionRefusedError) -> removed BEFORE acquire_process_lock, launcher acquires lock."""
    sock = tmp_path / "atlas.sock"
    pid_file = tmp_path / "atlas.pid"
    lock_file = tmp_path / "atlas.lock"
    sock.write_text("stale_socket", encoding="utf-8")

    mock_sock_instance = MagicMock()
    mock_sock_instance.connect.side_effect = ConnectionRefusedError("Connection refused")
    sock_existed_at_lock_time = {"val": True}
    orig_acquire = acquire_process_lock

    def spy_acquire(lp: Path):
        sock_existed_at_lock_time["val"] = sock.exists()
        return orig_acquire(lp)

    with (
        patch("socket.socket", return_value=mock_sock_instance),
        patch("scripts.atlas_daemon_launcher.acquire_process_lock", side_effect=spy_acquire),
    ):
        lock_obj = check_singleton(sock, pid_file, lock_file=lock_file)
        try:
            assert sock_existed_at_lock_time["val"] is False, "Stale socket must be removed BEFORE acquire_process_lock"
            assert not sock.exists(), "Stale socket must be unlinked on ConnectionRefusedError"
            assert lock_file.exists()
            assert lock_file.read_text(encoding="utf-8").strip() == str(os.getpid())
        finally:
            lock_obj.close()


def test_b_active_socket_exits_with_already_running(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Scenario (b): socket active -> check_singleton exits with 'daemon already running'."""
    sock = tmp_path / "atlas.sock"
    pid_file = tmp_path / "atlas.pid"
    lock_file = tmp_path / "atlas.lock"
    sock.write_text("active_socket", encoding="utf-8")

    mock_sock_instance = MagicMock()
    mock_sock_instance.connect.return_value = None  # Active socket accepts connect()

    with (
        patch("socket.socket", return_value=mock_sock_instance),
        patch(
            "atlas_memory.ipc.uds_client.send_uds_request_sync",
            return_value={"status": "ok"},
        ),
    ):
        with pytest.raises(SystemExit) as exc_info:
            check_singleton(sock, pid_file, lock_file=lock_file)
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "daemon already running" in out


def test_c_dead_pid_in_lock_file_cleaned_and_acquired(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Scenario (c): lock file exists with dead PID -> logs 'stale lock from dead PID <pid>, cleaning' and acquires lock."""
    lock_file = tmp_path / "atlas.lock"
    dead_pid = 99999999  # Non-existent PID
    lock_file.write_text(f"{dead_pid}\n", encoding="utf-8")

    lock_obj = acquire_process_lock(lock_file)
    try:
        out = capsys.readouterr().out
        assert f"stale lock from dead PID {dead_pid}, cleaning" in out
        assert lock_file.read_text(encoding="utf-8").strip() == str(os.getpid())
    finally:
        lock_obj.close()


def test_d_log_rotation_over_threshold(tmp_path: Path) -> None:
    """Scenario (d): daemon.log exceeding threshold is rotated to daemon.log.1."""
    log_file = tmp_path / "daemon.log"
    log_file.write_bytes(b"X" * 2048)

    rotated = rotate_daemon_log(log_file, max_bytes=1024)
    assert rotated is True
    assert not log_file.exists()
    assert (tmp_path / "daemon.log.1").exists()
    assert (tmp_path / "daemon.log.1").stat().st_size == 2048
