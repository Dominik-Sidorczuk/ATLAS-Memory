#!/usr/bin/env python3
"""Launcher for AtlasDaemon with systemd-style process management."""
from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Optional

from atlas_memory.server.atlas_daemon import AtlasDaemon

_LOCK_FILE_OBJ: Optional[Any] = None


def acquire_process_lock(lock_path: Path) -> Any:
    """Atomic mutual exclusion using kernel file lock (flock)."""
    global _LOCK_FILE_OBJ
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        print("daemon already running")
        sys.exit(1)
    try:
        f.seek(0)
        f.truncate(0)
        f.write(f"{os.getpid()}\n")
        f.flush()
    except Exception:
        pass
    _LOCK_FILE_OBJ = f
    return f


class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_daemon_logging(log_path: Optional[Path] = None) -> None:
    log_file = log_path or (Path.home() / ".hermes" / "atlas" / "daemon.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = FlushFileHandler(str(log_file), mode="a", encoding="utf-8")
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not any(isinstance(h, FlushFileHandler) for h in root_logger.handlers):
        root_logger.addHandler(handler)


async def run_daemon(sock: Path, pid_file: Path, lock_obj: Optional[Any] = None) -> None:
    if hasattr(signal, "SIGHUP"):
        try:
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
        except (ValueError, OSError):
            pass
    setup_daemon_logging()
    daemon = AtlasDaemon.create_default(socket_path=sock, pid_path=pid_file)
    if lock_obj is not None:
        daemon._process_lock_handle = lock_obj
    await daemon.start()
    daemon.register_signal_handlers()
    logging.info("AtlasDaemon started on %s (PID: %d)", sock, os.getpid())
    print(f"AtlasDaemon started: {sock}")
    try:
        while daemon._serving:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await daemon.stop()


def check_singleton(sock: Path, pid_file: Path, lock_file: Optional[Path] = None) -> Any:
    """Sprawdza czy daemon już działa. Jeśli tak -> exit code != 0 z komunikatem."""
    if lock_file is None:
        lock_file = Path(os.environ.get("ATLAS_LOCK_PATH", str(pid_file.with_suffix(".lock"))))

    # 1. Atomic flock exclusion
    lock_obj = acquire_process_lock(lock_file)

    # 2. Defense in depth: Check alive PID & UDS ping
    running_pid: Optional[int] = None
    if pid_file.exists():
        try:
            pid_str = pid_file.read_text(encoding="utf-8").strip()
            if pid_str:
                val = int(pid_str)
                os.kill(val, 0)
                running_pid = val
        except (OSError, ValueError):
            running_pid = None

    if sock.exists():
        try:
            from atlas_memory.server.client import send_uds_request_sync
            res = send_uds_request_sync(sock, "ping", timeout=0.2)
            if isinstance(res, dict) and res.get("status") == "ok":
                pid_info = f" (pid {running_pid})" if running_pid else ""
                print(f"daemon already running{pid_info}")
                sys.exit(1)
        except Exception:
            pass

    if running_pid is not None:
        print(f"daemon already running (pid {running_pid})")
        sys.exit(1)

    # Clean up stale files if no alive process
    if sock.exists():
        try:
            print(f"Removing stale socket file: {sock}")
            sock.unlink()
        except OSError:
            pass

    if pid_file.exists():
        try:
            pid_file.unlink()
        except OSError:
            pass

    return lock_obj


def main() -> None:
    if hasattr(signal, "SIGHUP"):
        try:
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
        except (ValueError, OSError):
            pass

    sock = Path(os.environ.get("ATLAS_SOCKET_PATH", str(Path.home() / ".hermes" / "atlas.sock")))
    pid_file = Path(os.environ.get("ATLAS_PID_PATH", str(Path.home() / ".hermes" / "atlas.pid")))

    # Single-instance enforcement (D-E singleton check with atomic flock)
    lock_obj = check_singleton(sock, pid_file)

    if "--daemon" in sys.argv or "--detach" in sys.argv or "-d" in sys.argv:
        if hasattr(os, "fork"):
            pid = os.fork()
            if pid > 0:
                print(f"AtlasDaemon spawned in background (PID: {pid})")
                sys.exit(0)
            os.setsid()
            pid2 = os.fork()
            if pid2 > 0:
                sys.exit(0)
            sys.stdout.flush()
            sys.stderr.flush()
            with open(os.devnull, "wb+", buffering=0) as devnull:
                os.dup2(devnull.fileno(), sys.stdin.fileno())
                os.dup2(devnull.fileno(), sys.stdout.fileno())
                os.dup2(devnull.fileno(), sys.stderr.fileno())

    try:
        asyncio.run(run_daemon(sock, pid_file, lock_obj=lock_obj))
    except KeyboardInterrupt:
        print("\nAtlasDaemon terminated by user.")


if __name__ == "__main__":
    main()
