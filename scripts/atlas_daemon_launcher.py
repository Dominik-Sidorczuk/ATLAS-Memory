#!/usr/bin/env python3
"""Launcher for AtlasDaemon with systemd-style process management."""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path[:] = [p for p in _sys.path if "python3.11" not in p and "hermes-agent" not in p]
_os.environ["PYTHONPATH"] = str(_os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "src")))

import asyncio
import fcntl
import logging
import os
import shutil
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any, Optional

from atlas_memory.server.atlas_daemon import AtlasDaemon

logger = logging.getLogger("atlas_daemon_launcher")

_LOCK_FILE_OBJ: Optional[Any] = None
LOG_ROTATE_MAX_BYTES = 50 * 1024 * 1024  # 50 MB


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def cleanup_stale_socket(sock: Path, min_age_s: float = 60.0, connect_timeout_s: float = 1.0) -> bool:
    """FIX-D (V106): Before acquire_process_lock, checks if socket is stale (ConnectionRefusedError) and removes it."""
    if not sock.exists():
        return False
    try:
        age_s = max(0.0, time.time() - sock.stat().st_mtime)
    except OSError:
        age_s = min_age_s

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(connect_timeout_s)
    try:
        s.connect(str(sock))
        s.close()
        return False
    except ConnectionRefusedError:
        logger.warning("Stale socket detected at %s (age=%.1fs, ConnectionRefusedError); removing", sock, age_s)
        try:
            sock.unlink()
            return True
        except OSError:
            return False
    except OSError as exc:
        if age_s >= min_age_s:
            logger.warning("Stale socket detected at %s (age=%.1fs, err=%s); removing", sock, age_s, exc)
            try:
                sock.unlink()
                return True
            except OSError:
                return False
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def rotate_daemon_log(log_file: Path, max_bytes: int = LOG_ROTATE_MAX_BYTES) -> bool:
    """FIX-D (V106): Rotates daemon.log to daemon.log.1 when file size exceeds 50 MB."""
    try:
        if log_file.exists() and log_file.stat().st_size > max_bytes:
            rotated = log_file.with_name(f"{log_file.name}.1")
            shutil.move(str(log_file), str(rotated))
            return True
    except OSError as exc:
        logger.warning("Failed to rotate log %s: %s", log_file, exc)
    return False


def acquire_process_lock(lock_path: Path) -> Any:
    """Atomic mutual exclusion using kernel file lock (flock) with dead-PID inspection."""
    global _LOCK_FILE_OBJ
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            raw_pid = lock_path.read_text(encoding="utf-8").strip()
            if raw_pid.isdigit():
                old_pid = int(raw_pid)
                if not _is_pid_alive(old_pid):
                    msg = f"stale lock from dead PID {old_pid}, cleaning"
                    logger.info(msg)
                    print(msg, flush=True)
        except OSError:
            pass

    f = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        logger.warning("lock busy, another daemon is running")
        print("lock busy, another daemon is running", flush=True)
        print("daemon already running", flush=True)
        sys.exit(1)
    try:
        f.seek(0)
        f.truncate(0)
        f.write(f"{os.getpid()}\n")
        f.flush()
    except Exception:
        pass
    logger.debug("acquired lock (%s, pid=%d)", lock_path, os.getpid())
    _LOCK_FILE_OBJ = f
    return f


class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_daemon_logging(log_path: Optional[Path] = None) -> None:
    log_file = log_path or (Path.home() / ".hermes" / "atlas" / "daemon.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    rotate_daemon_log(log_file)
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
    from atlas_memory.server.atlas_daemon import DEFAULT_SOCKET_PATH
    if Path(sock).resolve() == Path(DEFAULT_SOCKET_PATH).resolve():
        log_path = None  # default: ~/.hermes/atlas/daemon.log
    else:
        log_path = Path(sock).parent / "daemon.log"
    setup_daemon_logging(log_path)
    os.environ.setdefault("ATLAS_HIPPORAG", "1")
    atlas_dir = os.environ.get("ATLAS_DIR") or None
    daemon = AtlasDaemon.create_default(socket_path=sock, pid_path=pid_file, atlas_dir=atlas_dir)
    if lock_obj is not None:
        daemon._process_lock_handle = lock_obj
    await daemon.start()
    daemon.register_signal_handlers()
    print(f"AtlasDaemon started: {sock}", flush=True)
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

    # 0. Pre-lock stale socket cleanup (V106 FIX-D)
    cleanup_stale_socket(sock)

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
            from atlas_memory.ipc.uds_client import send_uds_request_sync
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
    except Exception as exc:
        logger.exception("Fatal unhandled exception in AtlasDaemon launcher: %s", exc)
        print(f"AtlasDaemon fatal error: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
