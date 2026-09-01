#!/usr/bin/env python3
"""Launcher for AtlasDaemon with systemd-style process management."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from atlas_memory.server.atlas_daemon import AtlasDaemon


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


async def run_daemon(sock: Path, pid_file: Path) -> None:
    setup_daemon_logging()
    daemon = AtlasDaemon.create_default(socket_path=sock, pid_path=pid_file)
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


def main() -> None:
    sock = Path(os.environ.get("ATLAS_SOCKET_PATH", str(Path.home() / ".hermes" / "atlas.sock")))
    pid_file = Path(os.environ.get("ATLAS_PID_PATH", str(Path.home() / ".hermes" / "atlas.pid")))

    if sock.exists():
        from atlas_memory.server.client import send_uds_request_sync
        res = send_uds_request_sync(sock, "ping", timeout=0.1)
        if isinstance(res, dict) and res.get("status") == "ok":
            print(f"Daemon already running and responding (socket: {sock})")
            sys.exit(0)
        else:
            print(f"Removing stale socket file: {sock}")
            try:
                sock.unlink()
            except OSError:
                pass

    if pid_file.exists():
        try:
            pid_file.unlink()
        except OSError:
            pass

    try:
        asyncio.run(run_daemon(sock, pid_file))
    except KeyboardInterrupt:
        print("\nAtlasDaemon terminated by user.")


if __name__ == "__main__":
    main()
