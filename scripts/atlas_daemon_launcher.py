#!/usr/bin/env python3
"""Launcher for AtlasDaemon with systemd-style process management."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from atlas_memory.server.atlas_daemon import AtlasDaemon


async def run_daemon(sock: Path, pid_file: Path) -> None:
    daemon = AtlasDaemon(socket_path=sock, pid_path=pid_file)
    await daemon.start()
    daemon.register_signal_handlers()
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
        print(f"Daemon already running (socket: {sock})")
        sys.exit(0)

    try:
        asyncio.run(run_daemon(sock, pid_file))
    except KeyboardInterrupt:
        print("\nAtlasDaemon terminated by user.")


if __name__ == "__main__":
    main()
