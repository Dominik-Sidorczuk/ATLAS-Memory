"""V87 (T3) — test daemon-live: licznik L0-TTT musi przezyc restart daemona."""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from atlas_memory.ipc.uds_client import send_uds_request_sync
from atlas_memory.server.atlas_daemon import AtlasDaemon


class _LiveDaemon:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.sock = root / "iso_atlas.sock"
        self.pid = root / "iso_atlas.pid"
        self.atlas_dir = root / "iso_atlas"
        assert len(str(self.sock)) < 100, f"ścieżka testowego socketa UDS za długa: {self.sock}"
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._daemon: AtlasDaemon | None = None

    def boot(self) -> None:
        ready = threading.Event()
        err_holder: list[BaseException] = []

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            try:
                self._daemon = AtlasDaemon.create_default(
                    socket_path=self.sock,
                    pid_path=self.pid,
                    atlas_dir=self.atlas_dir,
                    ingest_mnemosyne_on_start=False,
                )
                loop.run_until_complete(self._daemon.start())
                ready.set()
                loop.run_forever()
            except BaseException as exc:
                err_holder.append(exc)
                ready.set()
            finally:
                if self._daemon is not None:
                    try:
                        loop.run_until_complete(self._daemon.stop())
                    except Exception:
                        pass
                loop.close()

        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()
        assert ready.wait(timeout=15.0), "daemon nie wystartował w 15s"
        if err_holder:
            raise AssertionError(f"daemon boot error: {err_holder[0]}") from err_holder[0]
        res = send_uds_request_sync(self.sock, "ping", timeout=2.0)
        assert isinstance(res, dict) and res.get("status") == "ok"

    def rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        res = send_uds_request_sync(self.sock, method, params=params, timeout=15)
        assert isinstance(res, dict), f"{method} -> {res!r}"
        return res

    def stats(self) -> dict[str, Any]:
        return self.rpc("get_stats")

    def kill(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._thread = None
        self._loop = None
        self._daemon = None


@pytest.fixture()
def live_daemon(tmp_path: Path, uds_dir: Path):
    daemon = _LiveDaemon(uds_dir)
    daemon.boot()
    try:
        yield daemon
    finally:
        daemon.kill()


def test_v87_t3_ttt_steps_survive_daemon_restart(uds_dir: Path) -> None:
    """T3 RED: set → ttt_steps>0, kazdy zapis podbija, a po RESTARCIE licznik nie wraca do 0."""
    d = _LiveDaemon(uds_dir)
    try:
        d.boot()
        assert d.stats()["ttt_steps"] == 0, "swiezy daemon startuje z licznikiem 0 (stan poczatkowy)"

        d.rpc("set", {"key": "test:v87:t3:k1", "value": "pierwszy zapis przez prawdziwy UDS", "confidence": 0.9})
        s1 = d.stats()["ttt_steps"]
        assert s1 >= 1, f"po pierwszym set ttt_steps musi byc >=1, jest {s1}"

        d.rpc("set", {"key": "test:v87:t3:k2", "value": "drugi zapis przez prawdziwy UDS", "confidence": 0.9})
        s2 = d.stats()["ttt_steps"]
        assert s2 > s1, f"kazdy kolejny set musi podbic ttt_steps ({s1} -> {s2})"

        # Prawdziwy restart daemona (stop -> nowy create_default na tym samym ATLAS_DIR)
        d.kill()
        d.boot()

        s_after_restart = d.stats()["ttt_steps"]
        assert s_after_restart == s2, (
            f"po restarcie daemona ttt_steps zresetowal sie ({s2} -> {s_after_restart}) "
            "— brak odczytu licznika z trwalej warstwy przy starcie"
        )

        # I po restarcie kolejne zapisy dalej inkrementuja od odtworzonego stanu (nie od 0)
        d.rpc("set", {"key": "test:v87:t3:k3", "value": "zapis po restarcie", "confidence": 0.9})
        s3 = d.stats()["ttt_steps"]
        assert s3 == s2 + 1, f"po restarcie inkrementacja od {s2}: oczekiwano {s2 + 1}, jest {s3}"
    finally:
        d.kill()
