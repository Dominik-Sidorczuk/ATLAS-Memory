"""
AtlasPluginHooks — hermes plugin lifecycle hooks integration for ATLAS.

Provides AtlasPluginHooks wrapping AtlasMemoryProvider/HermesSessionHook
in a pure-sync contract consumable by Hermes Agent lifecycle events.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from atlas_memory.hermes.prefix_guard import HermesSessionHook
except ImportError:
    HermesSessionHook = None  # type: ignore[assignment,misc]


import threading
import time

logger = logging.getLogger(__name__)


class DedicatedAsyncBridge:
    """Wątkowo-bezpieczny most asynchroniczny z wyizolowaną pętlą zdarzeń.
    
    Eliminuje konflikty krzyżowania pętli (RuntimeError: Task got Future attached to a different loop),
    zapewniając deterministyczny czas wykonania oraz możliwość odłączonego zlecania (fire-and-forget).
    """
    _instance: Optional["DedicatedAsyncBridge"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "DedicatedAsyncBridge":
        with cls._lock:
            if cls._instance is None or cls._instance._closed or cls._instance._thread is None or not cls._instance._thread.is_alive():
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()
        self._closed = False
        self._last_status: Dict[str, Any] = {"dispatched_tasks": 0, "last_task_time": 0.0}
        self._thread = threading.Thread(target=self._runner, name="atlas-async-bridge", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _runner(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    def run_coroutine(self, coro: Any, timeout: float = 10.0) -> Any:
        """Uruchamia korutynę synchronicznie w izolowanej pętli mostu bez kolizji z pętlą bieżącą."""
        if self._closed or self._loop is None:
            raise RuntimeError("DedicatedAsyncBridge is closed or uninitialized")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def submit_fire_and_forget(self, coro_or_func: Any) -> bool:
        """Zleca zadanie asynchronicznie w tle (fire-and-forget) z rejestracją telemetrii."""
        if self._closed or self._loop is None:
            return False
        self._last_status["dispatched_tasks"] += 1
        self._last_status["last_task_time"] = time.time()
        if asyncio.iscoroutine(coro_or_func):
            asyncio.run_coroutine_threadsafe(coro_or_func, self._loop)
        elif callable(coro_or_func):
            self._loop.call_soon_threadsafe(coro_or_func)
        return True

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)


def _run_coro_sync(coro: Any, timeout: float = 10.0) -> Any:
    """Helper do bezpiecznego uruchamiania korutyn w synchronicznym kodzie bez konfliktów pętli."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Aktywna pętla w procesie -> bezpieczne rzutowanie do dedykowanego mostu wątkowego
        bridge = DedicatedAsyncBridge.get_instance()
        return bridge.run_coroutine(coro, timeout=timeout)
    else:
        # Brak aktywnej pętli w bieżącym wątku -> bezpośrednie uruchomienie
        return asyncio.run(coro)


class AtlasPluginHooks:
    """
    Plugin hooks adapter for ATLAS Memory Provider.
    Delegates session lifecycle events to HermesSessionHook and Provider.
    """

    def __init__(self, provider: Any, session_hook: Optional[Any] = None) -> None:
        self.provider = provider
        self._session_hook = session_hook

    def _extract_context(self, messages: Optional[List[Dict[str, Any]]] = None) -> tuple[str, int]:
        """Extracts session_id and turn_count from messages list."""
        if not messages or not isinstance(messages, list):
            return "", 0
        session_id = ""
        if len(messages) > 0 and isinstance(messages[0], dict):
            session_id = messages[0].get("session_id", "")
        return session_id, len(messages)

    def _get_session_hook(self) -> Optional[Any]:
        if self._session_hook is not None:
            return self._session_hook
        if self.provider is not None:
            orchestrator = getattr(self.provider, "_orchestrator", None)
            engine = getattr(orchestrator, "engine", None) if orchestrator else None
            if engine is not None and HermesSessionHook is not None:
                return HermesSessionHook(engine)
        return None

    def on_session_start(self, session_id: str = "") -> bool:
        """
        Sync watchdog hook triggered at the start of a Hermes session (V106 FIX-C).
        Checks `Path(provider._socket_path).exists()`; if absent, calls `provider._ensure_daemon_running(timeout_s=3.0)`.
        """
        return on_session_start(self.provider, session_id=session_id)

    def on_session_end(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        trigger_sleep_cycle: bool = True,
        await_completion: bool = True,
    ) -> None:
        """
        Sync hook triggered at the end of a Hermes session.
        Gwarantuje sub-5ms budżet dispatchu przy await_completion=False (fire-and-forget).
        """
        t0 = time.perf_counter()
        session_id, turn_count = self._extract_context(messages)
        logger.info(
            "AtlasPluginHooks.on_session_end invoked (session_id=%s, turn_count=%d, trigger_sleep_cycle=%s)",
            session_id,
            turn_count,
            trigger_sleep_cycle,
        )

        try:
            hook = self._get_session_hook()
            if hook is not None:
                if trigger_sleep_cycle and not await_completion:
                    # Odłączone zlecenie fire-and-forget (sub-5ms)
                    bridge = DedicatedAsyncBridge.get_instance()
                    coro = hook.on_session_end(
                        session_id=session_id,
                        trigger_sleep_cycle=True,
                    )
                    bridge.submit_fire_and_forget(coro)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    logger.debug("on_session_end dispatched sleep consolidation in %.2f ms", elapsed_ms)
                    return None
                else:
                    # Synchroniczne oczekiwanie (np. w testach lub przy braku cyklu snu)
                    stats = _run_coro_sync(
                        hook.on_session_end(
                            session_id=session_id,
                            trigger_sleep_cycle=trigger_sleep_cycle,
                        ),
                        timeout=10.0,
                    )
                    proposed = getattr(stats, "proposed_skills", []) or []
                    logger.info("Sleep cycle executed=%s, proposed_skills=%s", trigger_sleep_cycle, proposed)
                    return None
            elif hasattr(self.provider, "on_session_end"):
                if not await_completion:
                    bridge = DedicatedAsyncBridge.get_instance()
                    bridge.submit_fire_and_forget(lambda: self.provider.on_session_end(messages or []))
                else:
                    self.provider.on_session_end(messages or [])
        except Exception as exc:
            logger.error("Error executing on_session_end in plugin hook: %s", exc)

        return None


def on_session_start(provider: Any, session_id: str = "") -> bool:
    """V106 FIX-C: Module-level and hook-level watchdog ensuring daemon is running at session start."""
    if provider is None:
        logger.warning("on_session_start called with provider=None")
        return False
    from pathlib import Path

    sock_raw = getattr(provider, "_socket_path", None)
    if not sock_raw:
        logger.warning("on_session_start: provider missing _socket_path; fallback to in-process")
        return False
    sock_path = Path(str(sock_raw)).expanduser()
    if sock_path.exists():
        logger.info("on_session_start: daemon socket healthy at %s (session=%s)", sock_path, session_id)
        return True
    ensure_fn = getattr(provider, "_ensure_daemon_running", None)
    if callable(ensure_fn):
        ok = bool(ensure_fn(timeout_s=3.0))
        if ok:
            logger.info("on_session_start: daemon auto-bootstrapped at %s (session=%s)", sock_path, session_id)
        else:
            logger.warning(
                "on_session_start: daemon spawn timeout/fallback after 3.0s at %s (session=%s)",
                sock_path,
                session_id,
            )
        return ok
    logger.warning("on_session_start: provider lacks _ensure_daemon_running; fallback to in-process")
    return False


def on_session_end(
    provider: Any,
    messages: Optional[List[Dict[str, Any]]] = None,
    trigger_sleep_cycle: bool = True,
) -> None:
    """V106 FIX-C: Graceful session end cleanup without sending SIGTERM to the persistent daemon."""
    if provider is None:
        return None
    hooks = AtlasPluginHooks(provider=provider)
    hooks.on_session_end(messages=messages, trigger_sleep_cycle=trigger_sleep_cycle)
    return None


def create_plugin_hooks(provider: Any) -> AtlasPluginHooks:
    """Factory creating an AtlasPluginHooks instance."""
    return AtlasPluginHooks(provider=provider)
