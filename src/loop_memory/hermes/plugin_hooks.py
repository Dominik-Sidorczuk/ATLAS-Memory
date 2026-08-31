"""
AtlasPluginHooks — hermes plugin lifecycle hooks integration for ATLAS.

Provides AtlasPluginHooks wrapping AtlasMemoryProvider/HermesSessionHook
in a pure-sync contract consumable by Hermes Agent lifecycle events.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from loop_memory.hermes.prefix_guard import HermesSessionHook
except ImportError:
    HermesSessionHook = None  # type: ignore[assignment,misc]


def _run_coro_sync(coro: Any, timeout: float = 10.0) -> Any:
    """Helper to run async coroutines in sync contexts safely."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=timeout)
    else:
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

    def on_session_end(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        trigger_sleep_cycle: bool = True,
    ) -> None:
        """
        Sync hook triggered at the end of a Hermes session.
        Executes Sleep Cycle consolidation and proposed skill checks.
        """
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
                stats = _run_coro_sync(
                    hook.on_session_end(
                        session_id=session_id,
                        trigger_sleep_cycle=trigger_sleep_cycle,
                    )
                )
                proposed = getattr(stats, "proposed_skills", []) or []
                logger.info(
                    "Sleep cycle executed=%s, proposed_skills=%s",
                    trigger_sleep_cycle,
                    proposed,
                )
            elif hasattr(self.provider, "on_session_end"):
                self.provider.on_session_end(messages or [])
        except Exception as exc:
            logger.error("Error executing on_session_end in AtlasPluginHooks: %s", exc)

        return None


def create_plugin_hooks(provider: Any) -> AtlasPluginHooks:
    """Factory creating an AtlasPluginHooks instance."""
    return AtlasPluginHooks(provider=provider)
