"""
AtlasMemoryProvider — Hermes MemoryProvider integrujący ATLAS (6 dźwigni) SYNC-FIRST.

Architektura: ATLAS nie zastępuje Mnemosyne jako store — ORKIESTRUJE go.
Deleguje do mnemosyne_hermes.MnemosyneMemoryProvider jako backend, dodając:
  Lever 1: Cache Contract (system_prompt_block → immutable prefix)
  Lever 2: Retrieval Policy Gate (prefetch → should_retrieve; SKIP gdy brak encji)
  Lever 3: Epistemic Selection (epistemic-first nad zwykłym similarity)
  Lever 4: Token Budget (twardy limit 1500 tok)
  Lever 5: Shadow Reconciliation (queue → arbiter → supersede)
  Lever 6: Salience GC (prune_stale_facts)

Zgodność z per-conversation prompt caching (świętość Hermes):
  - system_prompt_block() zwraca TYLKO statyczny prefix (cache-friendly)
  - prefetch() zwraca dynamiczny kontekst (jedyna zmienna część)
  - queue_prefetch() robi recall w tle → ZERO opóźnienia w turze
  - WSZYSTKIE ścieżki SYNC (bez asyncio.run w hot path — Hermes prefetch jest sync)

Uwaga: prefetch jest synchroniczne w interfejsie MemoryProvider. ATLAS korzysta
ze zintegrowanych komponentów RetrievalPipeline (delegacja synchroniczna)
i nie czeka na async orchestrated_recall — pełna orkiestracja async jest dostępna
przez HermesMemoryAdapter (osobny przepływ).
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from atlas_memory.hermes.tools import (
        ATLAS_ACTIVE_SENSING_SCHEMA,
        ATLAS_DELETE_SCHEMA,
        ATLAS_GET_SCHEMA,
        ATLAS_RECALL_SCHEMA,
        ATLAS_REMEMBER_SCHEMA,
        ATLAS_SLEEP_SYNC_SCHEMA,
        ATLAS_STATS_SCHEMA,
        ATLAS_TASK_FEEDBACK_SCHEMA,
        ATLAS_TOOL_SCHEMAS,
        ATLAS_VERIFY_CHAIN_SCHEMA,
        ATLAS_WHAT_IF_SCHEMA,
        create_uds_tool_handlers,
    )
except ImportError:
    ATLAS_ACTIVE_SENSING_SCHEMA = {}  # type: ignore[assignment]
    ATLAS_DELETE_SCHEMA = {}  # type: ignore[assignment]
    ATLAS_GET_SCHEMA = {}  # type: ignore[assignment]
    ATLAS_RECALL_SCHEMA = {}  # type: ignore[assignment]
    ATLAS_REMEMBER_SCHEMA = {}  # type: ignore[assignment]
    ATLAS_SLEEP_SYNC_SCHEMA = {}  # type: ignore[assignment]
    ATLAS_STATS_SCHEMA = {}  # type: ignore[assignment]
    ATLAS_TASK_FEEDBACK_SCHEMA = {}  # type: ignore[assignment]
    ATLAS_TOOL_SCHEMAS = []  # type: ignore[assignment]
    ATLAS_VERIFY_CHAIN_SCHEMA = {}  # type: ignore[assignment]
    ATLAS_WHAT_IF_SCHEMA = {}  # type: ignore[assignment]
    create_uds_tool_handlers = None  # type: ignore[assignment]

from atlas_memory.cognitive.models import HERMES_DEFAULT_SESSION, clamp_conf
from atlas_memory.security.delete_guard import is_delete_protected

__all__ = ["AtlasMemoryProvider", "HERMES_DEFAULT_SESSION", "clamp_conf"]

try:
    from atlas_memory.models import (
        EpistemicSource,
        MemoryRecord,
    )
except ImportError:
    EpistemicSource = None  # type: ignore[assignment,misc]
    MemoryRecord = None  # type: ignore[assignment,misc]

from atlas_memory.cognitive.retrieval_pipeline import RetrievalPipeline

try:
    from atlas_memory.server.client import (
        AtlasDaemonClient,
        get_or_create_client,
        send_uds_request_sync,
    )
except ImportError:
    AtlasDaemonClient = None  # type: ignore[assignment,misc]
    get_or_create_client = None  # type: ignore[assignment,misc]
    send_uds_request_sync = None  # type: ignore[assignment,misc]

try:
    from atlas_memory.orchestrator import MemoryOrchestrator
    ATLAS_AVAILABLE = True
except ImportError:
    ATLAS_AVAILABLE = False
    MemoryOrchestrator = None  # type: ignore[assignment,misc]

try:
    from agent.memory_provider import MemoryProvider, RecallStatus, is_trivial_prompt
except ImportError:
    # Testy / standalone — fallback minimalny
    MemoryProvider = object  # type: ignore
    RecallStatus = None  # type: ignore
    def is_trivial_prompt(t: str) -> bool:  # type: ignore
        return not (t and t.strip())

# Delegate backend — Mnemosyne jako store
try:
    from mnemosyne_hermes import MnemosyneMemoryProvider
    MNEMOSYNE_DELEGATE = MnemosyneMemoryProvider
except ImportError:
    MNEMOSYNE_DELEGATE = None




def _sync_recall_wrapper(mnemosyne: Any):
    """Wraps Mnemosyne prefetch w sync (bezpiecznie, bo Mnemosyne używa
    wątków / sync backendu wewnątrz)."""
    def _recall(query: str, session_id: str = "") -> str:
        try:
            return mnemosyne.prefetch(query, session_id=session_id)
        except Exception as exc:
            logger.debug("Mnemosyne prefetch failed: %s", exc)
            return ""
    return _recall


# ---------------------------------------------------------------------------
# Parser formatu Mnemosyne Context (fix atlas-v5-mnemosyne-parser)
#
# MnemosyneMemoryProvider.prefetch() zwraca tekst, w którym KAŻDA linia to jeden
# rekord pamięci w formacie:
#     [TIMESTAMP] (importance X, source Y) [TRUST] CONTENT
# np.:
#     ## Mnemosyne Context
#      [2026-08-30T10:47:00] (importance 0.7, source task) [trust:stated] treść...
#      [2026-08-29T14:10] (importance 0.95, source canonical:procedure) [CANONICAL] ...
#
# KRUCHA HEURYSTYKA (`":" in line or "- " in line`) gubiła rekordy, których
# content nie zawiera dwukropka ani myślnika → records=[] → prefetch zwracał
# {"skipped": True, "reason": "no_records_from_backend"} → 0/20 trafień w runtime.
# ---------------------------------------------------------------------------
_MNEMOSYNE_LINE_RE = re.compile(
    r"^\s*\[(?P<ts>[^\]]+)\]\s*"
    r"\(importance\s+(?P<imp>[\d.]+),\s*source\s+(?P<src>[^)]*)\)\s*"
    r"(?:\[(?P<trust>[^\]]*)\])?\s*"
    r"(?P<content>.*)$"
)


def _parse_iso_timestamp(ts: str) -> float:
    """Konwertuje ISO timestamp Mnemosyne (np. 2026-08-30T10:47:00) na UNIX epoch."""
    if not ts:
        return time.time()
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, TypeError):
        return time.time()


def _trust_to_source(trust: Optional[str]) -> EpistemicSource:
    """Mapuje tag zaufania Mnemosyne na EpistemicSource (fallback EXTERNAL_DOC).

    EpistemicSource nie ma członu CANONICAL — rekordy kanoniczne / trust:stated
    lądują w EXTERNAL_DOC (0.65 w epistemicznym priorytecie), co jest spójne
    z fallbackiem prefetch (zadanie: source_type=EpistemicSource.EXTERNAL_DOC).
    """
    if not trust:
        return EpistemicSource.EXTERNAL_DOC
    t = trust.lower()
    if "user" in t:
        return EpistemicSource.USER_EXPLICIT
    if "tool" in t:
        return EpistemicSource.TOOL_OUTPUT
    if "infer" in t or "agent" in t:
        return EpistemicSource.AGENT_INFERENCE
    return EpistemicSource.EXTERNAL_DOC


def _parse_mnemosyne_context(raw: str, fallback_query: str = "") -> List[Any]:
    """Parsuje tekst zwrócony przez MnemosyneMemoryProvider.prefetch().

    - Pomija nagłówek '## Mnemosyne Context' oraz puste linie.
    - Każda linia zgodna z formatem -> jeden MemoryRecord (metadata: importance,
      source, timestamp, trust).
    - Brak dopasowań (nieznany format) + niepusty raw -> fallback: cały tekst
      jako JEDEN rekord EXTERNAL_DOC (lepsze niż milczące 0 trafień).
    - Pusty raw -> [] (prefetch zachowa zachowanie "no_records_from_backend").
    """
    if not raw or not raw.strip():
        return []

    records: List[MemoryRecord] = []
    for line in raw.split("\n"):
        if not line.strip() or line.strip().startswith("##"):
            continue
        m = _MNEMOSYNE_LINE_RE.match(line)
        if not m:
            continue
        try:
            importance = float(m.group("imp"))
        except (TypeError, ValueError):
            importance = 0.5
        records.append(MemoryRecord(
            subject=line.strip()[:80],
            predicate="context",
            object=m.group("content").strip()[:500],
            source_type=_trust_to_source(m.group("trust")),
            importance_score=importance,
            metadata={
                "importance": importance,
                "source": (m.group("src") or "").strip(),
                "timestamp": m.group("ts"),
                "trust": (m.group("trust") or "").strip(),
            },
            timestamp=_parse_iso_timestamp(m.group("ts")),
        ))

    if records:
        return records

    # Fallback: nieznany format — zapisz cały surowy tekst jako jeden rekord
    return [MemoryRecord(
        subject=fallback_query[:80] or "mnemosyne_context",
        predicate="context",
        object=raw.strip()[:1000],
        source_type=EpistemicSource.EXTERNAL_DOC,
        timestamp=time.time(),
    )]


class AtlasMemoryProvider(MemoryProvider):  # type: ignore[misc]
    """ATLAS: orchestrator nad Mnemosyne store. 6 dźwigni polityki pamięci."""

    def __init__(
        self,
        orchestrator: Optional[MemoryOrchestrator] = None,
        socket_path: Optional[str] = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._socket_path = socket_path or os.environ.get("ATLAS_SOCKET_PATH", str(Path.home() / ".hermes" / "atlas.sock"))
        self._mnemosyne = None
        if MNEMOSYNE_DELEGATE is not None:
            try:
                self._mnemosyne = MNEMOSYNE_DELEGATE()
            except Exception as exc:
                logger.warning("Mnemosyne delegate init failed: %s", exc)
        self._session_id = HERMES_DEFAULT_SESSION
        self._lock = threading.Lock()
        self._last_prefetch: Optional[Dict[str, Any]] = None
        self._queued_context: Optional[str] = None  # Cache z queue_prefetch
        self._queued_query: Optional[str] = None
        self._debug = os.environ.get("ATLAS_DEBUG", "0") == "1"
        self._tool_handlers: Optional[Dict[str, Callable[..., Any]]] = None

    def _get_tool_handlers(self) -> Dict[str, Callable[..., Any]]:
        """Zwraca lub inicjalizuje słownik handlerów narzędzi UDS."""
        if self._tool_handlers is None:
            if create_uds_tool_handlers is not None:
                self._tool_handlers = create_uds_tool_handlers(self._socket_path)
            else:
                self._tool_handlers = {}
        return self._tool_handlers

    def get_client(self) -> Optional[Any]:
        """Returns thread-local AtlasDaemonClient if available."""
        if get_or_create_client is None:
            return None
        return get_or_create_client(self._socket_path) if self._socket_path else get_or_create_client()

    def _call_uds_sync(self, method: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Executes sync JSON-RPC 2.0 call over UDS if socket is present, returns None on fallback."""
        sock = self._socket_path or os.environ.get("ATLAS_SOCKET_PATH", str(Path.home() / ".hermes" / "atlas.sock"))
        if not Path(sock).exists():
            return None

        if send_uds_request_sync is not None:
            return send_uds_request_sync(sock, method, params=params or {})

        client = self.get_client()
        if client is None or not client.socket_path.exists():
            return None

        import asyncio
        params_dict = params or {}
        try:
            return asyncio.run(client.call(method, params_dict))
        except Exception as exc:
            logger.debug("UDS call %s failed, falling back to in-process: %s", method, exc)
            return None

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "atlas"

    def is_available(self) -> bool:
        """ATLAS + backend dostępne? Zero network calls."""
        sock = Path(self._socket_path) if self._socket_path else Path(os.environ.get("ATLAS_SOCKET_PATH", str(Path.home() / ".hermes" / "atlas.sock")))
        if sock.exists():
            return True
        return ATLAS_AVAILABLE and (self._orchestrator is not None or self._mnemosyne is not None)

    def initialize(self, session_id: str = "", **kwargs) -> None:
        self._session_id = session_id or HERMES_DEFAULT_SESSION
        with self._lock:
            self._queued_context = None
            self._queued_query = None
        if self._mnemosyne is not None:
            try:
                self._mnemosyne.initialize(session_id=session_id, **kwargs)
            except Exception as exc:
                logger.warning("Mnemosyne delegate initialize failed: %s", exc)
        logger.info("ATLAS provider initialized (session=%s)", self._session_id)

    def shutdown(self) -> None:
        if self._mnemosyne is not None:
            try:
                self._mnemosyne.shutdown()
            except Exception as exc:
                logger.debug("Mnemosyne shutdown failed: %s", exc)
        logger.info("ATLAS provider shutdown")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Obsługuje zakończenie sesji: deleguje do HermesSessionHook lub UDS/fallback. SYNC.
        
        Zgodne z Hermes MemoryProvider ABC: def on_session_end(self, messages: List[Dict[str, Any]]) -> None
        """
        with self._lock:
            self._queued_context = None
            self._queued_query = None
        session_id = ""
        if messages and isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
            session_id = messages[0].get("session_id", "")
        sid = session_id or self._session_id
        turn_count = len(messages) if messages else 0
        if self._debug:
            logger.info("[ATLAS] on_session_end called for session=%s, turn_count=%d", sid, turn_count)

        # 1. Próba UDS call
        try:
            self._call_uds_sync("session/end", {"session_id": sid, "turn_count": turn_count})
        except Exception as exc:
            logger.debug("UDS session/end call failed: %s", exc)

        # 2. In-process fallback przez HermesSessionHook
        engine = None
        if self._orchestrator is not None:
            engine = getattr(self._orchestrator, "engine", None)

        if engine is not None:
            try:
                import asyncio
                import concurrent.futures

                from atlas_memory.hermes.prefix_guard import HermesSessionHook

                hook = HermesSessionHook(engine)
                coro = hook.on_session_end(session_id=sid, trigger_sleep_cycle=True)

                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        pool.submit(asyncio.run, coro).result(timeout=10.0)
                else:
                    asyncio.run(coro)
            except Exception as exc:
                logger.warning("In-process on_session_end failed: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Lever 1: Cache Contract — system_prompt_block
    # ------------------------------------------------------------------
    def system_prompt_block(self) -> str:
        """STATYCZNY prefix — niezmienny w trakcie sesji (cache-friendly)."""
        if self._orchestrator is None:
            return ""
        try:
            return self._orchestrator.build_cache_contract_prefix(
                profile_state={"provider": "atlas", "session": self._session_id},
                system_rules=[
                    "ATLAS memory: epistemic-first recall (USER > TOOL > EXT_DOC > INFERENCE)",
                    "ATLAS memory: retrieval policy gate active (skip when no entities)",
                    "ATLAS memory: shadow reconciliation writes async (never blocks)",
                ],
            )
        except Exception as exc:
            logger.warning("system_prompt_block failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Lever 2 + 3 + 4: Retrieval Gate → Epistemic Selection → Token Budget
    # ------------------------------------------------------------------
    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Zwraca sformatowany kontekst do wstrzyknięcia LUB "" (skip). SYNC."""
        # Trivial gate (Hermes-native: greetings, 'ok', slash commands)
        if is_trivial_prompt(query):
            with self._lock:
                self._last_prefetch = {"skipped": True, "reason": "trivial_prompt"}
            return ""

        sid = session_id or self._session_id

        # 1. In-process execution if orchestrator is configured
        if self._orchestrator is not None:
            if not ATLAS_AVAILABLE or EpistemicSource is None or MemoryRecord is None:
                return ""

            # Użyj cache'owanego wyniku z queue_prefetch jeśli zapytanie pasuje
            with self._lock:
                if self._queued_context and (self._queued_query is None or self._queued_query == query):
                    queued = self._queued_context
                    self._queued_context = None
                    self._queued_query = None
                    return queued

            # ATLAS gate: czysty czat bez encji → SKIP (0 tokenów)
            try:
                should_run, entities, reason = self._orchestrator.should_retrieve(query, explicit_entities=None)
            except Exception as exc:
                logger.warning("should_retrieve failed: %s", exc)
                with self._lock:
                    self._last_prefetch = {"skipped": True, "reason": f"error: {exc}"}
                return ""

            if not should_run:
                with self._lock:
                    self._last_prefetch = {
                        "skipped": True,
                        "reason": reason,
                        "tokens_saved": self._orchestrator.stats.get("tokens_saved_estimate", 0),
                    }
                if self._debug:
                    logger.info("[ATLAS] prefetch SKIP: %s", reason)
                return ""

            records: List[Any] = []
            if self._mnemosyne is not None:
                try:
                    raw = self._mnemosyne.prefetch(query, session_id=sid)
                    records = _parse_mnemosyne_context(raw, fallback_query=query)
                except Exception as exc:
                    logger.warning("Mnemosyne recall failed: %s", exc)

            if not records:
                with self._lock:
                    self._last_prefetch = {"skipped": True, "reason": "no_records_from_backend"}
                return ""

            # Lever 3 + 4: Delegated Epistemic Selection & Token Budget (SYNC — RetrievalPipeline)
            try:
                budgeted = RetrievalPipeline.format_inprocess_prefetch(self._orchestrator, records, query=query, max_tokens=1500)
            except Exception as exc:
                logger.warning("format_inprocess_prefetch failed: %s", exc)
                with self._lock:
                    self._last_prefetch = {"skipped": True, "reason": f"error: {exc}"}
                return ""

            block = budgeted.get("formatted_context", "")
            with self._lock:
                self._last_prefetch = {
                    "skipped": False,
                    "count": len(budgeted.get("selected_facts", [])),
                    "tokens": budgeted.get("estimated_tokens", 0),
                    "source": "atlas_orchestrator",
                }
            if self._debug:
                logger.info("[ATLAS] prefetch HIT: %d facts, ~%d tokens", self._last_prefetch["count"], self._last_prefetch["tokens"])
            return block

        # 2. Production path: Fast UDS recall from AtlasDaemon (< 15 µs)
        try:
            uds_res = self._call_uds_sync("prefetch", {"query": query, "session_id": sid})
            if uds_res and isinstance(uds_res, dict):
                ctx_block = uds_res.get("context_block", "")
                if ctx_block:
                    with self._lock:
                        self._last_prefetch = {
                            "skipped": False,
                            "count": uds_res.get("count", len(uds_res.get("records", []))),
                            "source": "atlas_daemon_uds",
                        }
                    return ctx_block
        except Exception as uds_err:
            logger.debug("UDS prefetch call failed: %s", uds_err)

        # 3. Fallback: Pass-through do Mnemosyne
        if self._mnemosyne is not None:
            try:
                result = self._mnemosyne.prefetch(query, session_id=sid)
                with self._lock:
                    self._last_prefetch = {"skipped": False, "source": "mnemosyne_passthrough", "count": len(result)}
                return result
            except Exception as exc:
                logger.warning("Mnemosyne passthrough failed: %s", exc)
                with self._lock:
                    self._last_prefetch = {"skipped": True, "reason": f"error: {exc}"}
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Background recall na NASTĘPNĄ turę. Zero opóźnienia w tej turze.

        Fix Bug 3 (Gemini Flash 3.6 review): wynik prefetch w tle jest
        cache'owany w self._queued_context i konsumowany przez następny prefetch.
        """
        if self._orchestrator is not None:
            thread = threading.Thread(
                target=self._prefetch_in_background,
                args=(query, session_id or self._session_id),
                daemon=True,
            )
            thread.start()

    def _prefetch_in_background(self, query: str, session_id: str) -> None:
        """Wykonuje recall w tle i cache'uje wynik dla następnej tury."""
        try:
            result = self.prefetch(query, session_id=session_id)
            if result:
                with self._lock:
                    self._queued_context = result
                    self._queued_query = query
        except Exception as exc:
            logger.debug("Background prefetch failed: %s", exc)

    def recall_status(self) -> Optional[RecallStatus]:
        if self._last_prefetch is None:
            return None
        if self._last_prefetch.get("skipped"):
            return None
        count = self._last_prefetch.get("count", 0)
        if count == 0:
            return None
        return RecallStatus(provider_label="atlas", count=count, glyph="🗺️")

    # ------------------------------------------------------------------
    # Lever 5: Shadow Reconciliation — sync_turn (non-blocking)
    # ------------------------------------------------------------------
    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist tury → shadow queue OR Mnemosyne (async, NEVER blocks main loop).

        WAŻNE (fix Bug 1 z code review Gemini Flash 3.6): gdy orchestrator jest
        obecny, tura MUSI trafić też do Mnemosyne store — inaczej konwersacja
        nie jest nigdzie zapisywana. Pętla zapisuje do obu: shadow queue
        (ekstrakcja) + Mnemosyne (pełny zapis tury).

        Fix 2026-08-28: MnemosyneMemoryProvider.sync_turn() nie wspiera messages=
        (tylko user_content, assistant_content, session_id) — pomijamy messages.
        Parametr messages pozostaje w sygnaturze dla zgodności z kontraktem
        MemoryProvider, ale nie jest przekazywany dalej.
        """
        # Zawsze deleguj do Mnemosyne store (pełna trwałość)
        if self._mnemosyne is not None:
            try:
                # MnemosyneMemoryProvider.sync_turn() NIE akceptuje messages= — przekazuj tylko podstawowe
                self._mnemosyne.sync_turn(
                    user_content, assistant_content,
                    session_id=session_id or self._session_id,
                )
            except Exception as exc:
                logger.warning("Mnemosyne sync_turn failed: %s", exc)
        # Dodatkowo: shadow reconciliation (ATLAS ekstrakcja SPO w tle)
        if self._orchestrator is not None:
            thread = threading.Thread(
                target=self._sync_turn_in_background,
                args=(user_content, assistant_content, session_id or self._session_id),
                daemon=True,
            )
            thread.start()

    def _sync_turn_in_background(self, user_msg: str, agent_response: str, session_id: str) -> None:
        """Shadow reconciliation w tle: ekstrakcja SPO z tury (bez LLM — regex fallback)."""
        try:
            if self._orchestrator is not None:
                with self._lock:
                    # Fix Bug 2: _extract_facts_regex NIE ISTNIEJE — jest _fallback_extract_facts
                    extracted = self._orchestrator._fallback_extract_facts(user_msg, agent_response)
                    if extracted and self._debug:
                        logger.info("[ATLAS] shadow sync: extracted %d facts", len(extracted))
        except Exception as exc:
            logger.debug("Shadow reconcile failed: %s", exc)

    # ------------------------------------------------------------------
    # Tools: first-class ATLAS active tool-calling (MemoryProvider ABC)
    # ------------------------------------------------------------------
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Zwraca schematy 10 narzędzi ATLAS (OpenAI function calling) dla MemoryManager."""
        schemas = ATLAS_TOOL_SCHEMAS or [
            ATLAS_RECALL_SCHEMA,
            ATLAS_REMEMBER_SCHEMA,
            ATLAS_WHAT_IF_SCHEMA,
            ATLAS_ACTIVE_SENSING_SCHEMA,
            ATLAS_STATS_SCHEMA,
            ATLAS_GET_SCHEMA,
            ATLAS_DELETE_SCHEMA,
            ATLAS_TASK_FEEDBACK_SCHEMA,
            ATLAS_VERIFY_CHAIN_SCHEMA,
            ATLAS_SLEEP_SYNC_SCHEMA,
        ]
        bare_schemas: List[Dict[str, Any]] = []
        for s in schemas:
            if not isinstance(s, dict):
                continue
            if s.get("type") == "function" and isinstance(s.get("function"), dict):
                bare_schemas.append(dict(s["function"]))
            elif s.get("name"):
                bare_schemas.append(dict(s))
        return bare_schemas

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        """Obsługuje wywołanie narzędzia ATLAS delegując do daemona UDS (lub in-process fallback)."""
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}

        handlers = self._get_tool_handlers()
        handler = handlers.get(tool_name)
        if handler is None:
            raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

        try:
            call_args = args if isinstance(args, dict) else {}
            res = handler(call_args)

            # In-process fallback if daemon was offline
            if (
                isinstance(res, dict)
                and res.get("status") in ("daemon_offline", "error")
            ):
                res = self._handle_in_process_tool_call(tool_name, call_args, original_error=res)

            if isinstance(res, str):
                return res
            return json.dumps(res, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error("Error executing %s in provider %s: %s", tool_name, self.name, exc)
            return json.dumps({"status": "error", "tool": tool_name, "error": str(exc)}, ensure_ascii=False)

    def _handle_in_process_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        original_error: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """In-process fallback when daemon is unreachable."""
        if tool_name == "atlas_delete":
            key = str(args.get("key", "")).strip()
            is_prot, pfx = is_delete_protected(key)
            if is_prot:
                return {
                    "status": "rejected",
                    "key": key,
                    "deleted": False,
                    "found": False,
                    "reason": "protected_key",
                    "message": f"Key '{key}' is protected (prefix '{pfx}'). Use soft tombstone or explicit ADR override.",
                    "mode": "in_process",
                }
            return {"status": "daemon_offline"}

        if tool_name == "atlas_stats":
            kv_count = 0
            has_graph = False
            if self._orchestrator is not None:
                engine = getattr(self._orchestrator, "engine", None)
                if engine is not None and hasattr(engine, "kv") and engine.kv is not None:
                    try:
                        if hasattr(engine.kv, "get_all_sync"):
                            kv_count = len(engine.kv.get_all_sync())
                        else:
                            kv_count = len(engine.kv.get_all())
                    except Exception:
                        kv_count = 0
                has_graph = engine is not None and hasattr(engine, "graph")
            return {
                "status": "ok",
                "serving": False,
                "mode": "in_process",
                "kv_records": kv_count,
                "has_graph": has_graph,
                "has_orchestrator": self._orchestrator is not None,
            }

        return {"status": "daemon_offline"}
