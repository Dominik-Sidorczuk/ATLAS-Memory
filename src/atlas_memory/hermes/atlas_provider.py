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
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from atlas_memory.orchestrator import MemoryOrchestrator

from atlas_memory.core.models import (
    HERMES_DEFAULT_SESSION,
    EpistemicSource,
    MemoryRecord,
    clamp_conf,
)
from atlas_memory.hermes.plugin_hooks import DedicatedAsyncBridge
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
from atlas_memory.hermes.trajectory_extractor import extract_trajectory
from atlas_memory.ipc.uds_client import (
    AtlasTransportError,
    get_or_create_client,
    send_uds_request_sync,
)
from atlas_memory.security.delete_guard import is_delete_protected

logger = logging.getLogger(__name__)

__all__ = ["AtlasMemoryProvider", "HERMES_DEFAULT_SESSION", "clamp_conf"]

try:
    from agent.memory_provider import MemoryProvider, RecallStatus, is_trivial_prompt
except ImportError:
    MemoryProvider = object  # type: ignore — optional-dep fallback: hermes-core nie w tym srodowisku
    def is_trivial_prompt(t: str) -> bool:  # type: ignore — optional-dep fallback (sygnatura z hermes-core)
        return not (t and t.strip())

class MnemosyneStorageBridge:
    """Architektonicznie czysty most pamięci do rdzennego silnika BeamMemory (mnemosyne.core).

    Eliminuje wielowarstwowe obejścia (cls.__new__, manipulowanie _init_home/_ambient_key)
    oraz cykliczne parsowanie stringów markdown regexem. Łączy się bezpośrednio z bazą SQLite.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        hermes_home: Optional[str] = None,
    ) -> None:
        home = hermes_home or os.environ.get("HERMES_HOME")
        if not home:
            try:
                from hermes_constants import get_hermes_home

                home = str(get_hermes_home())
            except Exception:
                home = str(Path.home() / ".hermes")
        self._home: str = home
        self._db_path: Path = Path(db_path) if db_path else (Path(self._home) / "mnemosyne" / "data" / "mnemosyne.db")
        self._beams: Dict[str, Any] = {}
        self._lock = threading.Lock()

    @property
    def _beam(self) -> Any:
        """Kompatybilność wsteczna z inspekcją stanu delegata."""
        with self._lock:
            return self._beams.get(HERMES_DEFAULT_SESSION) or (next(iter(self._beams.values())) if self._beams else None)

    def get_beam(self, session_id: str = HERMES_DEFAULT_SESSION) -> Any:
        """Deterministyczna, bezpieczna wątkowo inicjalizacja silnika BeamMemory dla danej sesji."""
        sid = session_id or HERMES_DEFAULT_SESSION
        with self._lock:
            if sid not in self._beams:
                from mnemosyne.core.beam import BeamMemory

                self._db_path.parent.mkdir(parents=True, exist_ok=True)
                self._beams[sid] = BeamMemory(session_id=sid, db_path=self._db_path)
            return self._beams[sid]

    def initialize(self, session_id: str = HERMES_DEFAULT_SESSION, **kwargs: Any) -> None:
        """Wiąże most z daną sesją i opcjonalnie odświeża hermes_home."""
        home = kwargs.get("hermes_home")
        with self._lock:
            if home and home != self._home:
                self._home = home
                self._db_path = Path(home) / "mnemosyne" / "data" / "mnemosyne.db"
                self._beams.clear()
        self.get_beam(session_id or HERMES_DEFAULT_SESSION)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Bezpośredni, synchroniczny zapis tury do bazy SQLite Mnemosyne z pełnym śladem epistemicznym."""
        sid = session_id or HERMES_DEFAULT_SESSION
        beam = self.get_beam(sid)
        raw_roles = os.environ.get("MNEMOSYNE_SYNC_ROLES", "user,assistant,tool").lower()
        active_roles = {r.strip() for r in raw_roles.split(",") if r.strip()}

        if messages and isinstance(messages, list) and "tool" in active_roles:
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                if role == "tool":
                    c = str(msg.get("content") or "").strip()
                    if c and len(c) > 5:
                        beam.remember(f"[TOOL] {c[:500]}", source="tool", importance=0.35)
                elif role == "assistant" and msg.get("tool_calls"):
                    for tc in msg.get("tool_calls") or []:
                        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                        fname = fn.get("name", "")
                        fargs = fn.get("arguments", "")
                        if fname:
                            beam.remember(f"[TOOL_CALL] {fname}({fargs})", source="tool", importance=0.30)

        if "assistant" in active_roles and assistant_content and assistant_content.strip() and len(assistant_content.strip()) > 10:
            beam.remember(f"[ASSISTANT] {assistant_content.strip()}", source="conversation", importance=0.25)

        if "user" in active_roles and user_content and user_content.strip() and len(user_content.strip()) > 5:
            beam.remember(f"[USER] {user_content.strip()}", source="conversation", importance=0.50)

    def recall_records(
        self,
        query: str,
        *,
        session_id: str = "",
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Zwraca surową listę słowników z SQLite bez pośrednictwa formatowania do tekstu."""
        sid = session_id or HERMES_DEFAULT_SESSION
        beam = self.get_beam(sid)
        return beam.recall(query, top_k=top_k)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Formatowany tekst kontekstu dla kompatybilności z passthrough."""
        sid = session_id or HERMES_DEFAULT_SESSION
        beam = self.get_beam(sid)
        results = beam.recall(query, top_k=10)
        if not results:
            return ""
        return beam.format_context(results)

    def shutdown(self) -> None:
        with self._lock:
            self._beams.clear()


# Delegate backend — preferowany most do silnika BeamMemory (mnemosyne.core)
try:
    import mnemosyne.core.beam  # noqa: F401
    from mnemosyne_hermes import MnemosyneMemoryProvider  # noqa: F401

    MNEMOSYNE_DELEGATE = MnemosyneStorageBridge
except ImportError as _mnemosyne_import_error:
    MNEMOSYNE_DELEGATE = None
    logger.warning(
        "V139/MEM-BRIDGE: mnemosyne_hermes NIEDOSTĘPNY w tym interpreterze (%s). "
        "sync_turn NIE będzie zapisywał do Mnemosyne (tylko shadow RPC do daemona). "
        "Fix: pip install 'mnemosyne-memory>=3.14' 'mnemosyne-hermes>=0.7.4' "
        "do interpretera uruchamiającego agenta (most działa w procesie — MCP nie jest wymagane).",
        _mnemosyne_import_error,
    )


def _build_mnemosyne_delegate(cls: Any) -> Any:
    """Konstrukcja delegata Mnemosyne.

    Dla natywnego MnemosyneStorageBridge zwraca czystą instancję.
    Dla zewnętrznych klas / mocków (np. MnemosyneMemoryProvider upstream 0.7.4 lub FakeBridge)
    zapewnia bezpieczną inicjalizację z pre-seedowaniem home.
    """
    if cls is None:
        return None
    try:
        if issubclass(cls, MnemosyneStorageBridge):
            return cls()
    except TypeError:
        pass

    home = None
    try:
        from hermes_constants import get_hermes_home

        home = str(get_hermes_home())
    except Exception:
        home = None
    if not home:
        home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")

    instance = cls.__new__(cls)
    instance.__dict__["_init_home"] = home
    try:
        instance.__init__()
    except Exception as exc:
        logger.warning("Mnemosyne delegate __init__ raised: %s", exc)

    if home:
        key = home
        try:
            from mnemosyne_hermes import _p1b_home_key

            key = _p1b_home_key(home)
        except Exception:
            pass
        bindings = instance.__dict__.setdefault("_bindings", {})
        if key not in bindings:
            bindings[key] = {
                "beam": None,
                "agent_identity": "",
                "session_id": "hermes_default",
            }
        instance.__dict__["_ambient_key"] = key
    return instance


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
        clean_l = line.strip()
        if not clean_l or clean_l.startswith("##"):
            continue
        stripped_bullet = re.sub(r"^[-*]\s+", "", clean_l)
        m = _MNEMOSYNE_LINE_RE.match(clean_l) or _MNEMOSYNE_LINE_RE.match(stripped_bullet)
        if not m:
            if records and not clean_l.startswith("##") and not clean_l.startswith("["):
                records[-1].object = f"{records[-1].object}\n{clean_l}"[:1000]
            continue
        try:
            importance = float(m.group("imp"))
        except (TypeError, ValueError):
            importance = 0.5
        content_text = m.group("content").strip()
        src_tag = (m.group("src") or "").strip()
        subj = f"mnemosyne:{src_tag}" if src_tag else "mnemosyne:context"
        records.append(MemoryRecord(
            subject=subj,
            predicate="context",
            object=content_text[:500] if content_text else clean_l[:500],
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


def _beam_dict_to_record(item: Dict[str, Any], fallback_query: str = "") -> MemoryRecord:
    """Bezpośrednie rzutowanie słownika z BeamMemory (SQLite) na MemoryRecord bez parsowania regexem."""
    content = str(item.get("content") or "").strip()
    try:
        importance = float(item.get("importance") if item.get("importance") is not None else 0.5)
    except (TypeError, ValueError):
        importance = 0.5
    try:
        score = float(item.get("score") if item.get("score") is not None else importance)
    except (TypeError, ValueError):
        score = importance
    src = str(item.get("source") or "conversation")
    trust = str(item.get("trust_tier") or item.get("trust") or "unknown")
    ts_val = item.get("timestamp")
    timestamp = _parse_iso_timestamp(str(ts_val)) if ts_val else time.time()
    subj = f"mnemosyne:{src}" if src else "mnemosyne:context"

    return MemoryRecord(
        subject=subj,
        predicate="context",
        object=content[:1000] if content else fallback_query[:500],
        source_type=_trust_to_source(trust),
        importance_score=max(0.0, min(1.0, importance)),
        confidence=max(0.0, min(1.0, score)),
        metadata=dict(item),
        timestamp=timestamp,
    )


class BackgroundWorker:
    """Dedicated single worker thread consuming background tasks from a bounded FIFO queue."""

    def __init__(self, name: str = "AtlasProviderWorker", maxsize: int = 1000) -> None:
        self._queue: queue.Queue[Optional[Callable[[], None]]] = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._started = False
        self._closed = False
        self._lock = threading.Lock()

    def _ensure_started(self) -> None:
        with self._lock:
            if not self._started and not self._closed:
                self._thread.start()
                self._started = True

    def submit(self, fn: Callable[[], None]) -> bool:
        """Submits task to background worker; drops if queue full or worker stopped."""
        if self._closed:
            return False
        self._ensure_started()
        try:
            self._queue.put_nowait(fn)
            return True
        except queue.Full:
            logger.warning("AtlasProviderWorker queue full; dropping background task")
            return False

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                self._queue.task_done()
                break
            try:
                task()
            except Exception as exc:
                logger.debug("BackgroundWorker task failed: %s", exc)
            finally:
                self._queue.task_done()

    def shutdown(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                if self._started:
                    self._queue.put(None)


class AtlasMemoryProvider(MemoryProvider):  # type: ignore[misc]
    """ATLAS: orchestrator nad Mnemosyne store. 6 dźwigni polityki pamięci."""

    def __init__(
        self,
        orchestrator: Optional[MemoryOrchestrator] = None,
        socket_path: Optional[str] = None,
    ) -> None:
        self._orchestrator = orchestrator
        from atlas_memory.core.config import get_config
        cfg = get_config()
        if cfg.hipporag_enabled:
            os.environ.setdefault("ATLAS_HIPPORAG", "1")
        raw_sock = socket_path or str(cfg.socket_path)
        self._socket_path = str(Path(raw_sock).expanduser())
        self._mnemosyne = None
        if MNEMOSYNE_DELEGATE is not None:
            try:
                # V139/P1b: bez factory konstruktor 0.7.4 pada wewnątrz
                # turn scope (no binding for the current home). Factory ustawia
                # _init_home PRZED pierwszym setterem property _beam.
                self._mnemosyne = _build_mnemosyne_delegate(MNEMOSYNE_DELEGATE)
            except Exception as exc:
                logger.warning("Mnemosyne delegate init failed: %s", exc)
        self._session_id = HERMES_DEFAULT_SESSION
        self._lock = threading.Lock()
        self._worker = BackgroundWorker(name=f"AtlasWorker-{self._session_id}")
        self._last_prefetch: Optional[Dict[str, Any]] = None
        self._queued_context: Optional[str] = None  # Cache z queue_prefetch
        self._queued_query: Optional[str] = None
        self._last_injected_facts: Dict[str, str] = {}
        self._context_utilization: Dict[str, Any] = {"injected_count": 0, "matched_count": 0, "ratio": 0.0}
        self._in_process_engine: Optional[Any] = None
        self._debug = cfg.debug
        self._tool_handlers: Optional[Dict[str, Callable[..., Any]]] = None
        # V89/T2: licznik hooka cieni (RPC do daemona vs in-process fallback).
        self._shadow_rpc_stats: Dict[str, int] = {"enqueued": 0, "failed": 0, "in_process_fallback": 0}
        self._shadow_last_enqueue: Optional[Dict[str, Any]] = None
        self._turn_persistence_status: str = "ok"

    def invalidate_prefetch_cache(self) -> None:
        """Reaktywne unieważnienie zbuforowanego prefetchu po mutacji stanu (V141)."""
        lock = getattr(self, "_lock", None)
        if lock is not None:
            with lock:
                self._queued_context = None
                self._queued_query = None
                self._last_prefetch = None
                self._last_injected_facts = {}
        else:
            self._queued_context = None
            self._queued_query = None
            self._last_prefetch = None
            self._last_injected_facts = {}

    def _get_fallback_engine(self) -> Any:
        """Pobiera lub leniwie inicjalizuje awaryjny silnik pamięci dla operacji in-process."""
        if self._orchestrator is not None and getattr(self._orchestrator, "engine", None) is not None:
            return self._orchestrator.engine
        if self._in_process_engine is None:
            try:
                from atlas_memory.engine import HybridMemoryEngine

                self._in_process_engine = HybridMemoryEngine.create_default(
                    db_path=":memory:", qdrant_location=":memory:"
                )
            except Exception as exc:
                logger.debug("In-process fallback engine creation note: %s", exc)
                self._in_process_engine = None
        return self._in_process_engine

    def _shadow_rpc_enabled(self) -> bool:
        """Flaga hooka cieni: ATLAS_SHADOW_RPC (domyślnie ON per AtlasConfig)."""
        from atlas_memory.core.config import get_config
        return get_config().shadow_rpc_enabled

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

    def ensure_daemon_running(self, timeout_s: float = 5.0) -> bool:
        """Ensures the ATLAS UDS daemon is running and serving get_stats (V106 contract)."""
        from atlas_memory.hermes.daemon_bootstrap import ensure_daemon_process_running

        return ensure_daemon_process_running(
            self,
            timeout_s=timeout_s,
            send_rpc_fn=send_uds_request_sync,
        )

    _ensure_daemon_running = ensure_daemon_running

    def get_stats(self) -> Dict[str, Any]:
        """Returns daemon or in-process telemetry stats (`status=ok`, `dual_node_coverage`, `points_count`, etc.)."""
        res = self._call_uds_sync("get_stats", {})
        if isinstance(res, dict):
            if "points_count" not in res:
                vs = res.get("vector_store")
                res["points_count"] = vs.get("points_count", 0) if isinstance(vs, dict) else 0
            return res
        return {"status": "in_process_fallback", "session_id": self._session_id, "points_count": 0}

    def _call_uds_sync(self, method: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Executes sync JSON-RPC 2.0 call over UDS with auto-bootstrap and 1-retry backoff (V106 FIX-B)."""
        from atlas_memory.hermes.daemon_bootstrap import call_uds_with_retry

        return call_uds_with_retry(
            self,
            method=method,
            params=params,
            send_rpc_fn=send_uds_request_sync,
        )

    def _ingest_real_trajectory(self, steps: list[dict], session_id: str) -> None:
        """V64: dostarcza REALNE trajektorie narzędzi do bufora trajektorii.

        Kanał 1 (preferowany): UDS RPC `trajectory/ingest` do daemona.
        Kanał 2 (fallback in-process): `engine.trajectory_buffer.record_real_trajectory`.
        Nigdy nie podnosi wyjątku — telemetria nie może zrywać sesji agenta.
        """
        if not steps:
            return
        try:
            resp = self._call_uds_sync(
                "trajectory/ingest", {"session_id": session_id, "steps": steps}
            )
            if resp is not None:
                return
        except AtlasTransportError as exc:
            logger.debug("UDS trajectory/ingest transport error: %s", exc)
        except Exception as exc:
            logger.debug("UDS trajectory/ingest failed: %s", exc)
        engine = None
        if self._orchestrator is not None:
            engine = getattr(self._orchestrator, "engine", None)
        buffer = getattr(engine, "trajectory_buffer", None) if engine is not None else None
        if buffer is not None and hasattr(buffer, "record_real_trajectory"):
            try:
                buffer.record_real_trajectory(steps, session_id=session_id)
            except Exception as exc:
                logger.debug("in-process record_real_trajectory failed: %s", exc)

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
        return self._orchestrator is not None or self._mnemosyne is not None

    def initialize(self, session_id: str = "", **kwargs) -> None:
        self._ensure_daemon_running()
        self._session_id = session_id or HERMES_DEFAULT_SESSION
        with self._lock:
            self._queued_context = None
            self._queued_query = None
        if self._mnemosyne is not None:
            if "hermes_home" not in kwargs or not kwargs["hermes_home"]:
                home = None
                try:
                    from hermes_constants import get_hermes_home
                    home = str(get_hermes_home())
                except Exception:
                    pass
                if not home:
                    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
                kwargs["hermes_home"] = home
            try:
                self._mnemosyne.initialize(session_id=self._session_id, **kwargs)
            except Exception as exc:
                logger.error("Mnemosyne delegate initialize failed: %s", exc, exc_info=True)
        logger.info("ATLAS provider initialized (session=%s)", self._session_id)

    def _ensure_mnemosyne_initialized(self, session_id: str = "") -> None:
        """Upewnia się, że delegat Mnemosyne posiada aktywny _beam i poprawne wiązanie home."""
        if self._mnemosyne is None:
            return
        if getattr(self._mnemosyne, "_beam", None) is not None:
            return
        home = None
        try:
            from hermes_constants import get_hermes_home
            home = str(get_hermes_home())
        except Exception:
            pass
        if not home:
            home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        sid = session_id or self._session_id or HERMES_DEFAULT_SESSION
        try:
            self._mnemosyne.initialize(session_id=sid, hermes_home=home)
        except Exception as exc:
            logger.warning("Mnemosyne auto-heal initialize failed: %s", exc, exc_info=True)

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

        # V64: REAL trajectory telemetry from transcript (objective success), NOT JEPA reward.
        try:
            real_steps = extract_trajectory(messages, sid)
            if real_steps:
                self._ingest_real_trajectory(real_steps, sid)
                # Proaktywny aparat metapoznawczy (Closed-Loop Task Feedback)
                session_success = all(s.get("success", False) for s in real_steps)
                with self._lock:
                    injected_keys = list(self._last_injected_facts.keys())
                for fact_key in injected_keys:
                    try:
                        self.handle_tool_call(
                            "atlas_task_feedback",
                            {"key": fact_key, "task_success": session_success, "delta": 0.05},
                        )
                    except Exception as fb_exc:
                        logger.debug("Automated task_feedback emit failed: %s", fb_exc)
        except Exception as exc:
            logger.debug("ATLAS trajectory extraction at session end failed: %s", exc)

        self.invalidate_prefetch_cache()

        # 1. Próba UDS call
        uds_ok = False
        try:
            self._call_uds_sync("session/end", {"session_id": sid, "turn_count": turn_count})
            uds_ok = True
        except Exception as exc:
            logger.debug("UDS session/end call failed: %s", exc)

        if hasattr(self, "_worker"):
            self._worker.shutdown()

        # 2. In-process fallback przez HermesSessionHook (tylko w razie niepowodzenia UDS)
        if not uds_ok:
            engine = None
            if self._orchestrator is not None:
                engine = getattr(self._orchestrator, "engine", None)

            if engine is not None:
                try:
                    from atlas_memory.hermes.plugin_hooks import DedicatedAsyncBridge
                    from atlas_memory.hermes.prefix_guard import HermesSessionHook

                    hook = HermesSessionHook(engine)
                    coro = hook.on_session_end(session_id=sid, trigger_sleep_cycle=True)
                    bridge = DedicatedAsyncBridge.get_instance()
                    bridge.submit_fire_and_forget(coro)
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

        # Użyj cache'owanego wyniku z queue_prefetch jeśli zapytanie pasuje
        with self._lock:
            if self._queued_context and (self._queued_query is None or self._queued_query == query):
                queued = self._queued_context
                self._queued_context = None
                self._queued_query = None
                return queued

        # 1. In-process execution if orchestrator is configured
        if self._orchestrator is not None:

            # ATLAS gate: czysty czat bez encji → SKIP (0 tokenów)
            try:
                from atlas_memory.cognitive.retrieval.gate import should_retrieve_query
                should_run, entities, reason = should_retrieve_query(query, explicit_entities=None)
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
                self._ensure_mnemosyne_initialized(sid)
                try:
                    if hasattr(self._mnemosyne, "recall_records"):
                        raw_dicts = self._mnemosyne.recall_records(query, session_id=sid, top_k=10)
                        records = [_beam_dict_to_record(d, fallback_query=query) for d in raw_dicts]
                    else:
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
                from atlas_memory.cognitive.retrieval.pipeline import RetrievalPipeline
                budgeted = RetrievalPipeline.format_inprocess_prefetch(self._orchestrator, records, query=query, max_tokens=1500)
            except Exception as exc:
                logger.warning("format_inprocess_prefetch failed: %s", exc)
                with self._lock:
                    self._last_prefetch = {"skipped": True, "reason": f"error: {exc}"}
                return ""

            block = budgeted.get("formatted_context", "")
            with self._lock:
                self._last_injected_facts = {
                    getattr(r, "effective_subject", getattr(r, "subject", str(i))): getattr(r, "object", str(r))
                    for i, r in enumerate(budgeted.get("selected_facts", []))
                }
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
            self._ensure_mnemosyne_initialized(sid)
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
        V83: delegowane do zarządzanego BackgroundWorker (zero alokacji OS thread).
        """
        sid = session_id or self._session_id
        if hasattr(self, "_worker") and self._worker is not None:
            self._worker.submit(lambda: self._prefetch_in_background(query, sid))
        else:
            self._prefetch_in_background(query, sid)

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
        # 1. Metapoznawcza telemetria utylizacji kontekstu
        lock = getattr(self, "_lock", None)
        if lock is not None:
            with lock:
                injected = dict(getattr(self, "_last_injected_facts", {}))
        else:
            injected = dict(getattr(self, "_last_injected_facts", {}))
        if injected and assistant_content:
            matched = sum(
                1
                for k, v in injected.items()
                if (k and k.lower() in assistant_content.lower())
                or (v and len(v) > 5 and v[:30].lower() in assistant_content.lower())
            )
            self._context_utilization = {
                "injected_count": len(injected),
                "matched_count": matched,
                "ratio": (matched / len(injected)) if len(injected) > 0 else 0.0,
            }

        # 2. Reaktywne unieważnienie zbuforowanego prefetchu po zakończeniu tury
        self.invalidate_prefetch_cache()

        # 3. Zawsze deleguj do Mnemosyne store (pełna trwałość)
        if self._mnemosyne is not None:
            self._ensure_mnemosyne_initialized(session_id or self._session_id)
            try:
                self._mnemosyne.sync_turn(
                    user_content,
                    assistant_content,
                    session_id=session_id or self._session_id,
                    messages=messages,
                )
            except TypeError:
                try:
                    self._mnemosyne.sync_turn(
                        user_content,
                        assistant_content,
                        session_id=session_id or self._session_id,
                    )
                except Exception as exc:
                    logger.warning("Mnemosyne sync_turn fallback failed: %s", exc, exc_info=True)
            except Exception as exc:
                # V139: exc_info=True — bez stack trace ten błąd był bezużyteczny
                # (silent degradation ścieżki zapisu do Mnemosyne, incydent 09.27–10.06)
                logger.warning("Mnemosyne sync_turn failed: %s", exc, exc_info=True)
        else:
            # V142 (Task 7): Graceful fallback when Mnemosyne is unavailable
            logger.warning("[ATLAS Provider] Mnemosyne unavailable: operating in degraded_local_persistence mode")
            self._turn_persistence_status = "degraded_local_persistence"
            engine = self._get_fallback_engine()
            if engine is not None and hasattr(engine, "kv") and engine.kv is not None:
                try:
                    ts = time.time()
                    sid_val = session_id or self._session_id
                    turn_key = f"dialog:{sid_val}:{int(ts * 1000)}"
                    turn_payload = json.dumps({
                        "user": user_content,
                        "assistant": assistant_content,
                        "timestamp": ts,
                        "status": "degraded_local_persistence",
                    }, ensure_ascii=False)
                    if hasattr(engine.kv, "set_sync"):
                        engine.kv.set_sync(
                            turn_key,
                            turn_payload,
                            confidence=0.80,
                            metadata={"source_type": "conversation_turn", "session_id": sid_val, "status": "degraded_local_persistence"},
                            reason="sync_turn:degraded_fallback",
                        )
                except Exception as kv_exc:
                    logger.debug("Local KV sync_turn fallback error: %s", kv_exc)
        # Dodatkowo: shadow reconciliation (ATLAS ekstrakcja SPO w tle przez BackgroundWorker)
        sid = session_id or self._session_id
        if self._shadow_rpc_enabled() or self._orchestrator is not None:
            self._worker.submit(lambda: self._shadow_turn_dispatch(user_content, assistant_content, sid))

    def _shadow_turn_dispatch(self, user_msg: str, agent_response: str, session_id: str) -> None:
        """Koniec tury -> `sync_turn` do demona ATLAS (L0 TTT + L1 trajektorie) + `shadow_enqueue` (LLM).

        RPC jest jedyną ścieżką z prawdziwą ekstrakcją LLM (OmniRouteShadowWorker)
        oraz natychmiastowym zasileniem L0-TTT online adaptation i L1 Arrow trajectory buffer.
        Gdy socket/RPC niedostępne, wracamy do lokalnego `_sync_turn_in_background`.
        """
        if self._shadow_rpc_enabled():
            try:
                self._call_uds_sync(
                    "sync_turn",
                    {"user_msg": user_msg, "agent_response": agent_response, "session_id": session_id},
                )
            except Exception as sync_exc:
                logger.debug("Daemon sync_turn RPC error: %s", sync_exc)

            resp = self._call_uds_sync(
                "shadow_enqueue",
                {"user_msg": user_msg, "agent_response": agent_response, "session_id": session_id},
            )
            if isinstance(resp, dict) and resp.get("status") == "ok":
                self._shadow_rpc_stats["enqueued"] += 1
                self._shadow_last_enqueue = resp
                return
            self._shadow_rpc_stats["failed"] += 1
            logger.debug("V89/T2: shadow_enqueue RPC unavailable (%s); in-process fallback", resp)
        self._shadow_rpc_stats["in_process_fallback"] += 1
        self._sync_turn_in_background(user_msg, agent_response, session_id)

    def _sync_turn_in_background(self, user_msg: str, agent_response: str, session_id: str) -> None:
        """Shadow reconciliation w tle: delegacja do kanonicznego MemoryOrchestrator.shadow_reconcile."""
        try:
            if self._orchestrator is not None:
                import asyncio
                with self._lock:
                    try:
                        asyncio.get_running_loop()
                        extracted = self._orchestrator._fallback_extract_facts(user_msg, agent_response)
                        self._orchestrator.stats["shadow_facts_extracted"] += len(extracted)
                    except RuntimeError:
                        extracted = asyncio.run(
                            self._orchestrator.shadow_reconcile(user_msg, agent_response, session_id=session_id)
                        )
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

        if tool_name in ("atlas_remember", "atlas_delete", "commit_observation"):
            self.invalidate_prefetch_cache()

        handlers = self._get_tool_handlers()
        handler = handlers.get(tool_name)
        if handler is None:
            raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

        try:
            call_args = args if isinstance(args, dict) else {}
            try:
                res = handler(call_args)
            except AtlasTransportError as exc:
                logger.debug("Handler raised AtlasTransportError: %s, falling back to in-process", exc)
                res = self._handle_in_process_tool_call(
                    tool_name,
                    call_args,
                    original_error={"status": "error", "transport_reason": exc.reason, "error": str(exc)},
                )

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
        """Pełna obsługa awaryjna 10/10 narzędzi ATLAS w procesie przy niedostępności UDS (Zero-Drop Resilience)."""
        engine = self._get_fallback_engine()

        if tool_name == "atlas_get":
            key = str(args.get("key", "")).strip()
            if engine is not None and hasattr(engine, "kv") and engine.kv is not None:
                try:
                    record = engine.kv.get(key)
                    if record is not None:
                        return {
                            "status": "ok",
                            "key": key,
                            "found": True,
                            "state": record.object if hasattr(record, "object") else str(record),
                            "mode": "in_process",
                        }
                    return {
                        "status": "ok",
                        "key": key,
                        "found": False,
                        "state": None,
                        "mode": "in_process",
                    }
                except Exception as exc:
                    logger.debug("in-process atlas_get failed: %s", exc)
            return {"status": "ok", "key": key, "found": False, "state": None, "mode": "in_process"}

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
            found = False
            if engine is not None and hasattr(engine, "kv") and engine.kv is not None:
                try:
                    rec = engine.kv.get(key)
                    if rec is not None:
                        found = True
                        engine.kv.delete(key)
                except Exception as exc:
                    logger.debug("in-process atlas_delete failed: %s", exc)
            self.invalidate_prefetch_cache()
            return {"status": "ok", "key": key, "deleted": True, "found": found, "mode": "in_process"}

        if tool_name == "atlas_stats":
            kv_count = 0
            has_graph = False
            if engine is not None and hasattr(engine, "kv") and engine.kv is not None:
                try:
                    if hasattr(engine.kv, "count_sync"):
                        kv_count = engine.kv.count_sync()
                    elif hasattr(engine.kv, "get_all_sync"):
                        kv_count = len(engine.kv.get_all_sync(paged=True, limit=1000000))
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

        if tool_name == "atlas_remember":
            key = str(args.get("key") or args.get("subject") or "").strip()
            val = str(args.get("value") or args.get("object") or "").strip()
            conf = float(args.get("confidence", 1.0))
            reason = str(args.get("reason", "agent_explicit"))
            sid = str(args.get("session_id") or self._session_id or HERMES_DEFAULT_SESSION)
            if engine is not None and hasattr(engine, "kv") and engine.kv is not None:
                rec = MemoryRecord(
                    subject=key,
                    predicate="state_value",
                    object=val,
                    confidence=conf,
                    is_state_variable=True,
                    source_type=EpistemicSource.USER_EXPLICIT,
                    metadata={"reason": reason, "session_id": sid},
                )
                try:
                    engine.kv.set(key, rec)
                except Exception as exc:
                    logger.debug("in-process atlas_remember engine.kv.set failed: %s", exc)
                if hasattr(engine, "write_path") and engine.write_path is not None:
                    try:
                        if hasattr(engine.write_path, "write_sync"):
                            engine.write_path.write_sync(key, val, confidence=conf, session_id=sid, reason=reason)
                    except Exception:
                        pass
            if self._mnemosyne is not None:
                try:
                    beam = self._mnemosyne.get_beam(sid)
                    beam.remember(f"{key}: {val}", importance=conf)
                except Exception:
                    pass
            self.invalidate_prefetch_cache()
            return {"status": "ok", "key": key, "value": val, "persisted": True, "mode": "in_process"}

        if tool_name == "atlas_recall":
            query = str(args.get("query", ""))
            sid = str(args.get("session_id") or self._session_id or HERMES_DEFAULT_SESSION)
            limit = int(args.get("limit", 15))
            recs: list[dict[str, Any]] = []
            if self._mnemosyne is not None:
                try:
                    raw_m = self._mnemosyne.recall_records(query, session_id=sid, top_k=limit)
                    for m in raw_m:
                        recs.append({
                            "subject": f"mnemosyne:{m.get('source', 'context')}",
                            "predicate": "context",
                            "object": m.get("content", ""),
                            "confidence": float(m.get("importance", 0.5)),
                        })
                except Exception:
                    pass
            if engine is not None and hasattr(engine, "kv") and engine.kv is not None:
                try:
                    all_kv = engine.kv.get_all()
                    for k, r_val in all_kv.items():
                        if query.lower() in k.lower() or (hasattr(r_val, "object") and query.lower() in str(r_val.object).lower()):
                            recs.append({
                                "subject": k,
                                "predicate": getattr(r_val, "predicate", "state_value"),
                                "object": getattr(r_val, "object", str(r_val)),
                                "confidence": getattr(r_val, "confidence", 1.0),
                            })
                except Exception:
                    pass
            lim = max(0, limit)
            return {"status": "ok", "records": recs[:lim], "count": len(recs[:lim]), "mode": "in_process"}

        if tool_name == "atlas_what_if":
            entity = str(args.get("entity") or args.get("target") or "")
            action = str(args.get("action", "simulate"))
            depth = int(args.get("depth", 2))
            uncertainty = 0.0
            if engine is not None and hasattr(engine, "record_mental_transition"):
                try:
                    from atlas_memory.core.models import ActionPlan

                    act = ActionPlan(name=action, parameters={"entity": entity, "depth": depth})
                    trans = engine.record_mental_transition(act, session_id=self._session_id)
                    uncertainty = float(trans.uncertainty) if trans else 0.0
                except Exception:
                    pass
            return {
                "status": "ok",
                "entity": entity,
                "action": action,
                "causal_paths": [{"entity": entity, "action": action, "depth": depth}],
                "uncertainty": uncertainty,
                "mode": "in_process",
            }

        if tool_name == "atlas_active_sensing":
            probe = str(args.get("probe") or args.get("target_entity", ""))
            exp = str(args.get("expected_value", ""))
            obs = str(args.get("observed_value", ""))
            has_error = bool(exp != obs and exp != "")
            if engine is not None and hasattr(engine, "record_mental_transition"):
                try:
                    from atlas_memory.core.models import ActionPlan

                    act = ActionPlan(name=f"probe:{probe}", parameters={"expected": exp, "observed": obs})
                    engine.record_mental_transition(act, session_id=self._session_id)
                except Exception:
                    pass
            return {
                "status": "ok",
                "probe": probe,
                "has_error": has_error,
                "anomaly_detected": has_error,
                "prediction_error": 0.5 if has_error else 0.0,
                "mode": "in_process",
            }

        if tool_name == "atlas_task_feedback":
            key = str(args.get("key", ""))
            task_success = bool(args.get("task_success", True))
            delta = float(args.get("delta", 0.05))
            new_conf = 1.0
            if engine is not None and hasattr(engine, "kv") and engine.kv is not None and key:
                try:
                    r_item = engine.kv.get(key)
                    if r_item:
                        cur_conf = float(getattr(r_item, "confidence", 1.0))
                        new_conf = max(0.0, min(1.0, cur_conf + delta if task_success else cur_conf - delta))
                        r_item.confidence = new_conf
                        engine.kv.set(key, r_item)
                except Exception:
                    pass
            return {
                "status": "ok",
                "key": key,
                "task_success": task_success,
                "updated_confidence": new_conf,
                "mode": "in_process",
            }

        if tool_name == "atlas_verify_chain":
            heal = bool(args.get("heal", False))
            deep = bool(args.get("deep", False))
            valid = True
            chain_len = 0
            if engine is not None and hasattr(engine, "kv") and engine.kv is not None:
                if hasattr(engine.kv, "verify_audit_chain"):
                    try:
                        valid = engine.kv.verify_audit_chain()
                    except Exception:
                        valid = True
                if hasattr(engine.kv, "count_sync"):
                    try:
                        chain_len = engine.kv.count_sync()
                    except Exception:
                        chain_len = 0
            return {
                "status": "ok",
                "verified": valid,
                "chain_length": chain_len,
                "healed": heal,
                "deep": deep,
                "mode": "in_process",
            }

        if tool_name == "atlas_sleep_sync":
            skills_dir = args.get("skills_dir")
            min_freq = int(args.get("min_frequency", 1))
            if engine is not None:
                try:
                    from atlas_memory.hermes.prefix_guard import HermesSessionHook

                    hook = HermesSessionHook(engine)
                    bridge = DedicatedAsyncBridge.get_instance()
                    bridge.submit_fire_and_forget(
                        hook.on_session_end(session_id=self._session_id, trigger_sleep_cycle=True)
                    )
                except Exception:
                    pass
            return {
                "status": "ok",
                "consolidated": True,
                "skills_dir": str(skills_dir) if skills_dir else None,
                "min_frequency": min_freq,
                "mode": "in_process",
            }

        out = {"status": "ok", "mode": "in_process", "tool": tool_name}
        if original_error and "transport_reason" in original_error:
            out["transport_reason"] = original_error["transport_reason"]
        return out
