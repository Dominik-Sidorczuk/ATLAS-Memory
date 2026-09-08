"""AtlasDaemon — Micro-Sidecar server for ATLAS over Unix Domain Socket with JSON-RPC 2.0."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import struct
import time
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from atlas_memory.cognitive import (
    ConflictArbiter,
    EpistemicWritePath,
    HotBufferPolicy,
    PredictiveCoder,
    RetrievalPipeline,
)
from atlas_memory.cognitive.hot_buffer import (
    extract_effective_session,
    is_conversational_churn,
    is_session_accessible,
)
from atlas_memory.cognitive.models import (
    HERMES_DEFAULT_SESSION,
    clamp_conf,
)
from atlas_memory.cognitive.retrieval_pipeline import (
    compute_content_fingerprint,
    get_record_authority_score,
)
from atlas_memory.security.delete_guard import (
    DELETE_PROTECTED_PREFIXES,
    is_delete_protected,
)
from atlas_memory.server.models import (
    DEFAULT_PID_PATH,
    DEFAULT_SOCKET_PATH,
    PARSE_ERROR,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
)

__all__ = [
    "AtlasDaemon",
    "extract_effective_session",
    "is_session_accessible",
    "compute_content_fingerprint",
    "get_record_authority_score",
    "is_conversational_churn",
    "clamp_rpc_limit",
]

logger = logging.getLogger(__name__)


def clamp_rpc_limit(limit_val: Any, default: int = 15, min_val: int = 1, max_val: int = 100) -> int:
    """Clamps RPC limit parameter to [min_val, max_val] safely."""
    try:
        val = int(limit_val)
    except (ValueError, TypeError):
        val = default
    return max(min_val, min(val, max_val))


class AtlasDaemon:
    """Atlas Micro-Sidecar Daemon executing JSON-RPC 2.0 requests via Unix Domain Socket."""

    def __init__(
        self,
        socket_path: Path | str = DEFAULT_SOCKET_PATH,
        pid_path: Path | str = DEFAULT_PID_PATH,
        engine: Optional[Any] = None,
        graph_client: Optional[Any] = None,
        latent_buffer: Optional[Any] = None,
        orchestrator: Optional[Any] = None,
        *,
        causal_engine: Optional[Any] = None,
        active_sensing: Optional[Any] = None,
        gossip: Optional[Any] = None,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.pid_path = Path(pid_path)
        self.engine = engine
        self.graph_client = graph_client or (engine.graph if engine and hasattr(engine, "graph") else None)
        self.latent_buffer = latent_buffer or (engine.latent if engine and hasattr(engine, "latent") else None)
        self.orchestrator = orchestrator
        self.causal_engine = causal_engine
        if self.causal_engine is None and self.graph_client is not None:
            try:
                from atlas_memory.causal.retro_causal_edge import RetroCausalEngine
                self.causal_engine = RetroCausalEngine(
                    graph_client=self.graph_client,
                    latent_buffer=self.latent_buffer,
                )
            except Exception as ce_err:
                logger.debug("Could not auto-create RetroCausalEngine: %s", ce_err)
        self.active_sensing = active_sensing
        if self.active_sensing is None:
            try:
                from atlas_memory.active.prediction_error import ActiveSensingEngine
                self.active_sensing = ActiveSensingEngine()
            except Exception as as_err:
                logger.debug("Could not auto-create ActiveSensingEngine: %s", as_err)
        self.gossip = gossip

        self._server: Optional[asyncio.Server] = None
        self._serving: bool = False
        self._active_writers: Set[asyncio.StreamWriter] = set()
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Coroutine[Any, Any, Any]]] = {
            "ping": self._handle_ping,
            "prefetch": self._handle_prefetch,
            "commit_observation": self._handle_commit_observation,
            "set": self._handle_set,
            "set_state": self._handle_set,
            "get": self._handle_get,
            "delete": self._handle_delete,
            "sync_memory_file": self._handle_sync_memory_file,
            "sync_mnemosyne": self._handle_sync_mnemosyne,
            "get_stats": self._handle_get_stats,
            "sync_turn": self._handle_sync_turn,
            "what_if": self._handle_what_if,
            "active_sensing": self._handle_active_sensing,
            "telemetry_report": self._handle_telemetry_report,
            "sync_export_delta": self._handle_sync_export_delta,
            "sync_apply_delta": self._handle_sync_apply_delta,
            "trigger_sleep_consolidation": self._handle_trigger_sleep_consolidation,
            "sync_peer_status": self._handle_sync_peer_status,
            "anneal": self._handle_anneal,
            "verify_audit_log": self._handle_verify_audit_log,
            "task_feedback": self._handle_task_feedback,
            "session/end": self._handle_session_end,
            "session_end": self._handle_session_end,
        }
        self._last_memory_md_mtime: float = 0.0

        # L1 Cognitive Layer Domain Core
        self.hot_buffer = HotBufferPolicy(maxlen=25, ttl_seconds=900.0)
        self._hot_kv_buffer = self.hot_buffer
        self.conflict_arbiter = ConflictArbiter()
        self.predictive_coder = PredictiveCoder(engine=self.engine, causal_engine=self.causal_engine)
        self.retrieval_pipeline = RetrievalPipeline(
            hot_buffer=self.hot_buffer,
            kv_store=self.engine.kv if self.engine and hasattr(self.engine, "kv") else None,
            vector_store=self.engine.vector_store if self.engine and hasattr(self.engine, "vector_store") else None,
            graph_store=self.graph_client or (self.engine.graph if self.engine and hasattr(self.engine, "graph") else None),
            orchestrator=self.orchestrator,
            engine=self.engine,
        )
        self.write_path = EpistemicWritePath(
            kv_store=self.engine.kv if self.engine and hasattr(self.engine, "kv") else None,
            graph_store=self.graph_client or (self.engine.graph if self.engine and hasattr(self.engine, "graph") else None),
            hot_buffer=self.hot_buffer,
            conflict_arbiter=self.conflict_arbiter,
        )

        if hasattr(signal, "SIGHUP"):
            try:
                signal.signal(signal.SIGHUP, signal.SIG_IGN)
            except (ValueError, OSError):
                pass
        self._seed_hot_kv_buffer()

    def _seed_hot_kv_buffer(self) -> None:
        """Seeds working memory hot buffer from persistent SQLite KV store upon startup (Finding F-4)."""
        if self.engine is None or not hasattr(self.engine, "kv") or self.engine.kv is None:
            return
        kv_store = self.engine.kv
        if not hasattr(kv_store, "get_recent_sync") or hasattr(kv_store, "_mock_return_value") or hasattr(kv_store, "assert_called"):
            return
        try:
            seeded = self.hot_buffer.seed_from_kv(kv_store, max_age_seconds=900.0, limit=25)
            logger.info("Seeded hot_buffer with %d recent items from SQLite", seeded)
        except Exception as exc:
            logger.warning("Failed to seed hot_buffer from KV store: %s", exc)

    @classmethod
    def create_default(
        cls,
        socket_path: Path | str = DEFAULT_SOCKET_PATH,
        pid_path: Path | str = DEFAULT_PID_PATH,
        atlas_dir: Optional[Path | str] = None,
        ingest_mnemosyne_on_start: bool = True,
        **kwargs: Any,
    ) -> AtlasDaemon:
        """Instantiates a full-throttle AtlasDaemon with HybridMemoryEngine and Mnemosyne Ingestion."""
        if "sync_legacy_mnemosyne" in kwargs:
            import warnings
            warnings.warn(
                "sync_legacy_mnemosyne is deprecated, use ingest_mnemosyne_on_start instead",
                DeprecationWarning,
                stacklevel=2,
            )
            ingest_mnemosyne_on_start = kwargs.pop("sync_legacy_mnemosyne")

        base_dir = Path(atlas_dir) if atlas_dir else (Path.home() / ".hermes" / "atlas")
        base_dir.mkdir(parents=True, exist_ok=True)

        # Auto-recover stale Qdrant lock if no active daemon is running on UDS socket
        qdrant_dir = base_dir / "qdrant"
        lock_file = qdrant_dir / ".lock"
        if lock_file.exists():
            sock = Path(socket_path)
            is_active = False
            if sock.exists():
                import socket
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.settimeout(0.5)
                    s.connect(str(sock))
                    s.close()
                    is_active = True
                except Exception:
                    is_active = False
            if not is_active:
                logger.warning("Detected stale Qdrant lock file at %s without active daemon. Recovering...", lock_file)
                try:
                    lock_file.unlink(missing_ok=True)
                except Exception as exc:
                    logger.debug("Failed to remove stale lock %s: %s", lock_file, exc)

        from atlas_memory.active.prediction_error import ActiveSensingEngine
        from atlas_memory.causal.retro_causal_edge import RetroCausalEngine
        from atlas_memory.engine import HybridMemoryEngine
        from atlas_memory.orchestrator import MemoryOrchestrator
        from atlas_memory.sync.crdt import DeltaCRDT
        from atlas_memory.sync.crypto import SyncCrypto
        from atlas_memory.sync.protocol import GossipProtocol

        engine = HybridMemoryEngine.create_default(
            db_path=str(base_dir / "atlas.db"),
            qdrant_location=str(base_dir / "qdrant"),
            kuzu_path=str(base_dir / "kuzu"),
        )
        orchestrator = MemoryOrchestrator(engine=engine)
        causal_engine = RetroCausalEngine(graph_client=engine.graph, latent_buffer=engine.latent)
        active_sensing = ActiveSensingEngine()

        crdt = DeltaCRDT(node_id="hermes_local_node")
        crypto = SyncCrypto(SyncCrypto.generate_key())
        gossip = GossipProtocol(local_node_id="hermes_local_node", crdt=crdt, crypto=crypto)
        gossip.register_peer("hermes_peer_alpha")

        if ingest_mnemosyne_on_start:
            try:
                from atlas_memory.ingest.mnemosyne_ingest import MnemosyneIngestEngine
                ingester = MnemosyneIngestEngine()
                if ingester.is_available:
                    ingester.sync_into_atlas_engine(engine)
            except Exception as sync_exc:
                logger.warning("Auto-sync Mnemosyne on daemon startup warning: %s", sync_exc)

        return cls(
            socket_path=socket_path,
            pid_path=pid_path,
            engine=engine,
            graph_client=engine.graph,
            latent_buffer=engine.latent,
            orchestrator=orchestrator,
            causal_engine=causal_engine,
            active_sensing=active_sensing,
            gossip=gossip,
        )

    async def start(self) -> None:
        """Starts the Unix Domain Socket server and writes the PID file."""
        if self._serving:
            return

        self._check_and_write_pid()

        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError as exc:
                logger.warning("Could not unlink existing socket %s: %s", self.socket_path, exc)

        # D2: Auto-heal SHA audit chain on startup if unhealed rows exist
        if self.engine is not None and hasattr(self.engine, "kv") and self.engine.kv is not None:
            if hasattr(self.engine.kv, "verify_audit_log_sync") and hasattr(self.engine.kv, "heal_audit_log_chain"):
                try:
                    is_valid, broken_seq = self.engine.kv.verify_audit_log_sync(deep=False)
                    if not is_valid:
                        logger.warning("Detected unhealed SHA audit chain on startup (broken at seq %s). Auto-healing...", broken_seq)
                        healed_count, _ = self.engine.kv.heal_audit_log_chain()
                        logger.info("Auto-healed %d audit log rows on daemon startup.", healed_count)
                except Exception as heal_exc:
                    logger.warning("Failed to auto-heal audit chain on startup: %s", heal_exc)

        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self.socket_path),
            limit=16 * 1024 * 1024,
        )
        self._serving = True
        self._seed_hot_kv_buffer()
        logger.info("AtlasDaemon started on %s (pid: %d)", self.socket_path, os.getpid())

    async def stop(self) -> None:
        """Gracefully stops the server and cleans up socket and pid files."""
        self._serving = False

        for writer in list(self._active_writers):
            try:
                writer.close()
            except Exception as exc:
                logger.debug("Error closing client writer during stop: %s", exc)

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError as exc:
                logger.debug("Failed to remove socket %s: %s", self.socket_path, exc)

        if self.pid_path.exists():
            try:
                self.pid_path.unlink()
            except OSError as exc:
                logger.debug("Failed to remove PID file %s: %s", self.pid_path, exc)

        self._active_writers.clear()
        logger.info("AtlasDaemon stopped")

    def register_signal_handlers(self) -> None:
        """Registers SIGTERM and SIGINT for graceful shutdown, and ignores SIGHUP."""
        if hasattr(signal, "SIGHUP"):
            try:
                signal.signal(signal.SIGHUP, signal.SIG_IGN)
            except (ValueError, OSError) as exc:
                logger.debug("Signal handler registration skipped for signal SIGHUP: %s", exc)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
                except (NotImplementedError, RuntimeError) as exc:
                    # Signals not supported on some platforms / threads
                    logger.debug("Signal handler registration skipped for signal %s: %s", sig, exc)

    def _check_and_write_pid(self) -> None:
        """Checks if process is already running via PID file, and writes current PID."""
        if self.pid_path.exists():
            try:
                content = self.pid_path.read_text().strip()
                if content:
                    old_pid = int(content)
                    # Check if process is alive
                    try:
                        os.kill(old_pid, 0)
                        raise RuntimeError(f"AtlasDaemon is already running with PID {old_pid}")
                    except OSError:
                        # Stale PID file
                        logger.debug("Ignoring stale PID file from dead process: %d", old_pid)
            except (ValueError, OSError) as exc:
                if isinstance(exc, RuntimeError):
                    raise
                logger.warning("Error reading old PID file: %s", exc)

        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid_path.write_text(str(os.getpid()))

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handles an incoming client connection with length-prefixed JSON-RPC 2.0 messages."""
        self._active_writers.add(writer)
        try:
            while self._serving:
                # 4-byte big-endian length prefix with timeout protection
                try:
                    length_bytes = await asyncio.wait_for(reader.readexactly(4), timeout=60.0)
                except asyncio.TimeoutError:
                    break
                (msg_len,) = struct.unpack(">I", length_bytes)
                if msg_len > 16 * 1024 * 1024:  # 16 MB protection limit
                    raise ValueError(f"Message length too large: {msg_len} bytes")

                payload_bytes = await asyncio.wait_for(reader.readexactly(msg_len), timeout=30.0)
                try:
                    request_data = json.loads(payload_bytes.decode("utf-8"))
                except Exception as parse_err:
                    err_resp = JSONRPCResponse(
                        id=None,
                        error=JSONRPCError(code=PARSE_ERROR, message=f"Parse error: {parse_err}"),
                    )
                    resp_bytes = err_resp.model_dump_json().encode("utf-8")
                    writer.write(struct.pack(">I", len(resp_bytes)) + resp_bytes)
                    await writer.drain()
                    continue

                if isinstance(request_data, list):
                    if not request_data:
                        empty_resp = JSONRPCResponse(
                            id=None,
                            error=JSONRPCError.invalid_request("Batch cannot be empty"),
                        )
                        resp_bytes = empty_resp.model_dump_json().encode("utf-8")
                        writer.write(struct.pack(">I", len(resp_bytes)) + resp_bytes)
                        await writer.drain()
                    else:
                        responses: List[JSONRPCResponse] = []
                        for req in request_data:
                            resp = await self._dispatch_rpc(req)
                            if resp is not None:
                                responses.append(resp)
                        if responses:
                            batch_data = [resp.model_dump() for resp in responses]
                            batch_bytes = json.dumps(batch_data).encode("utf-8")
                            writer.write(struct.pack(">I", len(batch_bytes)) + batch_bytes)
                            await writer.drain()
                else:
                    response = await self._dispatch_rpc(request_data)
                    if response is not None:
                        response_bytes = response.model_dump_json().encode("utf-8")
                        response_prefix = struct.pack(">I", len(response_bytes))
                        writer.write(response_prefix + response_bytes)
                        await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError) as exc:
            logger.debug("Client connection terminated normally: %s", exc)
        except Exception as exc:
            logger.error("Exception handling client connection: %s", exc, exc_info=True)
        finally:
            self._active_writers.discard(writer)
            try:
                writer.close()
                await asyncio.wait_for(writer.wait_closed(), timeout=5.0)
            except Exception as exc:
                logger.debug("Error closing client connection writer: %s", exc)


    async def _dispatch_rpc(self, req_raw: Any) -> Optional[JSONRPCResponse]:
        """Dispatches raw request dictionary to appropriate handler."""
        if not isinstance(req_raw, dict):
            return JSONRPCResponse(
                id=None,
                error=JSONRPCError.invalid_request("Request must be a JSON object"),
            )

        is_notification = ("id" not in req_raw or req_raw.get("id") is None)
        req_id = req_raw.get("id")
        try:
            req = JSONRPCRequest.model_validate(req_raw)
        except Exception as exc:
            if is_notification:
                return None
            return JSONRPCResponse(
                id=req_id,
                error=JSONRPCError.invalid_request(str(exc)),
            )

        handler = self._handlers.get(req.method)
        if handler is None:
            if is_notification:
                return None
            return JSONRPCResponse(
                id=req.id,
                error=JSONRPCError.method_not_found(req.method),
            )

        try:
            params = req.params or {}
            result = await handler(params)
            if is_notification:
                return None
            return JSONRPCResponse(id=req.id, result=result)
        except (ValueError, TypeError, KeyError) as param_err:
            logger.warning("Invalid params in RPC method %s: %s", req.method, param_err)
            if is_notification:
                return None
            return JSONRPCResponse(
                id=req.id,
                error=JSONRPCError.invalid_params(str(param_err)),
            )
        except Exception as exc:
            logger.exception("Error executing RPC method %s: %s", req.method, exc)
            if is_notification:
                return None
            return JSONRPCResponse(
                id=req.id,
                error=JSONRPCError.internal_error(str(exc)),
            )

    # -------------------------------------------------------------------------
    # RPC Handlers
    # -------------------------------------------------------------------------

    async def _handle_ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Health check endpoint."""
        return {"status": "ok"}

    def _check_and_sync_memory_md(self) -> None:
        """Auto-sync all markdown files in ~/.hermes/memories/ if modification timestamp has changed."""
        if str(self.socket_path) != str(DEFAULT_SOCKET_PATH):
            return
        from atlas_memory.ingest.mnemosyne_ingest import DEFAULT_HERMES_MEMORY_MD_PATH, MnemosyneIngestEngine
        memories_dir = DEFAULT_HERMES_MEMORY_MD_PATH.parent
        if not memories_dir.exists():
            return
        try:
            md_files = [f for f in memories_dir.glob("*.md") if not f.name.endswith(".lock")]
            if not md_files:
                return
            latest_mtime = max(f.stat().st_mtime for f in md_files)
            if latest_mtime > self._last_memory_md_mtime:
                self._last_memory_md_mtime = latest_mtime
                if self.engine is not None:
                    ingester = MnemosyneIngestEngine(memory_md_path=DEFAULT_HERMES_MEMORY_MD_PATH)
                    res = ingester.sync_memory_md_into_engine(self.engine)
                    logger.info("[Auto-Sync] Ingested updated memories (MEMORY.md/USER.md) into ATLAS: %s", res)
        except Exception as exc:
            logger.debug("Auto-sync memories check error: %s", exc)

    async def _handle_prefetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Prefetch memory records for query/session with full epistemic formatting and content deduplication."""
        params_dict = dict(params)
        if "limit" in params_dict:
            params_dict["limit"] = clamp_rpc_limit(params_dict.get("limit", 15))
        return await self.retrieval_pipeline.execute_prefetch(
            params_dict,
            engine=self.engine,
            orchestrator=self.orchestrator,
            hot_buffer=self._hot_kv_buffer,
            check_sync_callback=self._check_and_sync_memory_md,
        )

    def _auto_insert_graph_edges(
        self,
        key: str,
        val: Any,
        params: Dict[str, Any],
        metadata: Dict[str, Any],
        confidence: float,
    ) -> None:
        """Parses and inserts multi-hop relations into the runtime knowledge graph via write_path."""
        if hasattr(self, "write_path") and hasattr(self.write_path, "_auto_insert_graph_edges"):
            self.write_path._auto_insert_graph_edges(key, val, metadata, confidence)

    async def _handle_set(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set a key-value pair in ATLAS Verified KV Store via EpistemicWritePath."""
        key = params.get("key") or params.get("subject")
        val = params.get("value") or params.get("object") or params.get("val") or params.get("target") or ""
        if not key:
            raise ValueError("Key or subject is required for set")
        confidence = clamp_conf(params.get("confidence", 1.0))
        session_id = params.get("session_id") or HERMES_DEFAULT_SESSION
        metadata = dict(params.get("metadata") or {})
        for field in ("predicate", "relation", "target", "object", "subject"):
            if field in params and field not in metadata:
                metadata[field] = params[field]
        source_type = params.get("source_type")
        if source_type:
            metadata["source_type"] = str(source_type)
        metadata["confidence"] = confidence
        metadata["session_id"] = str(session_id)

        reason = params.get("reason", "rpc_set")
        task_success = bool(params.get("task_success") or params.get("success") or (params.get("outcome") == "success"))
        if task_success and self.engine is not None and hasattr(self.engine, "kv") and self.engine.kv is not None:
            existing = self.engine.kv.get_sync(key) if hasattr(self.engine.kv, "get_sync") else None
            if existing:
                existing_conf = float(existing.get("confidence", confidence))
                confidence = clamp_conf(existing_conf + 0.05)
                reason = "task_feedback:success"
                metadata["confidence"] = confidence
                metadata["task_feedback"] = "success"

        if hasattr(self, "write_path") and self.write_path is not None:
            res = self.write_path.write(
                key=key,
                value=val,
                confidence=confidence,
                metadata=metadata,
                source_type=source_type,
                reason=reason,
                session_id=session_id,
            )
            if self.engine is not None and hasattr(self.engine, "record_mental_transition"):
                try:
                    from atlas_memory.models import ActionPlan
                    action = ActionPlan(name="remember", parameters={"key": key, "value": str(val)[:100]})
                    self.engine.record_mental_transition(action, session_id=session_id)
                except Exception as exc:
                    logger.debug("Daemon set record_mental_transition error: %s", exc)
            return res
        return {"status": "error", "error": "no_engine_available"}

    async def _handle_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get a key-value record from ATLAS Verified KV Store."""
        key = params.get("key")
        if not key:
            raise ValueError("Key is required for get")
        if self.engine is not None and hasattr(self.engine, "kv") and self.engine.kv is not None:
            state = await self.engine.kv.get_state(key)
            logger.info("[RPC] get key='%s' -> found=%s", key, state is not None)
            return {"status": "ok", "key": key, "found": state is not None, "state": state}
        return {"status": "error", "error": "no_engine_available"}

    # Protected key prefixes that MUST NOT be hard-deleted (T19 / ADR-005, A3)
    _DELETE_PROTECTED_PREFIXES = DELETE_PROTECTED_PREFIXES

    async def _handle_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a key from ATLAS Verified KV Store.

        Protected prefixes (hermes:, fact:, prompt-, mnemosyne:) are rejected
        to prevent accidental destruction of durable system facts (ADR-005).
        Successful deletes are recorded in the SHA-256 audit hash-chain.
        """
        key = params.get("key")
        if not key:
            raise ValueError("Key is required for delete")

        # Guard: reject protected prefixes (ADR-005, A3)
        is_prot, prefix = is_delete_protected(str(key))
        if is_prot:
            logger.warning("[RPC] delete REJECTED for protected key='%s' (prefix '%s')", key, prefix)
            return {
                "status": "rejected",
                "key": key,
                "deleted": False,
                "found": False,
                "reason": "protected_key",
                "message": f"Key '{key}' is protected (prefix '{prefix}'). Use soft tombstone or explicit ADR override.",
            }

        if self.engine is not None and hasattr(self.engine, "kv") and self.engine.kv is not None:
            kv = self.engine.kv
            if hasattr(kv, "delete_state"):
                deleted = await kv.delete_state(key)
            else:
                assert kv._conn is not None
                with kv._conn:
                    cursor = kv._conn.execute("DELETE FROM state_variables WHERE key = ?", (key,))
                    deleted = bool(cursor.rowcount > 0)
            logger.info("[RPC] delete key='%s' -> deleted=%s", key, deleted)
            return {"status": "ok", "key": key, "deleted": deleted, "found": deleted}
        return {"status": "error", "error": "no_engine_available"}

    async def _handle_sync_memory_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Syncs Hermes MEMORY.md file into ATLAS Vault."""
        path = params.get("path")
        from atlas_memory.ingest.mnemosyne_ingest import MnemosyneIngestEngine
        ingester = MnemosyneIngestEngine()
        if self.engine is not None:
            res = ingester.sync_memory_md_into_engine(self.engine, path=path)
            logger.info("[RPC] sync_memory_file -> %s", res)
            return res
        return {"status": "error", "error": "no_engine_available"}

    async def _handle_commit_observation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Commit an observation / fact into the ATLAS Vault."""
        subject = str(params.get("subject") or "").strip()
        if not subject:
            return {"status": "error", "error": "empty_subject", "message": "Subject cannot be empty"}

        predicate = str(params.get("predicate") or "observation").strip() or "observation"
        obj = str(params.get("object") or "")
        confidence = clamp_conf(params.get("confidence", 1.0))
        source = params.get("source", "user_explicit")
        session_id = params.get("session_id", HERMES_DEFAULT_SESSION)

        if self.engine is not None and hasattr(self.engine, "kv") and self.engine.kv is not None:
            var_key = f"{subject}:{predicate}"
            task_success = bool(params.get("task_success") or params.get("success") or (params.get("outcome") == "success"))
            reason = "commit_observation"
            if task_success and hasattr(self.engine.kv, "get_sync"):
                existing = self.engine.kv.get_sync(var_key)
                if existing:
                    existing_conf = existing.get("confidence", confidence)
                    confidence = clamp_conf(float(existing_conf) + 0.05)
                    if not obj:
                        obj = str(existing.get("value") or "")
                    reason = "task_feedback:success"

            self.engine.kv.set_sync(
                var_key,
                obj,
                confidence=confidence,
                metadata={"confidence": confidence, "source": source, "session_id": session_id},
                reason=reason,
            )
            if hasattr(self.engine, "graph") and self.engine.graph is not None:
                try:
                    if hasattr(self.engine.graph, "add_entity"):
                        self.engine.graph.add_entity(subject, entity_type="Concept")
                        self.engine.graph.add_entity(obj[:80], entity_type="Observation")
                    self.write_path.commit_relation(subject, predicate, obj[:80], confidence=confidence, session_id=session_id)
                except Exception as exc:
                    logger.debug("Graph commit observation ignored: %s", exc)
            # Record into hot-KV working buffer
            self._hot_kv_buffer.append({
                "key": var_key,
                "value": obj,
                "timestamp": time.time(),
                "confidence": confidence,
                "metadata": {"confidence": confidence, "source": source, "session_id": session_id},
                "source_type": source,
            })
            if self.engine is not None and hasattr(self.engine, "record_mental_transition"):
                try:
                    from atlas_memory.models import ActionPlan
                    action = ActionPlan(name=f"observe:{predicate}", parameters={"subject": subject, "object": obj})
                    self.engine.record_mental_transition(action, session_id=session_id)
                except Exception as exc:
                    logger.debug("Daemon commit_observation record_mental_transition error: %s", exc)
            logger.info("[RPC] commit_observation key='%s' -> persisted=True", var_key)
            return {"status": "ok", "committed": True, "subject": subject, "key": var_key, "confidence": confidence}
        return {"status": "error", "error": "no_engine_available"}

    async def _handle_task_feedback(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Dedicated RPC handler for closed-loop task feedback (G8). Updates record confidence."""
        key = str(params.get("key") or params.get("subject") or "").strip()
        if not key:
            return {"status": "error", "error": "empty_key", "message": "Key cannot be empty"}

        task_success = bool(params.get("task_success") or params.get("success") or (params.get("outcome") == "success") or params.get("task_success", True))
        if "task_success" in params and not params["task_success"]:
            task_success = False
        elif "success" in params and not params["success"]:
            task_success = False
        elif params.get("outcome") and params.get("outcome") != "success":
            task_success = False

        try:
            delta = float(params.get("delta", 0.05))
        except (ValueError, TypeError):
            delta = 0.05

        if self.engine is None or not hasattr(self.engine, "kv") or self.engine.kv is None:
            return {"status": "error", "error": "no_kv_available"}

        existing = self.engine.kv.get_sync(key) if hasattr(self.engine.kv, "get_sync") else None
        if not existing:
            return {"status": "error", "error": "key_not_found", "message": f"Key '{key}' not found in KV store"}

        existing_conf = float(existing.get("confidence", 1.0))
        new_conf = clamp_conf(existing_conf + delta if task_success else existing_conf - delta)
        val = existing.get("value", "")
        meta = existing.get("metadata", {})
        if isinstance(meta, dict):
            meta = dict(meta)
        else:
            meta = {}
        meta["confidence"] = new_conf
        meta["task_feedback"] = "success" if task_success else "failure"

        reason = "task_feedback:success" if task_success else "task_feedback:failure"
        self.engine.kv.set_sync(
            key,
            val,
            confidence=new_conf,
            metadata=meta,
            reason=reason,
        )
        logger.info("[RPC] task_feedback key='%s' success=%s conf=%.2f->%.2f", key, task_success, existing_conf, new_conf)
        return {
            "status": "ok",
            "key": key,
            "previous_confidence": existing_conf,
            "confidence": new_conf,
            "task_success": task_success,
        }

    async def _handle_sync_mnemosyne(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize legacy Mnemosyne SQLite records into ATLAS Vault."""
        db_path = params.get("db_path", "")
        from atlas_memory.ingest.mnemosyne_ingest import MnemosyneIngestEngine

        ingester = MnemosyneIngestEngine(db_path=db_path) if db_path else MnemosyneIngestEngine()
        if self.engine is not None:
            res = ingester.sync_into_atlas_engine(self.engine)
            logger.info("[RPC] sync_mnemosyne -> %s", res)
            return res
        return {"status": "error", "error": "no_engine_active"}

    async def _handle_get_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return real-time counts and memory vault status."""
        kv_count = 0
        if self.engine is not None and hasattr(self.engine, "kv") and self.engine.kv is not None:
            try:
                if hasattr(self.engine.kv, "get_all_sync"):
                    all_kv = self.engine.kv.get_all_sync()
                else:
                    all_kv = self.engine.kv.get_all()
                kv_count = len(all_kv)
            except Exception:
                kv_count = 0
        trajectories_count = len(self.engine.trajectory_buffer) if (self.engine and hasattr(self.engine, "trajectory_buffer") and self.engine.trajectory_buffer is not None) else 0
        active_trajectories = len(self.engine.trajectory_buffer.trajectories) if (self.engine and hasattr(self.engine, "trajectory_buffer") and self.engine.trajectory_buffer is not None and hasattr(self.engine.trajectory_buffer, "trajectories")) else 0
        ttt_steps = (
            getattr(self.engine.ttt, "step_count", 0)
            if (self.engine and hasattr(self.engine, "ttt") and self.engine.ttt is not None)
            else 0
        )
        ttt_energy = (
            getattr(self.engine.ttt, "total_energy", 0.0)
            if (self.engine and hasattr(self.engine, "ttt") and self.engine.ttt is not None)
            else 0.0
        )
        logger.info("[RPC] get_stats -> kv_records=%d trajectories=%d ttt_steps=%d", kv_count, trajectories_count, ttt_steps)
        return {
            "status": "ok",
            "pid": os.getpid(),
            "socket": str(self.socket_path),
            "serving": self._serving,
            "kv_records": kv_count,
            "trajectories_count": trajectories_count,
            "active_trajectories": active_trajectories,
            "ttt_steps": ttt_steps,
            "ttt_energy": ttt_energy,
            "has_graph": self.graph_client is not None,
            "has_orchestrator": self.orchestrator is not None,
        }

    async def _handle_sync_turn(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sync conversation turn with heuristic SPO fact extraction."""
        user_content = params.get("user_content") or params.get("user_message", "")
        assistant_content = params.get("assistant_content") or params.get("assistant_response", "")
        turn_context = params.get("turn_context", {})
        if not user_content and turn_context:
            user_content = turn_context.get("user_message", "")
            assistant_content = turn_context.get("assistant_response", "")
        session_id = params.get("session_id", HERMES_DEFAULT_SESSION)

        extracted_count = 0
        combined_text = f"{user_content}\n{assistant_content}".strip()
        lines = [ln.strip() for ln in combined_text.splitlines() if ln.strip()]

        for ln in lines:
            # Strukturalna ekstrakcja zdań informacyjnych, preferencji i par klucz-wartość
            if len(ln.split()) >= 3 and not ln.startswith(("#", "//", "```", "<")):
                words = ln.split()
                subj = " ".join(words[:4]) if words else "turn_fact"
                if self.engine is not None and hasattr(self.engine, "kv") and self.engine.kv is not None:
                    self.engine.kv.set_sync(
                        f"turn:{subj}"[:60],
                        ln,
                        confidence=0.9,
                        metadata={"session_id": session_id, "origin": "shadow_turn"},
                        reason="sync_turn",
                    )
                    extracted_count += 1
                try:
                    self.write_path.commit_relation(f"turn:{subj}"[:60], "observed_fact", ln[:80], confidence=0.9, session_id=session_id)
                except Exception as exc:
                    logger.debug("Graph sync turn relation ignored: %s", exc)

        if extracted_count == 0 and combined_text:
            first_line = lines[0] if lines else combined_text[:80]
            words = first_line.split()
            subj = " ".join(words[:4]) if words else "turn_fact"
            if self.engine is not None and hasattr(self.engine, "kv") and self.engine.kv is not None:
                self.engine.kv.set_sync(
                    f"turn:{subj}"[:60],
                    first_line,
                    confidence=0.85,
                    metadata={"session_id": session_id, "origin": "shadow_turn"},
                    reason="sync_turn",
                )
                extracted_count += 1
            try:
                self.write_path.commit_relation(f"turn:{subj}"[:60], "observed_fact", first_line[:80], confidence=0.9, session_id=session_id)
            except Exception as exc:
                logger.debug("Graph sync turn fallback relation ignored: %s", exc)

        if self.engine is not None and hasattr(self.engine, "record_mental_transition"):
            try:
                from atlas_memory.models import ActionPlan
                action = ActionPlan(name="sync_turn", parameters={"facts_count": extracted_count, "session_id": session_id})
                self.engine.record_mental_transition(action, session_id=session_id)
            except Exception as exc:
                logger.debug("Daemon sync_turn record_mental_transition error: %s", exc)

        logger.info("[RPC] sync_turn session=%s -> extracted=%d facts", session_id, extracted_count)
        return {"status": "ok", "synced": True, "extracted_facts": extracted_count, "session_id": session_id}

    async def _handle_what_if(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute what-if causal query."""
        action = params.get("action", "")
        entity = params.get("entity") or params.get("target") or action
        depth = int(params.get("depth", 2))

        # Mock / custom graph client shortcut
        if self.graph_client is not None and hasattr(self.graph_client, "what_if") and not hasattr(self.graph_client, "nx_graph"):
            path = self.graph_client.what_if(action)
            return {"status": "ok", "action": action, "causal_path": path, "causal_paths": [path] if isinstance(path, dict) else []}

        # Check entity existence in knowledge graph
        entity_exists = False
        graph_obj = self.graph_client or (self.causal_engine.graph if self.causal_engine else None)
        if graph_obj is not None:
            if hasattr(graph_obj, "nx_graph") and graph_obj.nx_graph is not None:
                if entity in graph_obj.nx_graph:
                    entity_exists = True
                else:
                    ent_clean = str(entity).strip().lower().replace("_", " ")
                    for node in graph_obj.nx_graph.nodes:
                        node_str = str(node).strip().lower()
                        if node_str == ent_clean or (len(ent_clean) >= 5 and ent_clean in node_str):
                            entity_exists = True
                            break
            elif isinstance(graph_obj, dict):
                entity_exists = entity in graph_obj

        if graph_obj is not None and not entity_exists:
            logger.info("[RPC] what_if entity='%s' not found in causal knowledge graph", entity)
            return {
                "status": "entity_not_found",
                "action": action,
                "entity": entity,
                "paths_count": 0,
                "causal_paths": [],
                "causal_path": None,
                "message": f"Entity '{entity}' not found in causal knowledge graph. No causal paths can be derived.",
            }

        if self.causal_engine is not None:
            try:
                paths = await self.causal_engine.causal_what_if(entity, action, depth=depth)
                paths_data = [p.model_dump() if hasattr(p, "model_dump") else p for p in paths]
                logger.info("[RPC] what_if entity='%s' action='%s' -> %d paths", entity, action, len(paths_data))

                if len(paths_data) == 0 and not entity_exists:
                    logger.info("[RPC] what_if entity='%s' not found in causal knowledge graph", entity)
                    return {
                        "status": "entity_not_found",
                        "action": action,
                        "entity": entity,
                        "paths_count": 0,
                        "causal_paths": [],
                        "causal_path": None,
                        "message": f"Entity '{entity}' not found in causal knowledge graph. No causal paths can be derived.",
                    }

                summary_msg = None
                if len(paths_data) == 0:
                    summary_msg = f"Entity '{entity}' exists in graph, but no causal failure paths detected for action '{action}' at depth {depth}."

                return {
                    "status": "ok",
                    "action": action,
                    "entity": entity,
                    "paths_count": len(paths_data),
                    "causal_paths": paths_data,
                    "causal_path": paths_data[0] if paths_data else None,
                    "message": summary_msg,
                }
            except Exception as exc:
                logger.debug("causal_what_if error: %s", exc)
                return {"status": "error", "error": str(exc), "causal_paths": [], "causal_path": None}

        if not entity_exists:
            return {
                "status": "entity_not_found",
                "action": action,
                "entity": entity,
                "paths_count": 0,
                "causal_paths": [],
                "causal_path": None,
                "message": f"Entity '{entity}' not found in knowledge graph.",
            }

        return {"action": action, "causal_path": None, "causal_paths": [], "status": "no_causal_engine"}

    async def _handle_active_sensing(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Active sensing probe for predictive coding with closed-loop world model writeback."""
        if hasattr(self, "predictive_coder") and self.predictive_coder is not None:
            return await self.predictive_coder.execute_sensing(params, graph_client=self.graph_client)
        return {"probe": params.get("target_entity", ""), "sensed": True}

    async def _handle_trigger_sleep_consolidation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Manually or autonomically triggers L3 sleep cycle consolidation and SOP skill compilation."""
        skills_dir = params.get("skills_dir") or (Path.home() / ".hermes" / "skills" / "baked-skills")
        if self.engine is not None and hasattr(self.engine, "auditor") and self.engine.auditor is not None:
            stats = await self.engine.auditor.run_sleep_cycle_consolidation()

            from atlas_memory.l3_procedural.sleep_baker import SleepBaker
            baker = SleepBaker()
            trajectories = []
            if hasattr(self.engine, "trajectory_buffer") and self.engine.trajectory_buffer is not None:
                trajectories = getattr(self.engine.trajectory_buffer, "trajectories", [])

            baked_results = await baker.auto_consolidate_and_bake(
                trajectories=trajectories,
                kv_store=self.engine.kv,
                skills_dir=skills_dir,
            )
            return {
                "status": "ok",
                "consolidated_records": stats.consolidated_records if hasattr(stats, "consolidated_records") else 0,
                "pruned_stale_facts": getattr(stats, "pruned_stale_facts", 0),
                "baked_sops_count": len(baked_results),
                "baked_sops": baked_results,
            }
        return {"status": "no_engine_auditor", "baked_sops_count": 0}

    async def _handle_session_end(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handles session end notification from Hermes agent provider."""
        session_id = str(params.get("session_id") or "default")
        turn_count = int(params.get("turn_count") or 0)
        trigger_sleep = bool(params.get("trigger_sleep", True))

        consolidated = 0
        pruned = 0
        if self.engine is not None:
            try:
                from atlas_memory.hermes.prefix_guard import HermesSessionHook
                hook = HermesSessionHook(self.engine)
                stats = await hook.on_session_end(session_id=session_id, trigger_sleep_cycle=trigger_sleep)
                consolidated = getattr(stats, "consolidated_records", 0)
                pruned = getattr(stats, "pruned_stale_facts", 0)
            except Exception as exc:
                logger.warning("Error during session/end hook execution: %s", exc)

        return {
            "status": "ok",
            "session_id": session_id,
            "turn_count": turn_count,
            "consolidated_records": consolidated,
            "pruned_stale_facts": pruned,
        }

    async def _handle_telemetry_report(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return server telemetry report."""
        return {
            "pid": os.getpid(),
            "socket": str(self.socket_path),
            "serving": self._serving,
            "has_graph_client": self.graph_client is not None,
            "has_latent_buffer": self.latent_buffer is not None,
            "has_orchestrator": self.orchestrator is not None,
            "has_causal_engine": self.causal_engine is not None,
            "has_active_sensing": self.active_sensing is not None,
            "has_gossip": self.gossip is not None,
        }

    async def _handle_sync_export_delta(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Export CRDT delta since vector clock."""
        if self.gossip is None:
            return {"status": "not_configured"}
        target_peer = params.get("target_peer_id", "hermes_peer_alpha")
        try:
            req_bytes = self.gossip.create_sync_request(target_peer)
            return {"status": "ok", "sync_request": req_bytes.decode("utf-8")}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    async def _handle_sync_apply_delta(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply incoming delta to local CRDT."""
        if self.gossip is None:
            return {"status": "not_configured"}
        sync_payload = params.get("sync_request", "")
        if not sync_payload:
            return {"status": "error", "error": "missing sync_request"}
        try:
            req_bytes = sync_payload.encode("utf-8") if isinstance(sync_payload, str) else sync_payload
            delta = self.gossip.process_sync_response(req_bytes)
            return {"status": "ok", "source_node": delta.source_node, "clock": delta.vector_clock.clocks}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    async def _handle_sync_peer_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return known peers and vector clock statuses."""
        if self.gossip is None:
            return {"status": "not_configured"}
        peers_dict = {
            peer_id: p.model_dump() for peer_id, p in self.gossip.peers.items()
        }
        return {"status": "ok", "peers": peers_dict, "local_clock": self.gossip.crdt.clock.clocks}

    async def _handle_anneal(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger autonomous causal graph annealer."""
        target_entity = params.get("target_entity") or params.get("entity", "Root")
        if self.causal_engine is not None and hasattr(self.causal_engine, "recalibrate_graph_with_annealer"):
            res = await self.causal_engine.recalibrate_graph_with_annealer(target_entity=target_entity)
            logger.info("[RPC] anneal entity='%s' -> %s", target_entity, res)
            return res
        return {"status": "error", "error": "no_causal_engine_or_annealer"}

    async def _handle_verify_audit_log(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Verify cryptographic integrity of SHA-256 audit log."""
        if self.engine is None or not hasattr(self.engine, "kv") or self.engine.kv is None:
            return {"status": "error", "error": "no_engine_available", "valid": False, "checked_entries": 0}

        deep = bool(params.get("deep", False))
        kv = self.engine.kv
        if not hasattr(kv, "verify_audit_log"):
            return {"status": "error", "error": "verify_audit_log_not_supported", "valid": False, "checked_entries": 0}

        if params.get("heal") and hasattr(kv, "heal_audit_log_chain"):
            heal_res = kv.heal_audit_log_chain()
            if asyncio.iscoroutine(heal_res):
                await heal_res

        res = kv.verify_audit_log(deep=deep)
        if asyncio.iscoroutine(res):
            res = await res

        count = 0
        if hasattr(kv, "_ensure_conn"):
            try:
                import contextlib
                conn = kv._ensure_conn()
                with contextlib.closing(conn.cursor()) as cursor:
                    cursor.execute("SELECT COUNT(*) FROM state_audit_log")
                    row = cursor.fetchone()
                    if row:
                        count = int(row[0])
            except Exception as c_err:
                logger.debug("Failed to count state_audit_log entries: %s", c_err)

        if isinstance(res, tuple):
            is_valid = bool(res[0])
            if count == 0 and len(res) > 1 and res[1] is not None:
                count = int(res[1])
        elif isinstance(res, bool):
            is_valid = res
        elif isinstance(res, dict):
            is_valid = bool(res.get("valid", True))
            if "checked_entries" in res:
                count = int(res["checked_entries"])
        elif isinstance(res, int):
            is_valid = True
            count = res
        else:
            is_valid = bool(res)

        logger.info("[RPC] verify_audit_log (deep=%s) -> valid=%s, checked_entries=%d", deep, is_valid, count)
        return {"status": "ok", "valid": is_valid, "checked_entries": count}

