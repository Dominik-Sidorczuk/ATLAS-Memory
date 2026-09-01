"""AtlasDaemon — Micro-Sidecar server for ATLAS over Unix Domain Socket with JSON-RPC 2.0."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import struct
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, Optional, Set

from atlas_memory.server.models import (
    DEFAULT_PID_PATH,
    DEFAULT_SOCKET_PATH,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
)

logger = logging.getLogger(__name__)


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
        self.active_sensing = active_sensing
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
            "sync_peer_status": self._handle_sync_peer_status,
        }
        self._last_memory_md_mtime: float = 0.0

    @classmethod
    def create_default(
        cls,
        socket_path: Path | str = DEFAULT_SOCKET_PATH,
        pid_path: Path | str = DEFAULT_PID_PATH,
        atlas_dir: Optional[Path | str] = None,
        sync_legacy_mnemosyne: bool = True,
    ) -> AtlasDaemon:
        """Instantiates a full-throttle AtlasDaemon with HybridMemoryEngine and Mnemosyne Ingestion."""
        base_dir = Path(atlas_dir) if atlas_dir else (Path.home() / ".hermes" / "atlas")
        base_dir.mkdir(parents=True, exist_ok=True)

        from atlas_memory.engine import HybridMemoryEngine
        from atlas_memory.orchestrator import MemoryOrchestrator
        from atlas_memory.causal.retro_causal_edge import RetroCausalEngine
        from atlas_memory.active.prediction_error import ActiveSensingEngine
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

        if sync_legacy_mnemosyne:
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

        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self.socket_path),
        )
        self._serving = True
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
        """Registers SIGTERM and SIGINT for graceful shutdown."""
        loop = asyncio.get_running_loop()
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
                # 4-byte big-endian length prefix
                length_bytes = await reader.readexactly(4)
                (msg_len,) = struct.unpack(">I", length_bytes)
                if msg_len > 16 * 1024 * 1024:  # 16 MB protection limit
                    raise ValueError(f"Message length too large: {msg_len} bytes")

                payload_bytes = await reader.readexactly(msg_len)
                request_dict = json.loads(payload_bytes.decode("utf-8"))

                response = await self._dispatch_rpc(request_dict)
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
                await writer.wait_closed()
            except Exception as exc:
                logger.debug("Error closing client connection writer: %s", exc)


    async def _dispatch_rpc(self, req_raw: Dict[str, Any]) -> JSONRPCResponse:
        """Dispatches raw request dictionary to appropriate handler."""
        req_id = req_raw.get("id")
        try:
            req = JSONRPCRequest.model_validate(req_raw)
        except Exception as exc:
            return JSONRPCResponse(
                id=req_id,
                error=JSONRPCError(code=-32600, message=f"Invalid Request: {exc}"),
            )

        handler = self._handlers.get(req.method)
        if handler is None:
            return JSONRPCResponse(
                id=req.id,
                error=JSONRPCError(code=-32601, message=f"Method not found: {req.method}"),
            )

        try:
            params = req.params or {}
            result = await handler(params)
            return JSONRPCResponse(id=req.id, result=result)
        except Exception as exc:
            logger.exception("Error executing RPC method %s: %s", req.method, exc)
            return JSONRPCResponse(
                id=req.id,
                error=JSONRPCError(code=-32603, message=f"Internal error: {exc}"),
            )

    # -------------------------------------------------------------------------
    # RPC Handlers
    # -------------------------------------------------------------------------

    async def _handle_ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Health check endpoint."""
        return {"status": "ok"}

    def _check_and_sync_memory_md(self) -> None:
        """Auto-sync all markdown files in ~/.hermes/memories/ if modification timestamp has changed."""
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
        """Prefetch memory records for query/session with full epistemic formatting."""
        query = params.get("query", "")
        session_id = params.get("session_id", "")
        formatted_context = ""
        records_list: List[Dict[str, Any]] = []
        seen_keys: Set[str] = set()

        # 0. Auto-sync MEMORY.md if modified
        self._check_and_sync_memory_md()

        # 1. Search KV store for fresh state variables and keyword/stem matches
        if self.engine is not None and hasattr(self.engine, "kv") and self.engine.kv is not None:
            try:
                all_kv = self.engine.kv.get_all()
                q_lower = query.lower()
                words = [w.strip(".,!?:;\"'()[]{}") for w in q_lower.split() if len(w) > 2]
                stems = [w[:4] if len(w) >= 5 else w for w in words if len(w) >= 3]
                
                for k, item in all_kv.items():
                    val_str = str(item.get("value", "")).lower()
                    k_lower = k.lower()
                    
                    is_exact_key_match = (k_lower in q_lower) or (q_lower in k_lower)
                    is_match = is_exact_key_match or any(w in k_lower or w in val_str for w in words) or any(s in k_lower or s in val_str for s in stems)
                    if is_match and k not in seen_keys:
                        seen_keys.add(k)
                        is_native_state = not k.startswith("mnemosyne:")
                        if is_exact_key_match:
                            score = 1.0
                        elif is_native_state:
                            score = 0.95
                        else:
                            score = 0.75

                        records_list.append({
                            "subject": k,
                            "predicate": "state_variable" if is_native_state else "fact",
                            "object": str(item.get("value", "")),
                            "importance_score": score,
                            "source_type": "user_explicit",
                        })
            except Exception as kv_scan_exc:
                logger.debug("KV scan failed: %s", kv_scan_exc)

        # 2. Orchestrated recall (vector + graph + mnemosyne)
        if self.orchestrator is not None:
            if hasattr(self.orchestrator, "orchestrated_recall"):
                try:
                    records = await self.orchestrator.orchestrated_recall(query, session_id=session_id)
                    for r in records:
                        rec_dict = r.model_dump() if hasattr(r, "model_dump") else r
                        subj = rec_dict.get("subject", "")
                        if subj not in seen_keys:
                            seen_keys.add(subj)
                            records_list.append(rec_dict)
                except Exception as rec_exc:
                    logger.debug("orchestrated_recall failed: %s", rec_exc)

        # Sort records by importance_score descending (User explicit state variables > Facts > Mnemosyne)
        records_list.sort(key=lambda r: float(r.get("importance_score", 0.5)), reverse=True)
        records_list = records_list[:10]

        # Build clean formatted context block
        if records_list:
            lines = ["## ATLAS Cognitive Context (No-GIL Hardware Memory)"]
            for rec in records_list:
                subj = rec.get("subject", "")
                obj = rec.get("object", "")
                lines.append(f"• [{subj}] {obj}")
            formatted_context = "\n".join(lines)

        logger.info("[RPC] prefetch query='%s' -> %d records", query[:50], len(records_list))
        return {
            "records": records_list,
            "context_block": formatted_context,
            "query": query,
            "count": len(records_list),
        }

    async def _handle_set(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set a key-value pair in ATLAS Verified KV Store with SHA-256 audit log."""
        key = params.get("key") or params.get("subject")
        val = params.get("value") or params.get("object") or params.get("val")
        if not key:
            raise ValueError("Key or subject is required for set")
        confidence = float(params.get("confidence", 1.0))
        metadata = params.get("metadata") or {}
        reason = params.get("reason", "rpc_set")

        if self.engine is not None and hasattr(self.engine, "kv") and self.engine.kv is not None:
            self.engine.kv.set_sync(key, val, confidence=confidence, metadata=metadata, reason=reason)
            logger.info("[RPC] set key='%s' -> persisted=True", key)
            return {"status": "ok", "key": key, "persisted": True}
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

    async def _handle_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a key from ATLAS Verified KV Store."""
        key = params.get("key")
        if not key:
            raise ValueError("Key is required for delete")
        if self.engine is not None and hasattr(self.engine, "kv") and self.engine.kv is not None:
            assert self.engine.kv._conn is not None
            with self.engine.kv._conn:
                self.engine.kv._conn.execute("DELETE FROM state_variables WHERE key = ?", (key,))
            logger.info("[RPC] delete key='%s' -> deleted=True", key)
            return {"status": "ok", "key": key, "deleted": True}
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
        subject = params.get("subject", "")
        predicate = params.get("predicate", "observation")
        obj = params.get("object", "")
        confidence = float(params.get("confidence", 1.0))
        source = params.get("source", "user_explicit")
        session_id = params.get("session_id", "hermes_default")

        if self.engine is not None and hasattr(self.engine, "kv") and self.engine.kv is not None:
            var_key = f"{subject}:{predicate}"
            self.engine.kv.set_sync(
                var_key,
                obj,
                confidence=confidence,
                metadata={"confidence": confidence, "source": source, "session_id": session_id},
                reason="commit_observation",
            )
            if hasattr(self.engine, "graph") and self.engine.graph is not None:
                try:
                    self.engine.graph.add_entity(subject, entity_type="Concept")
                    self.engine.graph.add_entity(obj[:80], entity_type="Observation")
                    self.engine.graph.add_relation(subject, predicate, obj[:80])
                except Exception:
                    pass
            logger.info("[RPC] commit_observation key='%s' -> persisted=True", var_key)
            return {"status": "ok", "committed": True, "subject": subject, "key": var_key}
        return {"status": "error", "error": "no_engine_available"}

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
                kv_count = len(self.engine.kv.get_all())
            except Exception:
                kv_count = 0
        logger.info("[RPC] get_stats -> kv_records=%d", kv_count)
        return {
            "status": "ok",
            "pid": os.getpid(),
            "socket": str(self.socket_path),
            "serving": self._serving,
            "kv_records": kv_count,
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
        session_id = params.get("session_id", "hermes_default")

        extracted_count = 0
        combined_text = f"{user_content}\n{assistant_content}".strip()
        lines = [ln.strip() for ln in combined_text.splitlines() if ln.strip()]

        for ln in lines:
            if any(k in ln.lower() for k in ("preferuj", "język", "language", "klowstack", "peptyd", "obsidian", "mcp", "zasada", "wymagan", "bpc-157", "ghk-cu", "test-fact")):
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
                if self.graph_client is not None and hasattr(self.graph_client, "add_relation"):
                    try:
                        self.graph_client.add_relation(f"turn:{subj}"[:60], "observed_fact", ln[:80])
                    except Exception:
                        pass

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
            if self.graph_client is not None and hasattr(self.graph_client, "add_relation"):
                try:
                    self.graph_client.add_relation(f"turn:{subj}"[:60], "observed_fact", first_line[:80])
                except Exception:
                    pass

        logger.info("[RPC] sync_turn session=%s -> extracted=%d facts", session_id, extracted_count)
        return {"synced": True, "extracted_facts": extracted_count, "session_id": session_id}

    async def _handle_what_if(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute what-if causal query."""
        action = params.get("action", "")
        entity = params.get("entity") or params.get("target") or action
        depth = int(params.get("depth", 2))

        if self.graph_client is not None and hasattr(self.graph_client, "what_if"):
            path = self.graph_client.what_if(action)
            return {"action": action, "causal_path": path, "causal_paths": [path] if isinstance(path, dict) else []}

        if self.causal_engine is not None:
            try:
                paths = await self.causal_engine.causal_what_if(entity, action, depth=depth)
                paths_data = [p.model_dump() if hasattr(p, "model_dump") else p for p in paths]
                logger.info("[RPC] what_if entity='%s' action='%s' -> %d paths", entity, action, len(paths_data))
                return {
                    "status": "ok",
                    "action": action,
                    "entity": entity,
                    "paths_count": len(paths_data),
                    "causal_paths": paths_data,
                    "causal_path": paths_data[0] if paths_data else None,
                }
            except Exception as exc:
                logger.debug("causal_what_if error: %s", exc)
                return {"status": "error", "error": str(exc), "causal_paths": [], "causal_path": None}

        return {"action": action, "causal_path": None, "causal_paths": [], "status": "no_causal_engine"}

    async def _handle_active_sensing(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Active sensing probe for predictive coding & prediction error calculation."""
        target_entity = params.get("target_entity") or params.get("probe", "")
        observed_predicate = params.get("observed_predicate", "state")
        observed_value = str(params.get("observed_value", "detected"))
        expected_value = str(params.get("expected_value", ""))

        if self.active_sensing is not None:
            from atlas_memory.active.prediction_error import PredictionCheck
            check = PredictionCheck(
                check_id=f"chk_{target_entity}",
                target_entity=target_entity,
                expected_predicate=observed_predicate,
                expected_value=expected_value or observed_value,
            )
            self.active_sensing.register_expectation(check)
            error = self.active_sensing.detect_discrepancy(
                observed_entity=target_entity,
                observed_predicate=observed_predicate,
                observed_value=observed_value,
            )
            logger.info("[RPC] active_sensing entity='%s' -> has_error=%s", target_entity, error is not None)
            return {
                "status": "ok",
                "probe": target_entity,
                "has_error": error is not None,
                "prediction_error": error.model_dump() if error else None,
            }
        return {"probe": target_entity, "sensed": True}

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
