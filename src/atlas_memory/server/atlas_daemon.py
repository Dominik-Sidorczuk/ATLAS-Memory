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

from atlas_memory.server.models import JSONRPCError, JSONRPCRequest, JSONRPCResponse

logger = logging.getLogger(__name__)

DEFAULT_SOCKET_PATH = Path.home() / ".hermes" / "atlas.sock"
DEFAULT_PID_PATH = Path.home() / ".hermes" / "atlas.pid"


class AtlasDaemon:
    """Atlas Micro-Sidecar Daemon executing JSON-RPC 2.0 requests via Unix Domain Socket."""

    def __init__(
        self,
        socket_path: Path | str = DEFAULT_SOCKET_PATH,
        pid_path: Path | str = DEFAULT_PID_PATH,
        graph_client: Optional[Any] = None,
        latent_buffer: Optional[Any] = None,
        orchestrator: Optional[Any] = None,
        *,
        gossip: Optional[Any] = None,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.pid_path = Path(pid_path)
        self.graph_client = graph_client
        self.latent_buffer = latent_buffer
        self.orchestrator = orchestrator
        self.gossip = gossip

        self._server: Optional[asyncio.Server] = None
        self._serving: bool = False
        self._active_writers: Set[asyncio.StreamWriter] = set()
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Coroutine[Any, Any, Any]]] = {
            "ping": self._handle_ping,
            "prefetch": self._handle_prefetch,
            "sync_turn": self._handle_sync_turn,
            "what_if": self._handle_what_if,
            "active_sensing": self._handle_active_sensing,
            "telemetry_report": self._handle_telemetry_report,
            "sync_export_delta": self._handle_sync_export_delta,
            "sync_apply_delta": self._handle_sync_apply_delta,
            "sync_peer_status": self._handle_sync_peer_status,
        }

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

    async def _handle_prefetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Prefetch memory records for query/session."""
        query = params.get("query", "")
        session_id = params.get("session_id", "")
        if self.orchestrator is not None:
            if hasattr(self.orchestrator, "orchestrated_recall"):
                records = await self.orchestrator.orchestrated_recall(query, session_id=session_id)
                return {
                    "records": [r.model_dump() if hasattr(r, "model_dump") else r for r in records],
                    "query": query,
                }
            if hasattr(self.orchestrator, "prefetch"):
                res = self.orchestrator.prefetch(query, session_id=session_id)
                return {"result": res, "query": query}
        return {"records": [], "query": query, "status": "no_orchestrator"}

    async def _handle_sync_turn(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sync conversation turn."""
        user_content = params.get("user_content", "")
        assistant_content = params.get("assistant_content", "")
        session_id = params.get("session_id", "")

        extracted_count = 0
        if self.orchestrator is not None and hasattr(self.orchestrator, "_fallback_extract_facts"):
            facts = self.orchestrator._fallback_extract_facts(user_content, assistant_content)
            extracted_count = len(facts) if facts else 0

        return {"synced": True, "extracted_facts": extracted_count, "session_id": session_id}

    async def _handle_what_if(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute what-if causal query."""
        action = params.get("action", "")
        if self.graph_client is not None and hasattr(self.graph_client, "what_if"):
            path = self.graph_client.what_if(action)
            return {"action": action, "causal_path": path}
        return {"action": action, "causal_path": None, "status": "no_graph_client"}

    async def _handle_active_sensing(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Active sensing probe."""
        probe = params.get("probe", "")
        return {"probe": probe, "sensed": True}

    async def _handle_telemetry_report(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return server telemetry report."""
        return {
            "pid": os.getpid(),
            "socket": str(self.socket_path),
            "serving": self._serving,
            "has_graph_client": self.graph_client is not None,
            "has_latent_buffer": self.latent_buffer is not None,
            "has_orchestrator": self.orchestrator is not None,
            "has_gossip": self.gossip is not None,
        }

    async def _handle_sync_export_delta(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Export CRDT delta since vector clock."""
        if self.gossip is None:
            return {"status": "not_configured"}
        target_peer = params.get("target_peer_id", "")
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
