"""AtlasDaemonClient — Async Unix Domain Socket JSON-RPC 2.0 client for ATLAS."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import select
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from atlas_memory.ipc.wire_models import DEFAULT_SOCKET_PATH, JSONRPCRequest, JSONRPCResponse

logger = logging.getLogger(__name__)

_thread_local = threading.local()

DEFAULT_METHOD_TIMEOUTS: Dict[str, float] = {
    "trigger_sleep_consolidation": 30.0,
    "session/end": 30.0,
    "session_end": 30.0,
    "sync_mnemosyne": 30.0,
    "verify_audit_log": 20.0,
    "sync_memory_file": 20.0,
    "prefetch": 8.0,
}


from atlas_memory.core.exceptions import AtlasTransportError


class AtlasDaemonClient:
    """Async client communicating with AtlasDaemon over Unix Domain Socket with dynamic loop binding."""

    def __init__(
        self,
        socket_path: Path | str = DEFAULT_SOCKET_PATH,
        timeout: float = 2.0,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.timeout = timeout
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._request_counter: int = 0
        self._lock: Optional[asyncio.Lock] = None
        self._bound_loop: Optional[asyncio.AbstractEventLoop] = None

    def _ensure_lock(self) -> asyncio.Lock:
        """Dynamically binds asyncio.Lock and streams to the active running event loop."""
        cur_loop = asyncio.get_running_loop()
        if self._bound_loop is not cur_loop or self._lock is None:
            self._lock = asyncio.Lock()
            if self._bound_loop is not None and self._bound_loop is not cur_loop:
                self._reader = None
                self._writer = None
            self._bound_loop = cur_loop
        return self._lock

    @property
    def is_connected(self) -> bool:
        """Returns True if reader and writer exist, open, and bound to current loop."""
        try:
            cur_loop = asyncio.get_running_loop()
            if self._bound_loop is not None and self._bound_loop is not cur_loop:
                return False
        except RuntimeError:
            self._bound_loop = None
        return (
            self._reader is not None
            and self._writer is not None
            and not self._writer.is_closing()
        )

    async def connect(self, max_retries: int = 3, initial_delay: float = 0.01) -> bool:
        """Connects to the Unix Domain Socket with exponential backoff."""
        if self.is_connected:
            return True

        if not self.socket_path.exists():
            return False

        delay = initial_delay
        self._bound_loop = asyncio.get_running_loop()
        for attempt in range(max_retries):
            try:
                self._reader, self._writer = await asyncio.open_unix_connection(
                    path=str(self.socket_path)
                )
                return True
            except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
                logger.debug(
                    "Connect attempt %d/%d to %s failed: %s",
                    attempt + 1,
                    max_retries,
                    self.socket_path,
                    exc,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2

        return False

    async def close(self) -> None:
        """Closes the connection."""
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception as exc:
                logger.debug("Error closing client socket writer: %s", exc)
            finally:
                self._reader = None
                self._writer = None

    async def call(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Sends a JSON-RPC 2.0 request over UDS and waits for the response."""
        if timeout is not None:
            effective_timeout = timeout
        elif method in DEFAULT_METHOD_TIMEOUTS:
            effective_timeout = DEFAULT_METHOD_TIMEOUTS[method]
        else:
            effective_timeout = self.timeout

        async with self._ensure_lock():
            if not self.is_connected:
                connected = await self.connect()
                if not connected:
                    raise ConnectionError(f"Cannot connect to Atlas daemon at {self.socket_path}")

            reader = self._reader
            writer = self._writer
            if reader is None or writer is None:
                raise ConnectionError("Socket streams unavailable")

            self._request_counter += 1
            req_id = self._request_counter
            req = JSONRPCRequest(jsonrpc="2.0", method=method, params=params or {}, id=req_id)
            payload_bytes = req.model_dump_json().encode("utf-8")
            header = struct.pack(">I", len(payload_bytes))

            try:
                async def _send_and_recv() -> Any:
                    writer.write(header + payload_bytes)
                    await writer.drain()

                    length_bytes = await reader.readexactly(4)
                    (resp_len,) = struct.unpack(">I", length_bytes)
                    resp_bytes = await reader.readexactly(resp_len)
                    resp_dict = json.loads(resp_bytes.decode("utf-8"))
                    resp = JSONRPCResponse.model_validate(resp_dict)

                    if resp.error is not None:
                        # V124 (T4a): nagie RuntimeError gubiło JSONRPCError.code.
                        # AtlasTransportError(AtlasError(RuntimeError)) — instniejący
                        # `except RuntimeError` nadal łapie, a kod błędu jest zachowany.
                        raise AtlasTransportError(
                            reason="rpc_error",
                            detail=f"RPC error {resp.error.code}: {resp.error.message}",
                            code=int(resp.error.code),
                        )
                    return resp.result

                return await asyncio.wait_for(_send_and_recv(), timeout=effective_timeout)
            except Exception:
                await self.close()
                raise

    async def ping(self) -> bool:
        """Pings the daemon. Returns True if daemon responds status ok."""
        try:
            res = await self.call("ping", timeout=self.timeout)
            return isinstance(res, dict) and res.get("status") == "ok"
        except Exception:
            return False


class SyncUDSConnectionPool:
    """Thread-safe persistent connection pool for synchronous Unix Domain Socket communication."""

    def __init__(self, max_idle: int = 8, max_lifetime: float = 300.0) -> None:
        self.max_idle = max_idle
        self.max_lifetime = max_lifetime
        self._pools: Dict[str, list[tuple[socket.socket, float]]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _is_alive(sock: socket.socket) -> bool:
        """Check if idle socket is still alive and has not been closed by peer."""
        try:
            r, _, x = select.select([sock], [], [sock], 0)
            if r or x:
                return False
            return True
        except Exception:
            return False

    def acquire(self, socket_path: str, timeout: float) -> tuple[socket.socket, bool]:
        """Acquires a healthy connection from the pool or creates a new one.

        Returns (socket, from_pool).
        """
        now = time.monotonic()
        with self._lock:
            pool = self._pools.get(socket_path, [])
            while pool:
                sock, created_at = pool.pop()
                if now - created_at < self.max_lifetime and self._is_alive(sock):
                    try:
                        sock.settimeout(timeout)
                        return sock, True
                    except Exception as err:
                        logger.debug("Failed setting timeout on pooled socket: %s", err)
                try:
                    sock.close()
                except Exception as err:
                    logger.debug("Failed closing dead socket: %s", err)

        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(socket_path)
        return s, False

    def release(self, socket_path: str, sock: socket.socket) -> None:
        """Returns a healthy socket to the pool for reuse."""
        if not self._is_alive(sock):
            self.discard(socket_path, sock)
            return
        with self._lock:
            pool = self._pools.setdefault(socket_path, [])
            if len(pool) < self.max_idle:
                pool.append((sock, time.monotonic()))
                return
        try:
            sock.close()
        except Exception as err:
            logger.debug("Failed closing socket on pool overflow: %s", err)

    def discard(self, socket_path: str, sock: socket.socket) -> None:
        """Closes and discards an unhealthy socket."""
        try:
            sock.close()
        except Exception as err:
            logger.debug("Failed closing discarded socket: %s", err)

    def close_all(self) -> None:
        """Closes all pooled sockets across all paths."""
        with self._lock:
            for pool in self._pools.values():
                for sock, _ in pool:
                    try:
                        sock.close()
                    except Exception as err:
                        logger.debug("Failed closing socket in close_all: %s", err)
            self._pools.clear()


_GLOBAL_SYNC_POOL = SyncUDSConnectionPool()


def send_uds_request_sync(
    socket_path: Path | str = DEFAULT_SOCKET_PATH,
    method: str = "ping",
    params: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
) -> Optional[Any]:
    """Sends a synchronous JSON-RPC 2.0 request over UDS using persistent connection pool."""
    sock_p = str(socket_path)
    if not os.path.exists(sock_p):
        raise AtlasTransportError(
            reason="socket_missing",
            detail=f"UDS socket file does not exist: {sock_p}",
        )

    if timeout is not None:
        effective_timeout = timeout
    else:
        effective_timeout = DEFAULT_METHOD_TIMEOUTS.get(method, 5.0)

    req = {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": 1}
    payload_bytes = json.dumps(req).encode("utf-8")
    header = struct.pack(">I", len(payload_bytes))

    def _execute_over_socket(s: socket.socket) -> tuple[Optional[Any], bool]:
        s.sendall(header + payload_bytes)

        len_raw = s.recv(4)
        if len(len_raw) < 4:
            raise AtlasTransportError(
                reason="short_read",
                detail=f"Incomplete header: expected 4 bytes, received {len(len_raw)}",
            )
        (resp_len,) = struct.unpack(">I", len_raw)

        chunks = []
        bytes_received = 0
        while bytes_received < resp_len:
            chunk = s.recv(min(4096, resp_len - bytes_received))
            if not chunk:
                break
            chunks.append(chunk)
            bytes_received += len(chunk)

        if bytes_received < resp_len:
            raise AtlasTransportError(
                reason="short_read",
                detail=f"Incomplete payload: expected {resp_len} bytes, received {bytes_received}",
            )

        resp_data = b"".join(chunks)
        resp_dict = json.loads(resp_data.decode("utf-8"))
        if "error" in resp_dict and resp_dict["error"]:
            # V124 (T4a): zachowaj numeryczny kod błędu JSON-RPC także w ścieżce sync.
            try:
                rpc_code = int(resp_dict["error"].get("code"))
            except (TypeError, ValueError, AttributeError):
                rpc_code = None
            raise AtlasTransportError(
                reason="rpc_error",
                detail=str(resp_dict["error"]),
                code=rpc_code,
            )
        return resp_dict.get("result"), True

    s: Optional[socket.socket] = None
    from_pool = False
    reusable = False

    try:
        try:
            s, from_pool = _GLOBAL_SYNC_POOL.acquire(sock_p, effective_timeout)
            result, reusable = _execute_over_socket(s)
            return result
        except (BrokenPipeError, ConnectionResetError, OSError, AtlasTransportError) as retry_exc:
            if from_pool and s is not None and (
                not isinstance(retry_exc, AtlasTransportError) or retry_exc.reason == "short_read"
            ):
                # Connection was stale in pool; discard and retry once with fresh socket
                _GLOBAL_SYNC_POOL.discard(sock_p, s)
                s, from_pool = _GLOBAL_SYNC_POOL.acquire(sock_p, effective_timeout)
                result, reusable = _execute_over_socket(s)
                return result
            raise retry_exc
    except AtlasTransportError:
        raise
    except Exception as exc:
        logger.debug("send_uds_request_sync error: %s", exc)
        raise AtlasTransportError(
            reason="io_error",
            detail=str(exc),
        ) from exc
    finally:
        if s is not None:
            if reusable:
                _GLOBAL_SYNC_POOL.release(sock_p, s)
            else:
                _GLOBAL_SYNC_POOL.discard(sock_p, s)


def get_or_create_client(
    socket_path: Path | str = DEFAULT_SOCKET_PATH,
    timeout: float = 0.1,
) -> AtlasDaemonClient:
    """Thread-safe getter for thread-local AtlasDaemonClient."""
    key = f"atlas_client_{str(socket_path)}"
    clients = getattr(_thread_local, "clients", None)
    if clients is None:
        clients = {}
        _thread_local.clients = clients

    if key not in clients:
        clients[key] = AtlasDaemonClient(socket_path=socket_path, timeout=timeout)

    return clients[key]


def _resolve_rpc_timeout(method: str, timeout: float | None = None) -> float:
    if timeout is not None:
        return float(timeout)
    return float(DEFAULT_METHOD_TIMEOUTS.get(method, 2.0))

# V124 (T4b): usunięty dekoracyjny cykliczny import `server.uds_transport.__doc__`
# (ciągnął ipc -> server -> ipc bez żadnego konsumenckiego zastosowania).
