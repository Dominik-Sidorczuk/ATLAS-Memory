"""AtlasDaemonClient — Async Unix Domain Socket JSON-RPC 2.0 client for ATLAS."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from atlas_memory.server.models import DEFAULT_SOCKET_PATH, JSONRPCRequest, JSONRPCResponse

logger = logging.getLogger(__name__)

_thread_local = threading.local()


class AtlasDaemonClient:
    """Async client communicating with AtlasDaemon over Unix Domain Socket."""

    def __init__(
        self,
        socket_path: Path | str = DEFAULT_SOCKET_PATH,
        timeout: float = 0.1,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.timeout = timeout
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._request_counter: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        """Returns True if reader and writer exist and socket is open."""
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
        effective_timeout = timeout if timeout is not None else self.timeout

        async with self._lock:
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
                        raise RuntimeError(f"RPC error {resp.error.code}: {resp.error.message}")
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


def send_uds_request_sync(
    socket_path: Path | str = DEFAULT_SOCKET_PATH,
    method: str = "ping",
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0,
) -> Optional[Any]:
    """Sends a synchronous JSON-RPC 2.0 request over UDS using blocking standard socket."""
    sock_p = str(socket_path)
    if not os.path.exists(sock_p):
        return None

    import socket

    req = {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": 1}
    payload_bytes = json.dumps(req).encode("utf-8")
    header = struct.pack(">I", len(payload_bytes))

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(sock_p)
        s.sendall(header + payload_bytes)

        len_raw = s.recv(4)
        if len(len_raw) < 4:
            return None
        (resp_len,) = struct.unpack(">I", len_raw)

        chunks = []
        bytes_received = 0
        while bytes_received < resp_len:
            chunk = s.recv(min(4096, resp_len - bytes_received))
            if not chunk:
                break
            chunks.append(chunk)
            bytes_received += len(chunk)

        resp_data = b"".join(chunks)
        resp_dict = json.loads(resp_data.decode("utf-8"))
        if "error" in resp_dict and resp_dict["error"]:
            return None
        return resp_dict.get("result")
    except Exception as exc:
        logger.debug("send_uds_request_sync error: %s", exc)
        return None
    finally:
        s.close()


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
