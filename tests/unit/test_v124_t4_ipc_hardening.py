"""V124 (T4 / H1) — IPC UDS client hardening + protocol conformance + cyclic cleanup.

Audit findings on src/atlas_memory/ipc/uds_client.py:
 (a) async ``AtlasDaemonClient.call`` raised a bare ``RuntimeError`` on a
     JSON-RPC error response, losing ``JSONRPCError.code``; the sync
     ``send_uds_request_sync`` raised ``AtlasTransportError`` but also
     dropped the numeric code. Fix: ``AtlasTransportError`` grows an
     optional ``code`` attribute (no breaking change — positional
     ``reason``/``detail`` stay as-is, and AtlasError subclasses
     RuntimeError so every legacy ``except RuntimeError`` keeps working).
 (b) the module ended with a decoration-only cyclic import
     ``from atlas_memory.server.uds_transport import __doc__ as
     _uds_transport_compat_shim`` — removed; ipc must not depend on server.
 (c) ``UDSClientProtocol`` in core/protocols.py was purely decorative —
     this file adds the runtime_checkable conformance assertion binding
     ``AtlasDaemonClient`` to it.

Doctrine: only in-process fake servers bound to uds_dir/tmp_path sockets;
zero contact with the live daemon / ~/.hermes/atlas/*.
"""
from __future__ import annotations

import inspect
import json
import socket
import struct
import threading
from pathlib import Path

import pytest

from atlas_memory.core.exceptions import AtlasError, AtlasTransportError
from atlas_memory.core.protocols import UDSClientProtocol
from atlas_memory.ipc.uds_client import AtlasDaemonClient, send_uds_request_sync


# ------------------------------------------------------------------ (a) async path
def _serve_jsonrpc_error(server_sock: socket.socket, code: int, message: str) -> None:
    conn, _ = server_sock.accept()
    try:
        len_bytes = conn.recv(4)
        if len(len_bytes) == 4:
            (req_len,) = struct.unpack(">I", len_bytes)
            conn.recv(req_len)
        err_payload = json.dumps({
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": 1,
        }).encode("utf-8")
        conn.sendall(struct.pack(">I", len(err_payload)) + err_payload)
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_async_client_rpc_error_keeps_code(uds_dir: Path) -> None:
    """JSON-RPC error must surface as AtlasTransportError carrying .code."""
    sock_path = uds_dir / "v124t4_async_err.sock"
    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(str(sock_path))
    server_sock.listen(1)
    t = threading.Thread(target=_serve_jsonrpc_error, args=(server_sock, -32603, "boom-t4"), daemon=True)
    t.start()
    try:
        client = AtlasDaemonClient(socket_path=sock_path, timeout=2.0)
        with pytest.raises(AtlasTransportError) as exc_info:
            await client.call("some_method", {})
        err = exc_info.value
        assert err.code == -32603
        assert "boom-t4" in err.detail
        # Backwards compatibility: legacy `except RuntimeError` still catches it.
        assert isinstance(err, RuntimeError)
        assert isinstance(err, AtlasError)
        await client.close()
    finally:
        server_sock.close()
        t.join(timeout=2.0)


def test_sync_client_rpc_error_keeps_code(uds_dir: Path) -> None:
    """Sync transport rpc_error must also carry the numeric code (same contract)."""
    sock_path = uds_dir / "v124t4_sync_err.sock"
    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(str(sock_path))
    server_sock.listen(1)
    t = threading.Thread(target=_serve_jsonrpc_error, args=(server_sock, -32601, "no-method-t4"), daemon=True)
    t.start()
    try:
        with pytest.raises(AtlasTransportError) as exc_info:
            send_uds_request_sync(socket_path=sock_path, method="gone", timeout=2.0)
        err = exc_info.value
        assert err.reason == "rpc_error"
        assert err.code == -32601
        assert "no-method-t4" in err.detail
    finally:
        server_sock.close()
        t.join(timeout=2.0)


# ------------------------------------------------------------------ (b) cyclic import removed
def test_ipc_module_no_longer_imports_server_uds_transport() -> None:
    """The shim import at EOF of ipc/uds_client.py was cyclic decoration — it must be gone."""
    from atlas_memory.ipc import uds_client as uds_client_mod

    src = Path(inspect.getsourcefile(uds_client_mod) or "").read_text(encoding="utf-8")
    assert "from atlas_memory.server.uds_transport" not in src, (
        "ipc.uds_client must not import server.uds_transport (cyclic shim, V124/T4b)"
    )


# ------------------------------------------------------------------ (c) protocol conformance
def test_uds_client_protocol_runtime_checkable_conformance() -> None:
    """UDSClientProtocol must be enforced: AtlasDaemonClient satisfies it structurally."""
    client = AtlasDaemonClient(socket_path="/tmp/v124t4_unused.sock", timeout=0.5)
    assert isinstance(client, UDSClientProtocol)
    assert inspect.iscoroutinefunction(client.call)
    assert inspect.iscoroutinefunction(client.close)
