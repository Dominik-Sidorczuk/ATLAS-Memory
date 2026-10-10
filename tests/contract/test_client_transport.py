"""
Acceptance tests for UDS client transport error handling (V80 / T1 / F-110).

Verifies that send_uds_request_sync never returns silent None on errors:
(a) socket_missing: non-existent socket path
(b) rpc_error: daemon returns JSON-RPC response with 'error'
(c) short_read: server sends incomplete header (<4 bytes) or truncated payload
(d) io_error: connect raises OSError / ConnectionRefusedError
(+) valid null result: daemon returns result: null without error -> returns None
"""

from __future__ import annotations

import json
import socket
import struct
import threading
from pathlib import Path

import pytest

from atlas_memory.ipc.uds_client import AtlasTransportError, send_uds_request_sync


def test_transport_nonexistent_socket_raises_socket_missing(uds_dir: Path):
    """(a) Missing socket file must raise AtlasTransportError with reason='socket_missing'."""
    missing_sock = uds_dir / "nonexistent.sock"
    assert not missing_sock.exists()

    with pytest.raises(AtlasTransportError) as exc_info:
        send_uds_request_sync(socket_path=missing_sock, method="ping")

    err = exc_info.value
    assert isinstance(err, RuntimeError)
    assert err.reason == "socket_missing"
    assert str(missing_sock) in err.detail


def test_transport_rpc_error_raises_rpc_error(uds_dir: Path):
    """(b) JSON-RPC error response must raise AtlasTransportError with reason='rpc_error'."""
    sock_path = uds_dir / "test_rpc_err.sock"

    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(str(sock_path))
    server_sock.listen(1)

    def _serve_rpc_error():
        conn, _ = server_sock.accept()
        try:
            # Read request
            len_bytes = conn.recv(4)
            if len(len_bytes) == 4:
                (req_len,) = struct.unpack(">I", len_bytes)
                _ = conn.recv(req_len)

            # Reply with JSON-RPC error
            err_payload = json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": "Method not found"},
                "id": 1,
            }).encode("utf-8")
            conn.sendall(struct.pack(">I", len(err_payload)) + err_payload)
        finally:
            conn.close()

    t = threading.Thread(target=_serve_rpc_error, daemon=True)
    t.start()

    try:
        with pytest.raises(AtlasTransportError) as exc_info:
            send_uds_request_sync(socket_path=sock_path, method="invalid_method", timeout=2.0)

        err = exc_info.value
        assert err.reason == "rpc_error"
        assert "Method not found" in err.detail
    finally:
        server_sock.close()
        t.join(timeout=1.0)


def test_transport_short_read_header_raises_short_read(uds_dir: Path):
    """(c1) Truncated header (< 4 bytes) must raise AtlasTransportError with reason='short_read'."""
    sock_path = uds_dir / "test_short_hdr.sock"

    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(str(sock_path))
    server_sock.listen(1)

    def _serve_short_hdr():
        conn, _ = server_sock.accept()
        try:
            _ = conn.recv(1024)
            # Send only 2 bytes instead of 4
            conn.sendall(b"\x00\x01")
        finally:
            conn.close()

    t = threading.Thread(target=_serve_short_hdr, daemon=True)
    t.start()

    try:
        with pytest.raises(AtlasTransportError) as exc_info:
            send_uds_request_sync(socket_path=sock_path, method="ping", timeout=2.0)

        err = exc_info.value
        assert err.reason == "short_read"
        assert "header" in err.detail.lower()
    finally:
        server_sock.close()
        t.join(timeout=1.0)


def test_transport_short_read_payload_raises_short_read(uds_dir: Path):
    """(c2) Truncated payload (< declared resp_len) must raise AtlasTransportError with reason='short_read'."""
    sock_path = uds_dir / "test_short_payload.sock"

    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(str(sock_path))
    server_sock.listen(1)

    def _serve_short_payload():
        conn, _ = server_sock.accept()
        try:
            _ = conn.recv(1024)
            # Declares length of 100 bytes, but sends only 10 bytes then closes
            conn.sendall(struct.pack(">I", 100) + b"short_data")
        finally:
            conn.close()

    t = threading.Thread(target=_serve_short_payload, daemon=True)
    t.start()

    try:
        with pytest.raises(AtlasTransportError) as exc_info:
            send_uds_request_sync(socket_path=sock_path, method="ping", timeout=2.0)

        err = exc_info.value
        assert err.reason == "short_read"
        assert "payload" in err.detail.lower()
    finally:
        server_sock.close()
        t.join(timeout=1.0)


def test_transport_connect_os_error_raises_io_error(uds_dir: Path):
    """(d) OSError on connect (e.g. stale socket file refusing connection) must raise reason='io_error'."""
    sock_path = uds_dir / "stale_refusing.sock"
    # Create a socket file on disk, but don't listen on it (or close it)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(sock_path))
    s.close()
    assert sock_path.exists()

    with pytest.raises(AtlasTransportError) as exc_info:
        send_uds_request_sync(socket_path=sock_path, method="ping", timeout=1.0)

    err = exc_info.value
    assert err.reason == "io_error"
    assert len(err.detail) > 0


def test_transport_valid_null_result_returns_none(uds_dir: Path):
    """Legitimate JSON-RPC result: null must return None without raising an exception."""
    sock_path = uds_dir / "test_null_res.sock"

    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(str(sock_path))
    server_sock.listen(1)

    def _serve_null_result():
        conn, _ = server_sock.accept()
        try:
            _ = conn.recv(1024)
            payload = json.dumps({"jsonrpc": "2.0", "result": None, "id": 1}).encode("utf-8")
            conn.sendall(struct.pack(">I", len(payload)) + payload)
        finally:
            conn.close()

    t = threading.Thread(target=_serve_null_result, daemon=True)
    t.start()

    try:
        res = send_uds_request_sync(socket_path=sock_path, method="get", params={"key": "nonexistent"}, timeout=2.0)
        assert res is None
    finally:
        server_sock.close()
        t.join(timeout=1.0)

