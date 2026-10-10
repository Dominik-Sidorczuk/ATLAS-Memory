"""ATLAS Server package — Micro-Sidecar IPC over Unix Domain Socket."""
from __future__ import annotations

from atlas_memory.ipc.uds_client import (
    AtlasDaemonClient,
    get_or_create_client,
    send_uds_request_sync,
)
from atlas_memory.ipc.wire_models import JSONRPCError, JSONRPCRequest, JSONRPCResponse
from atlas_memory.server.atlas_daemon import AtlasDaemon

__all__ = [
    "send_uds_request_sync",
    "AtlasDaemon",
    "AtlasDaemonClient",
    "JSONRPCError",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "get_or_create_client",
]
