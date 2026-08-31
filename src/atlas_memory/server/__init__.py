"""ATLAS Server package — Micro-Sidecar IPC over Unix Domain Socket."""

from __future__ import annotations

from atlas_memory.server.atlas_daemon import AtlasDaemon
from atlas_memory.server.client import AtlasDaemonClient, get_or_create_client
from atlas_memory.server.models import JSONRPCError, JSONRPCRequest, JSONRPCResponse

__all__ = [
    "AtlasDaemon",
    "AtlasDaemonClient",
    "JSONRPCError",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "get_or_create_client",
]
