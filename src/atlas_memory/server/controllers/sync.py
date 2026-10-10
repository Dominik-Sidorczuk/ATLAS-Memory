"""Multi-Agent CRDT Synchronization & Gossip RPC Controller for AtlasDaemon.

Encapsulates CRDT delta export/apply, peer discovery, and vector clock status.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict

logger = logging.getLogger("atlas.server.controllers.sync")


class SyncRpcControllerMixin:
    """Mixin providing multi-agent CRDT synchronization and gossip RPC handlers."""

    async def _handle_sync_export_delta(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Export CRDT delta since vector clock."""
        gossip = getattr(self, "gossip", None)
        if gossip is None:
            return {"status": "not_configured"}
        target_peer = params.get("target_peer_id", "hermes_peer_alpha")
        since_seq = params.get("since_seq")
        try:
            req_bytes = gossip.create_sync_request(target_peer, since_seq=since_seq)
            return {"status": "ok", "sync_request": req_bytes.decode("utf-8")}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    async def _handle_sync_apply_delta(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply incoming delta to local CRDT."""
        gossip = getattr(self, "gossip", None)
        if gossip is None:
            return {"status": "not_configured"}
        sync_payload = params.get("sync_request", "")
        if not sync_payload:
            return {"status": "error", "error": "missing sync_request"}
        try:
            req_bytes = sync_payload.encode("utf-8") if isinstance(sync_payload, str) else sync_payload
            delta = await gossip.process_sync_response_async(req_bytes)
            return {"status": "ok", "source_node": delta.source_node, "clock": delta.vector_clock.clocks}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    async def _handle_sync_peer_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return known peers and vector clock statuses with JSON-safe payload."""
        gossip = getattr(self, "gossip", None)
        if gossip is None:
            return {"status": "not_configured"}

        peers_dict: Dict[str, Any] = {}
        for peer_id, p in gossip.peers.items():
            p_dict: Dict[str, Any] = {
                "peer_id": str(getattr(p, "peer_id", peer_id)),
            }
            vc = getattr(p, "vector_clock", None)
            if vc is not None:
                if hasattr(vc, "clocks") and isinstance(vc.clocks, dict):
                    p_dict["vector_clock"] = {
                        str(k): int(v) for k, v in vc.clocks.items() if isinstance(v, (int, float))
                    }
                elif isinstance(vc, dict):
                    raw_vc = vc.get("clocks", vc) if isinstance(vc.get("clocks"), dict) else vc
                    p_dict["vector_clock"] = {
                        str(k): int(v)
                        for k, v in raw_vc.items()
                        if not str(k).startswith("_") and isinstance(v, (int, float))
                    }
                else:
                    p_dict["vector_clock"] = {}
            else:
                p_dict["vector_clock"] = {}

            for field_name in ("last_seen", "rtt_ms", "status", "seq"):
                val = getattr(p, field_name, None)
                if val is None and isinstance(p, dict):
                    val = p.get(field_name)
                if val is not None and not callable(val) and not isinstance(val, (threading.Lock, type(threading.Lock()))):
                    p_dict[field_name] = val

            peers_dict[str(peer_id)] = p_dict

        local_clock: Dict[str, int] = {}
        raw_clock = None
        if gossip.bridge is not None and hasattr(gossip.bridge, "clock"):
            raw_clock = getattr(gossip.bridge.clock, "clocks", gossip.bridge.clock)
        elif hasattr(gossip, "crdt") and hasattr(gossip.crdt, "clock"):
            raw_clock = getattr(gossip.crdt.clock, "clocks", gossip.crdt.clock)

        if isinstance(raw_clock, dict):
            local_clock = {str(k): int(v) for k, v in raw_clock.items() if isinstance(v, (int, float))}

        return {"status": "ok", "peers": peers_dict, "local_clock": local_clock}
