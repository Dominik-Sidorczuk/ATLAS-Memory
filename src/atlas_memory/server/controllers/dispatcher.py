"""JSON-RPC 2.0 Dispatcher for AtlasDaemon.

Encapsulates method routing, request validation, timing telemetry,
error wrapping, and execution logging according to JSON-RPC 2.0 specification.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Coroutine, Dict, Optional

from atlas_memory.core.agi_logger import log_rpc
from atlas_memory.ipc.wire_models import (
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
)

logger = logging.getLogger("atlas.server.dispatcher")

RpcHandler = Callable[[Dict[str, Any]], Coroutine[Any, Any, Any]]


class JsonRpcDispatcher:
    """Dispatches and executes JSON-RPC 2.0 requests with structured logging and error handling."""

    def __init__(self, handlers: Optional[Dict[str, RpcHandler]] = None) -> None:
        self.handlers: Dict[str, RpcHandler] = dict(handlers or {})

    def register_handler(self, method: str, handler: RpcHandler) -> None:
        """Registers a handler for a given JSON-RPC method."""
        self.handlers[method] = handler

    def register_handlers(self, handlers: Dict[str, RpcHandler]) -> None:
        """Bulk registers handlers."""
        self.handlers.update(handlers)

    async def dispatch(self, req_raw: Any) -> Optional[JSONRPCResponse]:
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

        handler = self.handlers.get(req.method)
        if handler is None:
            if is_notification:
                return None
            return JSONRPCResponse(
                id=req.id,
                error=JSONRPCError.method_not_found(req.method),
            )

        try:
            t0 = time.perf_counter()
            params = req.params or {}
            result = await handler(params)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if not is_notification:
                sig: Dict[str, Any] = {}
                if isinstance(result, dict):
                    sig = {
                        k: v for k, v in result.items() if k in (
                            "status", "found", "count", "total", "ingested",
                            "consolidated_records", "baked_sops_count", "valid",
                            "iterations", "updated_edges", "converged",
                            "probe", "sensed", "deleted", "key", "reason",
                        ) and not isinstance(v, (dict, list))
                    }
                    if "baked_sops_count" in result:
                        sig["baked_sops_count"] = result.get("baked_sops_count", 0)
                    if isinstance(result.get("baked_sops"), list):
                        sig["sops"] = len(result["baked_sops"])
                log_rpc(logger, req.method, dt_ms, is_ok=True, sig=sig)
                return JSONRPCResponse(id=req.id, result=result)
            return None
        except (ValueError, TypeError, KeyError) as param_err:
            dt_ms = (time.perf_counter() - t0) * 1000.0 if "t0" in locals() else 0.0
            log_rpc(logger, req.method, dt_ms, is_ok=False, sig={"error": str(param_err)})
            logger.warning("Invalid params in RPC method %s: %s", req.method, param_err)
            if is_notification:
                return None
            return JSONRPCResponse(
                id=req.id,
                error=JSONRPCError.invalid_params(str(param_err)),
            )
        except Exception as exc:
            dt_ms = (time.perf_counter() - t0) * 1000.0 if "t0" in locals() else 0.0
            log_rpc(logger, req.method, dt_ms, is_ok=False, sig={"error": str(exc)})
            logger.exception("Error executing RPC method %s: %s", req.method, exc)
            if is_notification:
                return None
            return JSONRPCResponse(
                id=req.id,
                error=JSONRPCError.internal_error(str(exc)),
            )
