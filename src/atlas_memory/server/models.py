from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field

DEFAULT_SOCKET_PATH = Path.home() / ".hermes" / "atlas.sock"
DEFAULT_PID_PATH = Path.home() / ".hermes" / "atlas.pid"

# Standard JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Application-specific error codes (-32000 to -32099)
SERVER_NOT_INITIALIZED = -32000
ENGINE_UNAVAILABLE = -32001
STORAGE_ERROR = -32002
SESSION_ERROR = -32003


class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Optional[Any] = None

    @classmethod
    def invalid_request(cls, message: str, data: Optional[Any] = None) -> JSONRPCError:
        return cls(code=INVALID_REQUEST, message=f"Invalid Request: {message}", data=data)

    @classmethod
    def method_not_found(cls, method: str) -> JSONRPCError:
        return cls(code=METHOD_NOT_FOUND, message=f"Method not found: {method}")

    @classmethod
    def invalid_params(cls, message: str, data: Optional[Any] = None) -> JSONRPCError:
        return cls(code=INVALID_PARAMS, message=f"Invalid params: {message}", data=data)

    @classmethod
    def internal_error(cls, message: str, data: Optional[Any] = None) -> JSONRPCError:
        return cls(code=INTERNAL_ERROR, message=f"Internal error: {message}", data=data)

    @classmethod
    def engine_unavailable(cls, message: str = "Engine not available") -> JSONRPCError:
        return cls(code=ENGINE_UNAVAILABLE, message=message)


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request object."""

    jsonrpc: str = Field(default="2.0")
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[int, str]] = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response object."""

    jsonrpc: str = Field(default="2.0")
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None
    id: Optional[Union[int, str]] = None
