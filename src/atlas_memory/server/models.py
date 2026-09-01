from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field

DEFAULT_SOCKET_PATH = Path.home() / ".hermes" / "atlas.sock"
DEFAULT_PID_PATH = Path.home() / ".hermes" / "atlas.pid"


class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request object."""

    jsonrpc: str = Field(default="2.0")
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[int, str]] = Field(default=1)


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response object."""

    jsonrpc: str = Field(default="2.0")
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None
    id: Optional[Union[int, str]] = None
