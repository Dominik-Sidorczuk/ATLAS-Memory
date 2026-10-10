"""Canonical Layer-0 Domain & Transport Exceptions for ATLAS."""
from __future__ import annotations

from typing import Optional


class AtlasError(RuntimeError):
    """Base exception for all ATLAS memory subsystem errors."""


class AtlasTransportError(AtlasError):
    """Domain exception for UDS transport errors with explicit reason classification."""

    def __init__(self, reason: str, detail: str = "", code: Optional[int] = None) -> None:
        msg = f"AtlasTransportError({reason}): {detail}"
        if code is not None:
            msg += f" [code={int(code)}]"
        super().__init__(msg)
        self.reason = reason
        self.detail = detail
        self.code = int(code) if code is not None else None


class SecurityValidationError(AtlasError, ValueError):
    """Raised when input parameters violate security length, null-byte, or traversal guards."""


class InvariantViolation(AtlasError):
    """V113 (F-029-class): production invariant violated — replaces bare `assert`.

    Unlike `assert`, this survives `python -O` / `PYTHONOPTIMIZE=2` because it is a
    real conditional raise, not an optimized-out assertion. Used at every hot-path
    site where a silent drop would corrupt state (Kuzu ADD-ONLY migration, KV
    connection liveness, PPR igraph backend selection).

    Fields are intentionally structured (`module`, `name`, `observed`, `expected`,
    `hint`) so audit/ledger tooling can index them without parsing free-form text.
    """

    def __init__(
        self,
        module: str,
        name: str,
        observed: int = -1,
        expected: int = -1,
        hint: str = "",
    ) -> None:
        msg = f"InvariantViolation[{module}:{name}] observed={observed} expected={expected}"
        if hint:
            msg += f" — {hint}"
        super().__init__(msg)
        self.module = module
        self.name = name
        self.observed = int(observed)
        self.expected = int(expected)
        self.hint = hint


__all__ = [
    "AtlasError",
    "AtlasTransportError",
    "InvariantViolation",
    "SecurityValidationError",
]
