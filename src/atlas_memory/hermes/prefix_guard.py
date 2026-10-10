"""Public export of L1 Session Tracker & PrefixCacheGuard from
``atlas_memory.l1_working.session_tracker``.
"""
from __future__ import annotations

from atlas_memory.l1_working.session_tracker import (
    HermesSessionHook,
    PrefixCacheGuard,
)

try:
    from atlas_memory.l3_procedural.skill_compiler import compile_sop_to_skill
except ImportError:  # pragma: no cover
    compile_sop_to_skill = None  # type: ignore[assignment]


__all__ = [
    "compile_sop_to_skill",
    "PrefixCacheGuard",
    "HermesSessionHook",
]
