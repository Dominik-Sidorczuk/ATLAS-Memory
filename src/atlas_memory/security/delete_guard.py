"""Security guards for destructive operations in ATLAS (ADR-005, ADR-006)."""
from __future__ import annotations

from typing import Tuple

# Canonical single source of truth for deletion-protected key prefixes (ADR-005, A3).
# V88b/T2: 'rule:', 'preference:', 'constraint:', 'config:' dolacza sie do listy —
# HotBuffer.PERSISTENT_KEY_PREFIXES traktuje je jako trwale/globalne, wiec kasowanie
# ich RPC delete bylo asymetryczne (pomiar na PID 107795: delete 'rule:preferred_editor'
# -> deleted=True). Klucze probne ('probe:') i sesyjne ('session_tmp:') celu NIE sa —
# musza umiec znikac (patrz tests/unit/test_v88b_wave2.py).
DELETE_PROTECTED_PREFIXES: Tuple[str, ...] = (
    "hermes:",
    "fact:",
    "prompt-",
    "mnemosyne:",
    "rule:",
    "preference:",
    "constraint:",
    "config:",
)


def is_delete_protected(key: str) -> Tuple[bool, str]:
    """Check whether a key matches any deletion-protected prefix.

    Args:
        key: The state variable or fact key to check.

    Returns:
        Tuple of (is_protected: bool, matched_prefix: str).
    """
    for prefix in DELETE_PROTECTED_PREFIXES:
        if key.startswith(prefix):
            return True, prefix
    return False, ""
