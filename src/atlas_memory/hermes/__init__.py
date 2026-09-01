"""
Hermes Integration Subpackage for ATLAS Memory Provider and Session Hooks.
"""

from atlas_memory.hermes.prefix_guard import HermesSessionHook, PrefixCacheGuard
from atlas_memory.hermes.tools import (
    COMMIT_OBSERVATION_SCHEMA,
    SEARCH_MEMORY_SCHEMA,
    create_hermes_tool_handlers,
    register_hermes_memory_tools,
)

__all__ = [
    "PrefixCacheGuard",
    "HermesSessionHook",
    "SEARCH_MEMORY_SCHEMA",
    "COMMIT_OBSERVATION_SCHEMA",
    "create_hermes_tool_handlers",
    "register_hermes_memory_tools",
]
