"""Controllers package for AtlasDaemon.

Decomposes the monolithic AtlasDaemon into dedicated single-responsibility controllers:
- JsonRpcDispatcher: request parsing, error mapping, telemetry logging, and method routing.
- MemoryRpcControllerMixin: KV store CRUD, prefetch, observation commits, audit verification.
- CausalRpcControllerMixin: what_if counterfactual queries, graph hydration, annealing.
- SensingRpcControllerMixin: shadow worker extraction, predictive coding probes, attention bus.
- SleepRpcControllerMixin: trajectory ingestion, sleep consolidation, session lifecycle.
- SyncRpcControllerMixin: multi-agent CRDT sync, gossip protocol, peer status.
"""
from __future__ import annotations

from atlas_memory.server.controllers.causal import CausalRpcControllerMixin
from atlas_memory.server.controllers.dispatcher import JsonRpcDispatcher
from atlas_memory.server.controllers.memory import MemoryRpcControllerMixin
from atlas_memory.server.controllers.sensing import SensingRpcControllerMixin
from atlas_memory.server.controllers.sleep import SleepRpcControllerMixin
from atlas_memory.server.controllers.sync import SyncRpcControllerMixin

__all__ = [
    "JsonRpcDispatcher",
    "MemoryRpcControllerMixin",
    "CausalRpcControllerMixin",
    "SensingRpcControllerMixin",
    "SleepRpcControllerMixin",
    "SyncRpcControllerMixin",
]
