"""
ATLAS Memory: 4-Layer Cognitive Memory Architecture for Hermes Agent.

Subsystems:
- AtlasMemoryProvider: Native Hermes memory provider with prompt-caching contract.
- MemoryOrchestrator: 6-lever memory policy orchestration (Cache Contract, Retrieval Gate,
  Veracity-First Epistemic Ranking, Structured SPO + Supersede, Shadow Reconciliation, Token Budget Governor).
- RetroCausalEngine: Multi-hop causal extrapolation (causal_what_if) with L1 JEPA world state fusion.
- ActiveSensingEngine: Predictive coding error reduction (discrepancy detection).
- OmniRouteShadowWorker: Asynchronous background SPO extraction worker (daemon-owned lifecycle).
"""

import logging
import sys
from typing import Any

__version__ = "2.0.0"

# Alias ATLAS_memory for case-insensitive module lookups
sys.modules.setdefault("ATLAS_memory", sys.modules[__name__])

# Pickle backward compatibility for legacy on-disk Qdrant points (ADR-001)
from atlas_memory.core import models as _core_models

sys.modules.setdefault("atlas_memory.models", _core_models)

# Base models & client are pure Python (always available)
from atlas_memory.core.models import (
    ActionPlan,
    ConflictReport,
    ConsolidationStats,
    EpistemicSource,
    LatentState,
    MemoryRecord,
    PredictedTransition,
)
from atlas_memory.ipc.uds_client import AtlasDaemonClient

# ADR-001 §4: Canonical registry of optional modules and rationale
OPTIONAL_MODULES: dict[str, str] = {
    "atlas_memory.l3_procedural.weight_baker": "Optional LoRA/PEFT procedural distillation (waiting-consumer under ADR-001)",
}

# Leniwe ładowanie modułów (PEP 562) dla eliminacji narzutu importu C-extension/ONNX
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Core Engine & Orchestrator
    "HybridMemoryEngine": ("atlas_memory.engine", "HybridMemoryEngine"),
    "MemoryOrchestrator": ("atlas_memory.orchestrator", "MemoryOrchestrator"),
    "AtlasDaemon": ("atlas_memory.server", "AtlasDaemon"),
    "AtlasDaemonClient": ("atlas_memory.server", "AtlasDaemonClient"),
    "AtlasMemoryProvider": ("atlas_memory.hermes", "AtlasMemoryProvider"),
    # CRDT & Sync
    "SyncCrypto": ("atlas_memory.sync", "SyncCrypto"),
    "VectorClock": ("atlas_memory.sync", "VectorClock"),
    "DeltaCRDT": ("atlas_memory.sync", "DeltaCRDT"),
    "LWWElementSet": ("atlas_memory.sync", "LWWElementSet"),
    "GossipProtocol": ("atlas_memory.sync", "GossipProtocol"),
    "StateSyncBridge": ("atlas_memory.sync", "StateSyncBridge"),
    "PeerIdentity": ("atlas_memory.sync", "PeerIdentity"),
    "GossipTransport": ("atlas_memory.sync.transport", "GossipTransport"),
    "UDPGossipTransport": ("atlas_memory.sync.transport", "UDPGossipTransport"),
    "InMemoryGossipTransport": ("atlas_memory.sync.transport", "InMemoryGossipTransport"),
    "create_transport": ("atlas_memory.sync.transport", "create_transport"),
    # Procedural & Skill Compiler
    "compile_sop_to_skill": ("atlas_memory.l3_procedural.skill_compiler", "compile_sop_to_skill"),
    "register_in_hermes_environment": ("atlas_memory.l3_procedural.skill_compiler", "register_in_hermes_environment"),
    "compile_and_register_sop": ("atlas_memory.l3_procedural.skill_compiler", "compile_and_register_sop"),
    "ASTSafetyScanner": ("atlas_memory.l3_procedural.skill_compiler", "ASTSafetyScanner"),
    "SafetyViolationError": ("atlas_memory.l3_procedural.skill_compiler", "SafetyViolationError"),
    "SleepBaker": ("atlas_memory.l3_procedural.sleep_baker", "SleepBaker"),
    "StandardProcedure": ("atlas_memory.l3_procedural.sleep_baker", "StandardProcedure"),
    "Step": ("atlas_memory.l3_procedural.sleep_baker", "Step"),
    "MemoryAuditor": ("atlas_memory.l3_procedural.auditor", "MemoryAuditor"),
    "WeightBaker": ("atlas_memory.l3_procedural.weight_baker", "WeightBaker"),
    "BakedProcedure": ("atlas_memory.l3_procedural.weight_baker", "BakedProcedure"),
    # ATLAS Causal & Active
    "RetroCausalEngine": ("atlas_memory.causal.retro_causal_edge", "RetroCausalEngine"),
    "CausalPath": ("atlas_memory.causal.models", "CausalPath"),
    "CausalEdge": ("atlas_memory.causal.models", "CausalEdge"),
    "WhatIfResult": ("atlas_memory.causal.models", "WhatIfResult"),
    "ActiveSensingEngine": ("atlas_memory.active.prediction_error", "ActiveSensingEngine"),
    "PredictionCheck": ("atlas_memory.active.prediction_error", "PredictionCheck"),
    "PredictionError": ("atlas_memory.active.prediction_error", "PredictionError"),
    "OmniRouteShadowWorker": ("atlas_memory.active.shadow_worker", "OmniRouteShadowWorker"),
    # L0 / L1 Numba & Arrow
    "TTTLayer": ("atlas_memory.l0_dynamic.ttt_layer", "TTTLayer"),
    "JEPALatentBuffer": ("atlas_memory.l1_working.jepa_latent", "JEPALatentBuffer"),
    "ArrowTrajectoryBuffer": ("atlas_memory.l1_working.trajectory_buffer", "ArrowTrajectoryBuffer"),
    # L2 Produkcyjne bazy
    "QdrantVectorStore": ("atlas_memory.l2_semantic", "QdrantVectorStore"),
    "OnnxDenseEncoder": ("atlas_memory.l2_semantic", "OnnxDenseEncoder"),
    "KuzuGraphStore": ("atlas_memory.l2_semantic", "KuzuGraphStore"),
    "VerifiedKVStore": ("atlas_memory.l2_semantic", "VerifiedKVStore"),
    # Rozszerzenia klastra
    "SalienceDecayEngine": ("atlas_memory.extensions.decay_scorer", "SalienceDecayEngine"),
    "EntityCanonicalizer": ("atlas_memory.extensions.canonicalizer", "EntityCanonicalizer"),
    "EntityEntry": ("atlas_memory.extensions.canonicalizer", "EntityEntry"),
    "ContextCompactor": ("atlas_memory.extensions.compactor", "ContextCompactor"),
    "CompactionLevel": ("atlas_memory.extensions.compactor", "CompactionLevel"),
    # Hermes Tools & Guard
    "SEARCH_MEMORY_SCHEMA": ("atlas_memory.hermes.tools", "SEARCH_MEMORY_SCHEMA"),
    "COMMIT_OBSERVATION_SCHEMA": ("atlas_memory.hermes.tools", "COMMIT_OBSERVATION_SCHEMA"),
    "create_hermes_tool_handlers": ("atlas_memory.hermes.tools", "create_hermes_tool_handlers"),
    "register_hermes_memory_tools": ("atlas_memory.hermes.tools", "register_hermes_memory_tools"),
    "PrefixCacheGuard": ("atlas_memory.hermes.prefix_guard", "PrefixCacheGuard"),
    "HermesSessionHook": ("atlas_memory.hermes.prefix_guard", "HermesSessionHook"),
}

def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        mod_name, attr_name = _LAZY_EXPORTS[name]
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            val = getattr(mod, attr_name)
            globals()[name] = val
            return val
        except ImportError as _err:
            logging.getLogger(__name__).debug("Optional module %s not available: %s", mod_name, _err)
            raise AttributeError(f"module {__name__!r} attribute {name!r} unavailable: {_err}") from _err
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    # Core Engine & Orchestrator
    "HybridMemoryEngine",
    "MemoryOrchestrator",
    "AtlasDaemon",
    "AtlasDaemonClient",
    "AtlasMemoryProvider",
    # Modele bazowe
    "MemoryRecord",
    "EpistemicSource",
    "LatentState",
    "ActionPlan",
    "PredictedTransition",
    "ConflictReport",
    "ConsolidationStats",
    # CRDT & Sync
    "SyncCrypto",
    "VectorClock",
    "DeltaCRDT",
    "LWWElementSet",
    "GossipProtocol",
    "StateSyncBridge",
    "PeerIdentity",
    "GossipTransport",
    "UDPGossipTransport",
    "InMemoryGossipTransport",
    "create_transport",
    # Procedural & Skill Compiler
    "compile_sop_to_skill",
    "register_in_hermes_environment",
    "compile_and_register_sop",
    "ASTSafetyScanner",
    "SafetyViolationError",
    "SleepBaker",
    "StandardProcedure",
    "Step",
    "MemoryAuditor",
    "WeightBaker",
    "BakedProcedure",
    # ATLAS Causal & Active
    "RetroCausalEngine",
    "CausalPath",
    "CausalEdge",
    "WhatIfResult",
    "ActiveSensingEngine",
    "PredictionCheck",
    "PredictionError",
    "OmniRouteShadowWorker",
    # L0 / L1 Numba & Arrow
    "TTTLayer",
    "JEPALatentBuffer",
    "ArrowTrajectoryBuffer",
    # L2 Produkcyjne bazy
    "QdrantVectorStore",
    "OnnxDenseEncoder",
    "KuzuGraphStore",
    "VerifiedKVStore",
    # Rozszerzenia klastra
    "SalienceDecayEngine",
    "EntityCanonicalizer",
    "EntityEntry",
    "ContextCompactor",
    "CompactionLevel",
    # Hermes Tools & Guard
    "SEARCH_MEMORY_SCHEMA",
    "COMMIT_OBSERVATION_SCHEMA",
    "create_hermes_tool_handlers",
    "register_hermes_memory_tools",
    "PrefixCacheGuard",
    "HermesSessionHook",
]
