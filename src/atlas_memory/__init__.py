"""
ATLAS Memory: 4-Layer Cognitive Memory Architecture for Hermes Agent.

Subsystems:
- HermesMemoryAdapter: Integration adapter for Hermes Agent.
- MemoryOrchestrator: 6-lever memory policy orchestration (Cache Contract, Retrieval Gate,
  Veracity-First Epistemic Ranking, Structured SPO + Supersede, Shadow Reconciliation, Token Budget Governor).
- RetroCausalEngine: Multi-hop causal extrapolation (causal_what_if) with L1 JEPA world state fusion.
- ActiveSensingEngine: Predictive coding error reduction (discrepancy detection).
- OmniRouteShadowWorker: Asynchronous background extraction worker.
- CacheHitMonitor: Prompt caching telemetry and token savings tracking.
"""

import sys

from atlas_memory.active.prediction_error import ActiveSensingEngine, PredictionCheck, PredictionError
from atlas_memory.active.shadow_worker import OmniRouteShadowWorker
from atlas_memory.arrow_buffer.trajectory_buffer import ArrowTrajectoryBuffer
from atlas_memory.causal.models import CausalEdge, CausalPath, WhatIfResult

# ATLAS Subsystems
from atlas_memory.causal.retro_causal_edge import RetroCausalEngine

# Alias ATLAS_memory for case-insensitive module lookups
sys.modules.setdefault("ATLAS_memory", sys.modules[__name__])
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.extensions.canonicalizer import EntityCanonicalizer, EntityEntry
from atlas_memory.extensions.compactor import CompactionLevel, ContextCompactor
from atlas_memory.extensions.decay_scorer import SalienceDecayEngine
from atlas_memory.extensions.epistemic import EpistemicCalibrator
from atlas_memory.hermes.prefix_guard import HermesSessionHook, PrefixCacheGuard
from atlas_memory.hermes.tools import (
    COMMIT_OBSERVATION_SCHEMA,
    SEARCH_MEMORY_SCHEMA,
    create_hermes_tool_handlers,
    register_hermes_memory_tools,
)
from atlas_memory.hermes_integration import HermesMemoryAdapter
from atlas_memory.l0_dynamic.ttt_layer import TTTLayer
from atlas_memory.l1_working.jepa_latent import JEPALatentBuffer
from atlas_memory.l2_semantic import (
    EpisodicVectorStore,
    FastEmbedEncoder,
    KuzuGraphStore,
    QdrantVectorStore,
    SimpleEmbeddingEncoder,
    SymbolicGraphStore,
    VerifiedKVStore,
)
from atlas_memory.l3_procedural.auditor import MemoryAuditor
from atlas_memory.l3_procedural.skill_compiler import (
    ASTSafetyScanner,
    SafetyViolationError,
    compile_and_register_sop,
    compile_sop_to_skill,
    register_in_hermes_environment,
)
from atlas_memory.l3_procedural.sleep_baker import SleepBaker, StandardProcedure, Step
from atlas_memory.l3_procedural.weight_baker import BakedProcedure, WeightBaker
from atlas_memory.models import (
    ActionPlan,
    ConflictReport,
    ConsolidationStats,
    EpistemicSource,
    LatentState,
    MemoryRecord,
    PredictedTransition,
    RecallResult,
)
from atlas_memory.orchestrator import MemoryOrchestrator
from atlas_memory.quantization import (
    MIBQuantizer,
    QuantizationConfig,
    QuantizedVector,
    SIMDHamming,
)
from atlas_memory.server import AtlasDaemon, AtlasDaemonClient
from atlas_memory.sync import (
    DeltaCRDT,
    GossipProtocol,
    LWWElementSet,
    SyncCrypto,
    VectorClock,
)
from atlas_memory.sync.transport import (
    GossipTransport,
    InMemoryGossipTransport,
    UDPGossipTransport,
    create_transport,
)
from atlas_memory.telemetry.cache_monitor import CacheEvent, CacheHitMonitor

__all__ = [
    # Core Engine & Orchestrator
    "HybridMemoryEngine",
    "MemoryOrchestrator",
    "AtlasDaemon",
    "AtlasDaemonClient",
    "HermesMemoryAdapter",
    # Modele bazowe
    "MemoryRecord",
    "EpistemicSource",
    "LatentState",
    "ActionPlan",
    "PredictedTransition",
    "ConflictReport",
    "ConsolidationStats",
    "RecallResult",
    # CRDT & Sync
    "SyncCrypto",
    "VectorClock",
    "LWWElementSet",
    "DeltaCRDT",
    "GossipProtocol",
    "GossipTransport",
    "UDPGossipTransport",
    "InMemoryGossipTransport",
    "create_transport",
    # Quantization & Acceleration
    "MIBQuantizer",
    "SIMDHamming",
    "QuantizationConfig",
    "QuantizedVector",
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
    "CacheHitMonitor",
    "CacheEvent",
    # L0 / L1 Numba & Arrow
    "TTTLayer",
    "JEPALatentBuffer",
    "ArrowTrajectoryBuffer",
    # L2 Produkcyjne bazy
    "QdrantVectorStore",
    "FastEmbedEncoder",
    "KuzuGraphStore",
    "VerifiedKVStore",
    "EpisodicVectorStore",
    "SymbolicGraphStore",
    "SimpleEmbeddingEncoder",
    # Rozszerzenia klastra
    "SalienceDecayEngine",
    "EntityCanonicalizer",
    "EntityEntry",
    "ContextCompactor",
    "CompactionLevel",
    "EpistemicCalibrator",
    # Hermes Tools & Guard
    "SEARCH_MEMORY_SCHEMA",
    "COMMIT_OBSERVATION_SCHEMA",
    "create_hermes_tool_handlers",
    "register_hermes_memory_tools",
    "PrefixCacheGuard",
    "HermesSessionHook",
]
