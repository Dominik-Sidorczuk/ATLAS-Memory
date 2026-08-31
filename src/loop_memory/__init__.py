"""
ATLAS Memory (loop_memory backward compatibility wrapper).
All core modules and classes are defined in atlas_memory and re-exported here.
"""

from atlas_memory import *
import atlas_memory as _atlas_memory
import sys
sys.modules["loop_memory"] = _atlas_memory

from atlas_memory.active.prediction_error import ActiveSensingEngine, PredictionCheck, PredictionError
from atlas_memory.active.shadow_worker import OmniRouteShadowWorker
from atlas_memory.arrow_buffer.trajectory_buffer import ArrowTrajectoryBuffer
from atlas_memory.causal.models import CausalEdge, CausalPath, WhatIfResult

from atlas_memory.causal.retro_causal_edge import RetroCausalEngine
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.extensions.canonicalizer import EntityCanonicalizer, EntityEntry
from atlas_memory.extensions.compactor import CompactionLevel, ContextCompactor
from atlas_memory.extensions.decay_scorer import SalienceDecayEngine
from atlas_memory.extensions.epistemic import EpistemicCalibrator
from loop_memory.hermes.prefix_guard import HermesSessionHook, PrefixCacheGuard
from loop_memory.hermes.tools import (
    COMMIT_OBSERVATION_SCHEMA,
    SEARCH_MEMORY_SCHEMA,
    create_hermes_tool_handlers,
    register_hermes_memory_tools,
)
from loop_memory.hermes_integration import HermesMemoryAdapter
from loop_memory.l0_dynamic.ttt_layer import TTTLayer
from loop_memory.l1_working.jepa_latent import JEPALatentBuffer
from loop_memory.l2_semantic import (
    EpisodicVectorStore,
    FastEmbedEncoder,
    KuzuGraphStore,
    QdrantVectorStore,
    SimpleEmbeddingEncoder,
    SymbolicGraphStore,
    VerifiedKVStore,
)
from loop_memory.l3_procedural.auditor import MemoryAuditor
from loop_memory.l3_procedural.skill_compiler import (
    ASTSafetyScanner,
    SafetyViolationError,
    compile_sop_to_skill,
)
from loop_memory.l3_procedural.sleep_baker import SleepBaker, StandardProcedure, Step
from loop_memory.l3_procedural.weight_baker import BakedProcedure, WeightBaker
from loop_memory.models import (
    ActionPlan,
    ConflictReport,
    ConsolidationStats,
    EpistemicSource,
    LatentState,
    MemoryRecord,
    PredictedTransition,
    RecallResult,
)
from loop_memory.orchestrator import MemoryOrchestrator
from loop_memory.quantization import (
    MIBQuantizer,
    QuantizationConfig,
    QuantizedVector,
    SIMDHamming,
)
from loop_memory.server import AtlasDaemon, AtlasDaemonClient
from loop_memory.sync import (
    DeltaCRDT,
    GossipProtocol,
    LWWElementSet,
    SyncCrypto,
    VectorClock,
)
from loop_memory.sync.transport import (
    GossipTransport,
    InMemoryGossipTransport,
    UDPGossipTransport,
    create_transport,
)
from loop_memory.telemetry.cache_monitor import CacheEvent, CacheHitMonitor

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
