"""
Domain models for the ATLAS Cognitive Layer (V27+ Architecture).

Clean, strictly typed Pydantic v2 models for:
- ArbitrationCandidate, ArbitrationResult (Two-Stage Conflict Arbiter)
- PredictionVerdict, PredictionDiscrepancy (Active Sensing / Predictive Coding)
- RetrievalRecord, RetrievalContext (Cognitive Prefetch & Recall)
- HotBufferEntry (Working Memory Hot Buffer)
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

HERMES_DEFAULT_SESSION: str = "hermes_default"
HERMES_GLOBAL_SESSION: str = "global"


def clamp_conf(val: Any) -> float:
    """Klampuje wartość zaufania/pewności (confidence) do przedziału [0.0, 1.0]."""
    try:
        raw = float(val if val is not None else 1.0)
        if math.isnan(raw):
            raw = 1.0
    except (ValueError, TypeError):
        raw = 1.0
    return max(0.0, min(1.0, raw))



class ArbitrationCandidate(BaseModel):
    """Candidate record considered during epistemic conflict arbitration."""

    model_config = ConfigDict(extra="allow")

    key: str = Field(..., description="Key of the candidate record in storage")
    score: float = Field(default=0.0, description="Multi-factor candidate conflict score")
    reason: str = Field(..., description="Reason for candidate collision (e.g. exclusivity_collision, prohibition_contradiction)")
    conflict_type: Optional[str] = Field(default=None, description="Detailed category of contradiction")
    conflicting_fact: Optional[Any] = Field(default=None, description="Value or fact text of existing candidate")
    incoming_claim: Optional[Any] = Field(default=None, description="Incoming claim value or text")
    overlapping_terms: List[str] = Field(default_factory=list, description="Shared or conflicting terms/stems")
    message: Optional[str] = Field(default=None, description="Explanatory human-readable arbitration message")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence of existing candidate")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata associated with candidate")


class ArbitrationResult(BaseModel):
    """Result of cognitive conflict arbitration between incoming write and memory."""

    model_config = ConfigDict(extra="allow")

    is_conflict: bool = Field(default=False, description="True only for high-confidence true positive conflicts")
    is_belief_revision: bool = Field(default=False, description="True when incoming write supersedes established belief")
    action: str = Field(default="accept", description="'accept', 'warn_conflict', 'supersede', 'reject'")
    superseded_key: Optional[str] = Field(default=None, description="Key of established item superseded by incoming write")
    conflicting_key: Optional[str] = Field(default=None, description="Key of conflicting item causing collision")
    reason: Optional[str] = Field(default=None, description="Classification of conflict or revision")
    message: Optional[str] = Field(default=None, description="Descriptive diagnostic message")
    warning: Optional[str] = Field(default=None, description="Warning tag emitted (e.g. 'epistemic_conflict')")
    conflict_details: Optional[Dict[str, Any]] = Field(default=None, description="Detailed dictionary for legacy caller compatibility")
    candidates: List[ArbitrationCandidate] = Field(default_factory=list, description="All evaluated candidates sorted descending by score")
    best_candidate: Optional[ArbitrationCandidate] = Field(default=None, description="Top-scoring candidate")
    best_score: float = Field(default=0.0, description="Highest score among candidates")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional arbitration context and candidates")


class PredictionVerdict(BaseModel):
    """Cognitive active sensing verdict validating expectations against observations."""

    model_config = ConfigDict(extra="allow")

    verdict_id: str = Field(..., description="Unique verdict identifier")
    status: str = Field(default="ok", description="'ok', 'discrepancy_detected', 'error'")
    has_error: bool = Field(default=False, description="True if observation deviates from expected value")
    severity: str = Field(default="LOW", description="'INFO', 'LOW', 'MEDIUM', 'WARNING', 'CRITICAL'")
    discrepancy_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalized discrepancy magnitude")
    target_entity: str = Field(..., description="Entity being probed")
    expected_value: Any = Field(..., description="Predicted / prior expected value")
    observed_value: Any = Field(..., description="Actual observed value from environment")
    tolerance: Optional[float] = Field(default=None, description="Allowed deviation range for numeric values")
    anneal_triggered: bool = Field(default=False, description="True if discrepancy triggered causal graph auto-annealing")
    world_model_updated: bool = Field(default=False, description="True if corrective observation was committed back to world model KV")
    timestamp: float = Field(default_factory=time.time, description="Timestamp of verdict evaluation")
    message: Optional[str] = Field(default=None, description="Diagnostic verdict description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Telemetry and sensor metadata")


class PredictionDiscrepancy(BaseModel):
    """Detailed discrepancy payload between world model expectations and environment observations."""

    model_config = ConfigDict(extra="allow")

    discrepancy_id: str = Field(..., description="Unique discrepancy event ID")
    target_entity: str = Field(..., description="Target entity or probe name")
    predicate: str = Field(default="state", description="Observed predicate or attribute")
    expected_value: str = Field(..., description="Expected value string representation")
    observed_value: str = Field(..., description="Observed value string representation")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Discrepancy score")
    relative_difference: Optional[float] = Field(default=None, description="Fractional relative difference if numeric")
    severity: str = Field(default="WARNING", description="'INFO', 'LOW', 'WARNING', 'CRITICAL'")
    timestamp: float = Field(default_factory=time.time, description="Timestamp of discrepancy detection")
    context: Dict[str, Any] = Field(default_factory=dict, description="Contextual environment properties")


class RetrievalRecord(BaseModel):
    """High-veracity memory record surfaced in cognitive retrieval/prefetch."""

    model_config = ConfigDict(extra="allow")

    subject: str = Field(..., description="Entity or key subject")
    predicate: str = Field(default="state", description="Relationship predicate or slot")
    object: str = Field(..., description="Content, value or entity target")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Epistemic confidence level")
    source_type: str = Field(default="user_explicit", description="'user_explicit', 'tool_output', 'agent_inference'")
    score: float = Field(default=1.0, description="Ranking veracity score")
    is_state_variable: bool = Field(default=False, description="True if record is verified KV state variable")
    is_superseded: bool = Field(default=False, description="True if record has been superseded by newer knowledge")
    timestamp: float = Field(default_factory=time.time, description="Creation or update epoch timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata tags")


class RetrievalContext(BaseModel):
    """Prefetch or recall context payload supplied to the agent."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(..., description="User query or context prompt")
    session_id: Optional[str] = Field(default=None, description="Hermes session identifier")
    records: List[RetrievalRecord] = Field(default_factory=list, description="Retrieved records sorted by veracity")
    count: int = Field(default=0, description="Number of records returned")
    skipped: bool = Field(default=False, description="True if retrieval gate skipped execution")
    latency_ms: float = Field(default=0.0, description="Execution duration in milliseconds")
    cache_hit: bool = Field(default=False, description="True if served from cognitive prefetch cache")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session and gate telemetry")


class HotBufferEntry(BaseModel):
    """Item stored in working memory hot-KV buffer for immediate continuation."""

    model_config = ConfigDict(extra="allow")

    key: str = Field(..., description="Unique memory key")
    value: Any = Field(..., description="Stored value payload")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence rating")
    source_type: str = Field(default="user_explicit", description="Epistemic source provenance")
    timestamp: float = Field(default_factory=time.time, description="Epoch timestamp of insertion")
    is_superseded: bool = Field(default=False, description="True if invalidated or superseded")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Item metadata")

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            val = getattr(self, key)
            return val if val is not None else default
        return self.metadata.get(key, default)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.metadata[key]
