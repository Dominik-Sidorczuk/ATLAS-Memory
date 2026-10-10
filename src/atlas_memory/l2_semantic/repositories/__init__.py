"""L2 Semantic storage repositories package.

Decomposed repositories following Single Responsibility Principle (SRP):
- SqliteConnectionManager: connection lifecycle, pragmas, WAL checkpoints, online backup.
- SqliteStateRepository: key-value storage, FTS5 full-text indexing, candidate queries.
- Sha256AuditLedgerRepository: SHA-256 hash-chain audit log, verification, redactions.
- TransactionIntentRepository: distributed Saga/2PC intent log.
- SleepCheckpointRepository: sleep consolidation checkpoints.
- L0StepCounterRepository: persistent monotonic step counters for L0-TTT.
"""
from __future__ import annotations

from atlas_memory.l2_semantic.repositories.audit_repository import (
    REDACTION_LEGACY_REASON,
    REDACTION_MARKER_PREFIX,
    REDACTION_REGISTRY_TABLE,
    Sha256AuditLedgerRepository,
)
from atlas_memory.l2_semantic.repositories.checkpoint_repository import (
    SleepCheckpointRepository,
)
from atlas_memory.l2_semantic.repositories.connection_manager import (
    SqliteConnectionManager,
)
from atlas_memory.l2_semantic.repositories.l0_counter_repository import (
    L0_STEP_COUNTER_TABLE,
    L0_TTT_COUNTER_NAME,
    L0StepCounterRepository,
)
from atlas_memory.l2_semantic.repositories.state_repository import (
    SqliteStateRepository,
)
from atlas_memory.l2_semantic.repositories.transaction_repository import (
    TransactionIntentRepository,
)

__all__ = [
    "SqliteConnectionManager",
    "SqliteStateRepository",
    "Sha256AuditLedgerRepository",
    "TransactionIntentRepository",
    "SleepCheckpointRepository",
    "L0StepCounterRepository",
    "REDACTION_MARKER_PREFIX",
    "REDACTION_REGISTRY_TABLE",
    "REDACTION_LEGACY_REASON",
    "L0_STEP_COUNTER_TABLE",
    "L0_TTT_COUNTER_NAME",
]
