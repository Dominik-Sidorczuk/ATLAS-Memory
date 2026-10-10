from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from atlas_memory.cognitive.models import clamp_conf
from atlas_memory.core.exceptions import InvariantViolation
from atlas_memory.core.models import MemoryRecord
from atlas_memory.l2_semantic.repositories import (
    L0_STEP_COUNTER_TABLE,
    L0_TTT_COUNTER_NAME,
    REDACTION_LEGACY_REASON,
    REDACTION_MARKER_PREFIX,
    REDACTION_REGISTRY_TABLE,
    L0StepCounterRepository,
    Sha256AuditLedgerRepository,
    SleepCheckpointRepository,
    SqliteConnectionManager,
    SqliteStateRepository,
    TransactionIntentRepository,
)

__all__ = [
    "VerifiedKVStore",
    "REDACTION_MARKER_PREFIX",
    "REDACTION_REGISTRY_TABLE",
    "REDACTION_LEGACY_REASON",
    "L0_STEP_COUNTER_TABLE",
    "L0_TTT_COUNTER_NAME",
]

logger = logging.getLogger("atlas.l2.kv_store")


class VerifiedKVStore:
    """
    L2: Verified Key-Value / SQL Store (SQLite / JSONB) z obsługą transakcji Saga / 2PC
    oraz kryptograficznym SHA-256 Hash-Chain Audit Log (Source of Truth & Immutability).

    Refactored to Clean Architecture / Facade coordinating dedicated repositories:
    - SqliteConnectionManager: connection lifecycle, pragmas, WAL checkpoints, online backup.
    - SqliteStateRepository: key-value storage, FTS5 indexing, candidate queries.
    - Sha256AuditLedgerRepository: SHA-256 hash-chain audit log, verification, redactions.
    - TransactionIntentRepository: distributed Saga/2PC intent log.
    - SleepCheckpointRepository: sleep consolidation checkpoints.
    - L0StepCounterRepository: persistent monotonic step counters for L0-TTT.
    """

    def __init__(self, db_path: str = ":memory:", audit_db_path: Optional[str] = None):
        self.db_path = db_path
        self._audit_db_path_override = audit_db_path
        self.audit_db_path: Optional[str] = None
        self._lock = asyncio.Lock()
        self._sync_lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._prev_hash: str = "0" * 64
        self._audit_cols: Set[str] = set()

        self._connection_mgr = SqliteConnectionManager()
        self._state_repo = SqliteStateRepository()
        self._audit_repo = Sha256AuditLedgerRepository()
        self._tx_repo = TransactionIntentRepository()
        self._checkpoint_repo = SleepCheckpointRepository()
        self._l0_repo = L0StepCounterRepository()

        self._init_db()

    def _ensure_conn(self) -> sqlite3.Connection:
        """Zwraca aktywny uchwyt bazy danych lub rzuca RuntimeError gdy baza jest zamknięta."""
        if self._conn is None:
            raise RuntimeError("VerifiedKVStore database connection is closed.")
        return self._conn

    def ensure_connection(self) -> sqlite3.Connection:
        """V113 (F-029-class): public invariant guard replacing the private `assert kv._conn is not None`.

        Returns the live sqlite3 connection (which supports the `with` context-manager
        protocol so callers can do `with kv.ensure_connection() as conn:` instead of
        touching the private `_conn` attribute).
        """
        if self._conn is None:
            raise InvariantViolation(
                module="l2_semantic.kv_store",
                name="kv_connection_open",
                observed=0,
                expected=1,
                hint=(
                    "VerifiedKVStore connection is closed; do not write to a dead "
                    "handle. Reopen the store (delete_state or new VerifiedKVStore) "
                    "instead of bypassing the public API."
                ),
            )
        return self._conn

    @staticmethod
    def _compute_entry_hash(seq: int, timestamp: float, key: str, value_hash: str, prev_hash: str) -> str:
        """Deterministyczne obliczanie hasza węzła łańcucha audytu SHA-256."""
        return Sha256AuditLedgerRepository.compute_entry_hash(seq, timestamp, key, value_hash, prev_hash)

    @staticmethod
    def _row_to_state_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """Konwertuje wiersz tabeli state_variables do ujednoliconego słownika stanu z kwarantanną i weryfikacją tamperingu."""
        return SqliteStateRepository.row_to_state_dict(row, json_loads=json.loads)

    def _init_db(self) -> None:
        self._conn, self.audit_db_path, target_audit = self._connection_mgr.connect(
            db_path=self.db_path,
            audit_db_path_override=self._audit_db_path_override,
        )
        with self._conn:
            self._state_repo.init_schema(self._conn)
            self._audit_cols, self._prev_hash = self._audit_repo.init_schema(
                conn=self._conn,
                db_path=self.db_path,
                audit_db_path=self.audit_db_path,
                target_audit=target_audit,
            )
            self._tx_repo.init_schema(self._conn)
            self._checkpoint_repo.init_schema(self._conn)
            self._l0_repo.init_schema(self._conn)

    def _insert_audit_entry_sync(
        self,
        cursor: sqlite3.Cursor,
        seq: int,
        timestamp: float,
        key: str,
        val_hash: str,
        prev_h: str,
        entry_hash: str,
        val_json: str,
        confidence: float,
        reason: str = "update",
    ) -> None:
        """Wstawia wpis do audit logu z zachowaniem zgodności wstecznej schematu."""
        self._audit_repo.insert_audit_entry(
            conn=cursor,
            audit_cols=getattr(self, "_audit_cols", set()),
            seq=seq,
            timestamp=timestamp,
            key=key,
            val_hash=val_hash,
            prev_h=prev_h,
            entry_hash=entry_hash,
            val_json=val_json,
            confidence=confidence,
            reason=reason,
        )

    def _redaction_is_authorized(self, conn: sqlite3.Connection, seq: int, key: str, value_hash: str) -> bool:
        """V87 (T1) / ADR-008: czy redakcja wpisu seq ma wpis w rejestrze audytu."""
        return self._audit_repo.redaction_is_authorized(conn, seq, key, value_hash)

    def authorize_redaction(
        self,
        seq: int,
        key: str,
        value_hash: str,
        reason: str = REDACTION_LEGACY_REASON,
    ) -> Dict[str, Any]:
        """V87 (T1) / ADR-008: publiczne API rejestrowania redakcji w audit_redactions."""
        with self._sync_lock:
            conn = self._ensure_conn()
            with conn:
                return self._audit_repo.authorize_redaction(conn, seq, key, value_hash, reason)

    def redact_audit_entry(
        self,
        seq: int,
        reason: str = "SEC-001",
        redacted_value: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Autoryzowana redakcja TREŚCI wpisu audytu (V87/T1, ADR-008)."""
        with self._sync_lock:
            conn = self._ensure_conn()
            return self._audit_repo.redact_audit_entry(conn, seq, reason, redacted_value)

    def save_l0_step_counter(self, step_count: int, total_energy: float = 0.0) -> None:
        """Zapisuje (monotonicznie) trwały licznik kroków adaptacji L0-TTT (V87/T3)."""
        with self._sync_lock:
            conn = self._ensure_conn()
            with conn:
                self._l0_repo.save_step_counter(conn, step_count, total_energy)

    def load_l0_step_counter(self) -> Optional[Dict[str, Any]]:
        """Zwraca ostatni trwały stan licznika L0-TTT albo None (brak wpisu/błędy)."""
        with self._sync_lock:
            conn = self._ensure_conn()
            return self._l0_repo.load_step_counter(conn)

    async def append_audit_log(self, entry: Dict[str, Any]) -> str:
        """Dołącza nowy wpis do kryptograficznego SHA-256 Hash-Chain."""
        async with self._lock:
            with self._sync_lock:
                conn = self._ensure_conn()
                with conn:
                    entry_hash = self._audit_repo.append_audit_log(
                        conn=conn,
                        audit_cols=getattr(self, "_audit_cols", set()),
                        entry=entry,
                    )
                self._prev_hash = entry_hash
                return entry_hash

    def verify_audit_log_sync(self, deep: bool = True) -> Tuple[bool, int]:
        """Synchronous verification of SHA-256 hash-chain and F-10 cross-table consistency."""
        with self._sync_lock:
            conn = self._ensure_conn()
            cols = getattr(self, "_audit_cols", set())
            is_valid, broken_seq, last_hash = self._audit_repo.verify_audit_log(
                conn=conn,
                audit_cols=cols,
                deep=deep,
            )
            if is_valid and last_hash != "0" * 64:
                self._prev_hash = last_hash
            return is_valid, broken_seq

    def verify_chain_integrity_sync(self) -> Tuple[bool, int]:
        """Synchronous alias delegujący do verify_audit_log_sync(deep=False)."""
        return self.verify_audit_log_sync(deep=False)

    async def verify_audit_log(self, deep: bool = True) -> Tuple[bool, int]:
        """Weryfikuje kryptograficzną integralność całego łańcucha audytu SHA-256."""
        async with self._lock:
            return self.verify_audit_log_sync(deep=deep)

    async def verify_chain_integrity(self) -> Tuple[bool, int]:
        """Alias delegujący do verify_audit_log(deep=False)."""
        return await self.verify_audit_log(deep=False)

    async def verify_audit_log_chain(self, deep: bool = True) -> Tuple[bool, int]:
        """Alias delegujący do verify_audit_log()."""
        return await self.verify_audit_log(deep=deep)

    def verify_audit_log_chain_sync(self, deep: bool = True) -> Tuple[bool, int]:
        """Synchronous alias delegujący do verify_audit_log_sync()."""
        return self.verify_audit_log_sync(deep=deep)

    def heal_audit_log_chain(self) -> Tuple[int, bool]:
        """Synchronously heal SHA-256 hash-chain integrity and clamp confidence values."""
        with self._sync_lock:
            conn = self._ensure_conn()
            cols = getattr(self, "_audit_cols", set())
            count, ok, last_hash = self._audit_repo.heal_audit_log_chain(
                conn=conn,
                audit_cols=cols,
            )
            self._prev_hash = last_hash
            return count, ok

    async def heal_audit_log(self) -> Tuple[int, bool]:
        """Async alias for heal_audit_log_chain."""
        async with self._lock:
            return self.heal_audit_log_chain()

    def purge_stale_probe_keys(self, prefixes: Optional[List[str]] = None) -> int:
        """Purges test / probe keys from state_variables."""
        if not prefixes:
            raise ValueError("prefixes list must be explicitly provided to purge_stale_probe_keys")
        with self._sync_lock:
            conn = self._ensure_conn()
            with conn:
                return self._state_repo.purge_stale_probe_keys(conn, prefixes)

    async def purge_stale_probes(self, prefixes: Optional[List[str]] = None) -> int:
        """Async alias for purge_stale_probe_keys."""
        async with self._lock:
            return self.purge_stale_probe_keys(prefixes=prefixes)

    async def create_transaction_intent(self, tx_id: str, record: MemoryRecord) -> None:
        """Zapisuje intencję transakcji przed wysłaniem do zewnętrznych baz (Kùzu/Qdrant)."""
        async with self._lock:
            conn = self._ensure_conn()
            with conn:
                self._tx_repo.create_intent(conn, tx_id, record)

    async def mark_intent_committed(self, tx_id: str) -> None:
        """Potwierdza pomyślne zakończenie zapisu we wszystkich bazach."""
        async with self._lock:
            conn = self._ensure_conn()
            with conn:
                self._tx_repo.mark_committed(conn, tx_id)

    async def mark_intent_failed(self, tx_id: str, error: str) -> None:
        """Oznacza transakcję jako nieudaną i wymagającą kompensacji."""
        async with self._lock:
            conn = self._ensure_conn()
            with conn:
                self._tx_repo.mark_failed(conn, tx_id, error)

    async def get_dangling_intents(self) -> List[Dict[str, Any]]:
        """Zwraca listę transakcji, które pozostały w stanie PENDING."""
        async with self._lock:
            conn = self._ensure_conn()
            return self._tx_repo.get_dangling_intents(conn)

    def set_sync(
        self,
        key: str,
        value: Any,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        reason: str = "update",
        ttl: Optional[int] = None,
    ) -> None:
        """Synchronous version of set_state for sync ingest & UDS handlers."""
        confidence = clamp_conf(confidence)
        now = time.time()
        meta = dict(metadata or {})
        if ttl is not None:
            meta["ttl"] = int(ttl)
            meta["is_temporary"] = True
            meta["expires_at"] = now + float(ttl)
        val_json = json.dumps(value, default=str)
        meta_json = json.dumps(meta, default=str)
        val_hash = hashlib.sha256(val_json.encode("utf-8")).hexdigest()

        with self._sync_lock:
            conn = self._ensure_conn()
            with conn:
                self._state_repo.set_state(conn, key, val_json, confidence, now, meta_json, val_hash)
                next_seq, prev_h = self._audit_repo.get_last_entry(conn)
                entry_hash = self._audit_repo.compute_entry_hash(next_seq, now, key, val_hash, prev_h)
                self._audit_repo.insert_audit_entry(
                    conn=conn,
                    audit_cols=getattr(self, "_audit_cols", set()),
                    seq=next_seq,
                    timestamp=now,
                    key=key,
                    val_hash=val_hash,
                    prev_h=prev_h,
                    entry_hash=entry_hash,
                    val_json=val_json,
                    confidence=confidence,
                    reason=reason,
                )
            self._prev_hash = entry_hash

    async def set_state(
        self,
        key: str,
        value: Any,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        reason: str = "update",
        ttl: Optional[int] = None,
    ) -> None:
        """Zapisuje stan zmiennej z audytem w bazie danych."""
        async with self._lock:
            self.set_sync(
                key=key,
                value=value,
                confidence=confidence,
                metadata=metadata,
                reason=reason,
                ttl=ttl,
            )

    async def record_sleep_checkpoint(
        self,
        checkpoint_id: str,
        last_processed_step_id: int,
        sessions_consolidated: int = 0,
        sops_baked: int = 0,
        facts_superseded: int = 0,
        duration_ms: float = 0.0,
    ) -> bool:
        """Rejestruje jeden cykl konsolidacji snu w tabeli checkpointów."""
        if not checkpoint_id:
            raise ValueError("checkpoint_id is required for sleep checkpoint record")
        val_json = json.dumps({
            "last_processed_step_id": int(last_processed_step_id),
            "sessions_consolidated": int(sessions_consolidated),
            "sops_baked": int(sops_baked),
            "facts_superseded": int(facts_superseded),
            "duration_ms": float(duration_ms),
        }, default=str)
        now = time.time()

        async with self._lock:
            with self._sync_lock:
                conn = self._ensure_conn()
                with conn:
                    created = self._checkpoint_repo.record_checkpoint(
                        conn=conn,
                        checkpoint_id=checkpoint_id,
                        last_processed_step_id=last_processed_step_id,
                        sessions_consolidated=sessions_consolidated,
                        sops_baked=sops_baked,
                        facts_superseded=facts_superseded,
                        duration_ms=duration_ms,
                        timestamp=now,
                    )
                    if not created:
                        return False
                    val_hash = hashlib.sha256(val_json.encode("utf-8")).hexdigest()
                    next_seq, prev_h = self._audit_repo.get_last_entry(conn)
                    entry_hash = self._audit_repo.compute_entry_hash(next_seq, now, checkpoint_id, val_hash, prev_h)
                    self._audit_repo.insert_audit_entry(
                        conn=conn,
                        audit_cols=getattr(self, "_audit_cols", set()),
                        seq=next_seq,
                        timestamp=now,
                        key=checkpoint_id,
                        val_hash=val_hash,
                        prev_h=prev_h,
                        entry_hash=entry_hash,
                        val_json=val_json,
                        confidence=1.0,
                        reason="sleep_checkpoint",
                    )
                    self._prev_hash = entry_hash
            return True

    async def get_sleep_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Zwraca pojedynczy checkpoint snu lub None."""
        async with self._lock:
            conn = self._ensure_conn()
            return self._checkpoint_repo.get_checkpoint(conn, checkpoint_id)

    async def list_sleep_checkpoints(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Zwraca ostatnie checkpointy konsolidacji snu (od najnowszego)."""
        async with self._lock:
            conn = self._ensure_conn()
            return self._checkpoint_repo.list_checkpoints(conn, limit)

    async def count_sleep_checkpoints(self) -> int:
        """Liczba zapisanych cykli konsolidacji snu."""
        async with self._lock:
            conn = self._ensure_conn()
            return self._checkpoint_repo.count_checkpoints(conn)

    def get_sync(self, key: str) -> Optional[Dict[str, Any]]:
        """Synchronous version of get_state with case-insensitive fallback."""
        with self._sync_lock:
            conn = self._ensure_conn()
            return self._state_repo.get_state(conn, key)

    async def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Asynchroniczne pobranie stanu zmiennej."""
        async with self._lock:
            return self.get_sync(key)

    async def get_states(self, keys: List[str]) -> Dict[str, Any]:
        """Asynchroniczne pobranie wielu stanów zmiennych."""
        async with self._lock:
            conn = self._ensure_conn()
            return self._state_repo.get_states(conn, keys)

    def filter_existing_keys_sync(self, keys: Sequence[str]) -> Set[str]:
        """Zwraca podzbiór kluczy, które faktycznie istnieją w KV (1 zapytanie SQL)."""
        conn = self._ensure_conn()
        return self._state_repo.filter_existing_keys(conn, keys)

    async def filter_existing_keys(self, keys: Sequence[str]) -> Set[str]:
        """Asynchroniczne wsadowe sprawdzanie istnienia kluczy w KV."""
        async with self._lock:
            return self.filter_existing_keys_sync(keys)

    def count_sync(self) -> int:
        """Zwraca liczbę rekordów w state_variables (O(1) w SQLite)."""
        with self._sync_lock:
            conn = self._ensure_conn()
            return self._state_repo.count(conn)

    async def count(self) -> int:
        """Asynchroniczne zliczanie rekordów."""
        async with self._lock:
            return self.count_sync()

    def delete_sync(self, key: str) -> bool:
        """Synchroniczne usunięcie zmiennej stanu ze sklepu KV z wpisem do łańcucha audytu SHA-256."""
        with self._sync_lock:
            conn = self._ensure_conn()
            with conn:
                deleted = self._state_repo.delete_state(conn, key)
                if deleted:
                    now = time.time()
                    val_hash = hashlib.sha256(b'"__DELETED__"').hexdigest()
                    next_seq, prev_h = self._audit_repo.get_last_entry(conn)
                    entry_hash = self._audit_repo.compute_entry_hash(next_seq, now, key, val_hash, prev_h)
                    self._audit_repo.insert_audit_entry(
                        conn=conn,
                        audit_cols=getattr(self, "_audit_cols", set()),
                        seq=next_seq,
                        timestamp=now,
                        key=key,
                        val_hash=val_hash,
                        prev_h=prev_h,
                        entry_hash=entry_hash,
                        val_json='"__DELETED__"',
                        confidence=0.0,
                        reason="delete",
                    )
                    self._prev_hash = entry_hash
            return deleted

    async def delete_state(self, key: str) -> bool:
        """Asynchroniczne usunięcie zmiennej stanu ze sklepu KV z wpisem do łańcucha audytu SHA-256."""
        async with self._lock:
            return self.delete_sync(key)

    def search_candidates_sync(
        self,
        query: str = "",
        limit: int = 100,
        words: Optional[List[str]] = None,
        stems: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Wyszukiwanie kandydatów w SQL (indeks PK + FTS5) zamiast O(N) skanu tabeli."""
        with self._sync_lock:
            conn = self._ensure_conn()
            return self._state_repo.search_candidates(
                conn=conn,
                query=query,
                limit=limit,
                words=words,
                stems=stems,
            )

    def get_arbitration_candidates_sync(
        self,
        key: str,
        value: Any = None,
        limit: int = 100,
    ) -> Dict[str, Dict[str, Any]]:
        """Zawężanie kandydatów dla ConflictArbiter w SQL zamiast O(N) skanu."""
        with self._sync_lock:
            conn = self._ensure_conn()
            return self._state_repo.get_arbitration_candidates(
                conn=conn,
                key=key,
                value=value,
                limit=limit,
            )

    def get_all_sync(self, limit: int = 1000, offset: int = 0, paged: bool = True) -> Dict[str, Any]:
        """Stronicowane pobieranie rekordów KV."""
        if paged is False:
            raise ValueError(
                "get_all_sync without pagination is prohibited (V80.3 T1). "
                "Specify paged=True and limit to iterate safely over state_variables."
            )
        with self._sync_lock:
            conn = self._ensure_conn()
            return self._state_repo.get_all(conn, limit=limit, offset=offset)

    async def get_all_states(self, limit: int = 1000, offset: int = 0, paged: bool = True) -> Dict[str, Any]:
        """Asynchroniczne stronicowane pobieranie rekordów KV."""
        if paged is False:
            raise ValueError(
                "get_all_states without pagination is prohibited (V80.3 T1). "
                "Specify paged=True and limit."
            )
        async with self._lock:
            return self.get_all_sync(limit=limit, offset=offset, paged=True)

    def get_recent_sync(self, max_age_seconds: float = 900.0, limit: int = 25) -> List[Dict[str, Any]]:
        """Zwraca najświeższe zapisane stany posortowane chronologicznie."""
        with self._sync_lock:
            conn = self._ensure_conn()
            return self._state_repo.get_recent(conn, max_age_seconds=max_age_seconds, limit=limit)

    # Aliases
    set = set_sync
    get = get_sync
    get_all = get_all_sync
    get_recent = get_recent_sync

    def checkpoint_wal_sync(self, truncate: bool = True) -> Dict[str, Any]:
        """Wykonuje kontrolowany WAL checkpoint dla bazy głównej i bazy audytu."""
        with self._sync_lock:
            return self._connection_mgr.checkpoint_wal(
                conn=self._conn,
                db_path=self.db_path,
                audit_db_path=self.audit_db_path,
                truncate=truncate,
            )

    async def checkpoint_wal(self, truncate: bool = True) -> Dict[str, Any]:
        """Asynchroniczny wrapper dla checkpoint_wal_sync."""
        async with self._lock:
            return self.checkpoint_wal_sync(truncate=truncate)

    def backup_audit_db_sync(self, dest_path: Path | str) -> Path:
        """Bezpieczna procedura kopii zapasowej bazy audytu (FINDING-072)."""
        with self._sync_lock:
            return self._connection_mgr.backup_audit_db(
                conn=self._conn,
                audit_db_path=self.audit_db_path,
                dest_path=dest_path,
                checkpoint_fn=self.checkpoint_wal_sync,
            )

    def close_sync(self) -> None:
        """Zamyka połączenie ze sklepem KV po wykonaniu TRUNCATE checkpoint na WAL."""
        with self._sync_lock:
            self._connection_mgr.close(
                conn=self._conn,
                db_path=self.db_path,
                audit_db_path=self.audit_db_path,
                checkpoint_fn=self.checkpoint_wal_sync,
            )
            self._conn = None

    async def close(self) -> None:
        """Asynchroniczne zamknięcie sklepu KV z checkpointem WAL."""
        async with self._lock:
            self.close_sync()

    def count_audit_entries_sync(self) -> int:
        """Return total number of entries in state_audit_log cleanly without exposing raw SQLite connection."""
        with self._sync_lock:
            conn = self._ensure_conn()
            return self._audit_repo.count_audit_entries(conn)

    def savepoint(self, name: str = "saga_tx") -> None:
        """Creates a transaction savepoint across KV and attached audit databases."""
        with self._sync_lock:
            conn = self._ensure_conn()
            conn.execute(f"SAVEPOINT {name};")
            if not hasattr(self, "_savepoint_prev_hashes"):
                self._savepoint_prev_hashes: Dict[str, str] = {}
            self._savepoint_prev_hashes[name] = str(self._prev_hash)

    def rollback_savepoint(self, name: str = "saga_tx") -> None:
        """Rolls back to named savepoint, restoring previous DB state and SHA-256 tip without audit mutation."""
        with self._sync_lock:
            conn = self._ensure_conn()
            conn.execute(f"ROLLBACK TO {name};")
            conn.execute(f"RELEASE {name};")
            if hasattr(self, "_savepoint_prev_hashes") and name in self._savepoint_prev_hashes:
                self._prev_hash = self._savepoint_prev_hashes.pop(name)
            else:
                _, self._prev_hash = self._audit_repo.get_last_entry(conn)

    def release_savepoint(self, name: str = "saga_tx") -> None:
        """Releases named savepoint on successful multi-store commit."""
        with self._sync_lock:
            conn = self._ensure_conn()
            conn.execute(f"RELEASE {name};")
            if hasattr(self, "_savepoint_prev_hashes"):
                self._savepoint_prev_hashes.pop(name, None)

    def get_latest_audit_hash(self) -> str:
        """Returns the current cryptographic tip hash of the SHA-256 audit log."""
        with self._sync_lock:
            conn = self._ensure_conn()
            _, prev_h = self._audit_repo.get_last_entry(conn)
            return prev_h
