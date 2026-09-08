from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from atlas_memory.cognitive.models import clamp_conf
from atlas_memory.models import MemoryRecord

logger = logging.getLogger("atlas.l2.kv_store")


class VerifiedKVStore:
    """
    L2: Verified Key-Value / SQL Store (SQLite / JSONB) z obsługą transakcji Saga / 2PC
    oraz kryptograficznym SHA-256 Hash-Chain Audit Log (Source of Truth & Immutability).
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._lock = asyncio.Lock()
        self._sync_lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._prev_hash: str = "0" * 64
        self._init_db()

    def _ensure_conn(self) -> sqlite3.Connection:
        """Zwraca aktywny uchwyt bazy danych lub rzuca RuntimeError gdy baza jest zamknięta."""
        if self._conn is None:
            raise RuntimeError("VerifiedKVStore database connection is closed.")
        return self._conn

    @staticmethod
    def _compute_entry_hash(seq: int, timestamp: float, key: str, value_hash: str, prev_hash: str) -> str:
        """Deterministyczne obliczanie hasza węzła łańcucha audytu SHA-256."""
        payload_str = f"{seq}:{timestamp}:{key}:{value_hash}:{prev_hash}"
        return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

    @staticmethod
    def _row_to_state_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """Konwertuje wiersz tabeli state_variables do ujednoliconego słownika stanu."""
        val = row["value"]
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except Exception:
                val = row["value"]

        meta = row["metadata"]
        if isinstance(meta, str):
            try:
                parsed_meta = json.loads(meta) if meta else {}
                meta = parsed_meta if isinstance(parsed_meta, dict) else {}
            except Exception:
                meta = {}
        elif not isinstance(meta, dict):
            meta = {}

        return {
            "key": row["key"],
            "value": val,
            "confidence": row["confidence"],
            "timestamp": row["timestamp"],
            "metadata": meta,
        }

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Włączenie trybu WAL i zoptymalizowanych flag I/O poza transakcją
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn.execute("PRAGMA cache_size=-64000;")

        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS state_variables (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata TEXT
                )
            """)
            # Tabela SHA-256 Hash-Chain Audit Log
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS state_audit_log (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    key TEXT NOT NULL,
                    value_hash TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL,
                    new_value TEXT,
                    confidence REAL DEFAULT 1.0,
                    reason TEXT DEFAULT 'update'
                )
            """)
            # Tabela intencji transakcyjnych dla wzorca Saga (zapobieganie split-brain state)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS transaction_intent_log (
                    tx_id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    status TEXT NOT NULL, -- PENDING, COMMITTED, COMPENSATING, FAILED
                    timestamp REAL NOT NULL,
                    error TEXT
                )
            """)

            with contextlib.closing(self._conn.cursor()) as cursor:
                # Sprawdź dostępne kolumny w state_audit_log i dodaj brakujące
                cursor.execute("PRAGMA table_info(state_audit_log)")
                self._audit_cols = {row["name"] for row in cursor.fetchall()}
                for col_name, col_def in (
                    ("new_value", "TEXT"),
                    ("confidence", "REAL DEFAULT 1.0"),
                    ("reason", "TEXT DEFAULT 'update'"),
                ):
                    if col_name not in self._audit_cols:
                        try:
                            cursor.execute(f"ALTER TABLE state_audit_log ADD COLUMN {col_name} {col_def}")
                            self._audit_cols.add(col_name)
                        except Exception:
                            pass

                # Inicjalizacja _prev_hash na podstawie ostatniego wpisu w bazie (jeśli istnieje)
                cursor.execute("SELECT entry_hash FROM state_audit_log ORDER BY seq DESC LIMIT 1")
                row = cursor.fetchone()
                if row and row["entry_hash"]:
                    self._prev_hash = row["entry_hash"]
                else:
                    self._prev_hash = "0" * 64

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
        if "new_value" in getattr(self, "_audit_cols", set()):
            cursor.execute("""
                INSERT INTO state_audit_log (seq, timestamp, key, value_hash, prev_hash, entry_hash, new_value, confidence, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (seq, timestamp, key, val_hash, prev_h, entry_hash, val_json, confidence, reason))
        else:
            cursor.execute("""
                INSERT INTO state_audit_log (seq, timestamp, key, value_hash, prev_hash, entry_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (seq, timestamp, key, val_hash, prev_h, entry_hash))

    async def append_audit_log(self, entry: Dict[str, Any]) -> str:
        """
        Dołącza nowy wpis do kryptograficznego SHA-256 Hash-Chain.
        entry_hash = SHA-256(seq + timestamp + key + value_hash + prev_hash)
        """
        async with self._lock:
            conn = self._ensure_conn()
            with conn:
                with contextlib.closing(conn.cursor()) as cursor:
                    # Pobierz kolejny seq
                    cursor.execute("SELECT IFNULL(MAX(seq), 0) + 1 AS next_seq FROM state_audit_log")
                    next_seq = cursor.fetchone()["next_seq"]

                    now = float(entry.get("timestamp", time.time()))
                    key = str(entry.get("key", ""))
                    raw_val = entry.get("value")
                    if "value_hash" in entry:
                        value_hash = str(entry["value_hash"])
                    else:
                        val_bytes = json.dumps(raw_val, sort_keys=True).encode("utf-8") if raw_val is not None else b""
                        value_hash = hashlib.sha256(val_bytes).hexdigest()

                    prev_h = self._prev_hash
                    entry_hash = self._compute_entry_hash(next_seq, now, key, value_hash, prev_h)

                    cursor.execute("""
                        INSERT INTO state_audit_log (seq, timestamp, key, value_hash, prev_hash, entry_hash)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (next_seq, now, key, value_hash, prev_h, entry_hash))

            self._prev_hash = entry_hash
            return entry_hash

    def verify_audit_log_sync(self, deep: bool = False) -> Tuple[bool, int]:
        """
        Synchronous version of verify_audit_log.
        Weryfikuje kryptograficzną integralność całego łańcucha audytu SHA-256 bez pętli asynchronicznej.
        Gdy deep=True, weryfikuje również czy value_hash odpowiada sha256(new_value).
        Zwraca (is_valid, broken_at_seq). Jeśli wszystko poprawne: (True, 0).
        """
        with self._sync_lock:
            conn = self._ensure_conn()
            with contextlib.closing(conn.cursor()) as cursor:
                cols = getattr(self, "_audit_cols", set())
                if "new_value" in cols:
                    cursor.execute(
                        "SELECT seq, timestamp, key, value_hash, prev_hash, entry_hash, new_value "
                        "FROM state_audit_log ORDER BY seq ASC"
                    )
                else:
                    cursor.execute(
                        "SELECT seq, timestamp, key, value_hash, prev_hash, entry_hash "
                        "FROM state_audit_log ORDER BY seq ASC"
                    )
                rows = cursor.fetchall()

            expected_prev = "0" * 64
            for row in rows:
                seq = row["seq"]
                ts = row["timestamp"]
                key = row["key"]
                val_h = row["value_hash"]
                prev_h = row["prev_hash"]
                stored_entry_h = row["entry_hash"]

                if prev_h != expected_prev:
                    return False, seq

                if deep and "new_value" in cols and row["new_value"] is not None:
                    calc_val_h = hashlib.sha256(str(row["new_value"]).encode("utf-8")).hexdigest()
                    if calc_val_h != val_h:
                        return False, seq

                computed_h = self._compute_entry_hash(seq, ts, key, val_h, prev_h)
                if computed_h != stored_entry_h:
                    return False, seq

                expected_prev = stored_entry_h

            if rows:
                self._prev_hash = expected_prev

            return True, 0

    def verify_chain_integrity_sync(self) -> Tuple[bool, int]:
        """Synchronous alias delegujący do verify_audit_log_sync()."""
        return self.verify_audit_log_sync(deep=False)

    async def verify_audit_log(self, deep: bool = False) -> Tuple[bool, int]:
        """
        Weryfikuje kryptograficzną integralność całego łańcucha audytu SHA-256.
        Gdy deep=True, weryfikuje również czy value_hash odpowiada sha256(new_value).
        Zwraca (is_valid, broken_at_seq). Jeśli wszystko poprawne: (True, 0).
        """
        async with self._lock:
            return self.verify_audit_log_sync(deep=deep)

    async def verify_chain_integrity(self) -> Tuple[bool, int]:
        """Alias delegujący do verify_audit_log()."""
        return await self.verify_audit_log(deep=False)

    async def verify_audit_log_chain(self, deep: bool = False) -> Tuple[bool, int]:
        """Alias delegujący do verify_audit_log()."""
        return await self.verify_audit_log(deep=deep)

    def verify_audit_log_chain_sync(self, deep: bool = False) -> Tuple[bool, int]:
        """Synchronous alias delegujący do verify_audit_log_sync()."""
        return self.verify_audit_log_sync(deep=deep)

    def heal_audit_log_chain(self) -> Tuple[int, bool]:
        """
        Synchronously heals cryptographic hash-chain integrity of state_audit_log
        and clamps confidence values to [0.0, 1.0].
        Returns (repaired_rows_count, True).
        """
        with self._sync_lock:
            conn = self._ensure_conn()
            cols = getattr(self, "_audit_cols", set())
            has_confidence = "confidence" in cols
            with conn:
                with contextlib.closing(conn.cursor()) as cursor:
                    if has_confidence:
                        cursor.execute("SELECT seq, timestamp, key, value_hash, confidence FROM state_audit_log ORDER BY seq ASC")
                    else:
                        cursor.execute("SELECT seq, timestamp, key, value_hash FROM state_audit_log ORDER BY seq ASC")
                    rows = cursor.fetchall()

                    expected_prev = "0" * 64
                    for row in rows:
                        seq = row["seq"]
                        ts = row["timestamp"]
                        key = row["key"]
                        val_h = row["value_hash"]
                        prev_h = expected_prev
                        entry_h = self._compute_entry_hash(seq, ts, key, val_h, prev_h)
                        if has_confidence:
                            conf = clamp_conf(row["confidence"])
                            cursor.execute(
                                "UPDATE state_audit_log SET prev_hash = ?, entry_hash = ?, confidence = ? WHERE seq = ?",
                                (prev_h, entry_h, conf, seq),
                            )
                        else:
                            cursor.execute(
                                "UPDATE state_audit_log SET prev_hash = ?, entry_hash = ? WHERE seq = ?",
                                (prev_h, entry_h, seq),
                            )
                        expected_prev = entry_h

                    # Also clamp confidence in state_variables if present
                    try:
                        cursor.execute("UPDATE state_variables SET confidence = 1.0 WHERE confidence > 1.0")
                        cursor.execute("UPDATE state_variables SET confidence = 0.0 WHERE confidence < 0.0")
                    except Exception as e:
                        logger.debug("Clamping state_variables confidence warning: %s", e)

            self._prev_hash = expected_prev
            return len(rows), True

    async def heal_audit_log(self) -> Tuple[int, bool]:
        """Async alias for heal_audit_log_chain."""
        async with self._lock:
            return self.heal_audit_log_chain()

    def purge_stale_probe_keys(self, prefixes: Optional[List[str]] = None) -> int:
        """Purges test / probe keys from state_variables."""
        if prefixes is None:
            prefixes = ["huge:value", "test:adv:%", "knapsack:%"]
        with self._sync_lock:
            conn = self._ensure_conn()
            total_deleted = 0
            with conn:
                with contextlib.closing(conn.cursor()) as cursor:
                    for prefix in prefixes:
                        if "%" in prefix:
                            cursor.execute("DELETE FROM state_variables WHERE key LIKE ?", (prefix,))
                        else:
                            cursor.execute("DELETE FROM state_variables WHERE key = ?", (prefix,))
                        total_deleted += cursor.rowcount
            return total_deleted

    async def purge_stale_probes(self, prefixes: Optional[List[str]] = None) -> int:
        """Async alias for purge_stale_probe_keys."""
        async with self._lock:
            return self.purge_stale_probe_keys(prefixes=prefixes)

    async def create_transaction_intent(self, tx_id: str, record: MemoryRecord) -> None:
        """Zapisuje intencję transakcji przed wysłaniem do zewnętrznych baz (Kùzu/Qdrant)."""
        async with self._lock:
            conn = self._ensure_conn()
            with conn:
                conn.execute("""
                    INSERT INTO transaction_intent_log (tx_id, subject, predicate, object, status, timestamp, error)
                    VALUES (?, ?, ?, ?, 'PENDING', ?, NULL)
                """, (tx_id, record.effective_subject, record.predicate, str(record.object), time.time()))

    async def mark_intent_committed(self, tx_id: str) -> None:
        """Potwierdza pomyślne zakończenie zapisu we wszystkich bazach."""
        async with self._lock:
            conn = self._ensure_conn()
            with conn:
                conn.execute("UPDATE transaction_intent_log SET status = 'COMMITTED' WHERE tx_id = ?", (tx_id,))

    async def mark_intent_failed(self, tx_id: str, error: str) -> None:
        """Oznacza transakcję jako nieudaną i wymagającą kompensacji."""
        async with self._lock:
            conn = self._ensure_conn()
            with conn:
                conn.execute("UPDATE transaction_intent_log SET status = 'FAILED', error = ? WHERE tx_id = ?", (error, tx_id))

    async def get_dangling_intents(self) -> List[Dict[str, Any]]:
        """Zwraca listę transakcji, które pozostały w stanie PENDING (np. po nagłym restarcie)."""
        async with self._lock:
            conn = self._ensure_conn()
            with contextlib.closing(conn.cursor()) as cursor:
                cursor.execute("SELECT tx_id, subject, predicate, object, status, timestamp FROM transaction_intent_log WHERE status = 'PENDING'")
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

    async def set_state(
        self,
        key: str,
        value: Any,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        reason: str = "update",
    ) -> None:
        """Zapisuje stan zmiennej z audytem w bazie danych."""
        confidence = clamp_conf(confidence)
        val_json = json.dumps(value, default=str)
        meta_json = json.dumps(metadata or {}, default=str)
        now = time.time()

        async with self._lock:
            with self._sync_lock:
                conn = self._ensure_conn()
                with conn:
                    with contextlib.closing(conn.cursor()) as cursor:
                        cursor.execute("""
                            INSERT INTO state_variables (key, value, confidence, timestamp, metadata)
                            VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT(key) DO UPDATE SET
                                value = excluded.value,
                                confidence = excluded.confidence,
                                timestamp = excluded.timestamp,
                                metadata = excluded.metadata
                        """, (key, val_json, confidence, now, meta_json))

                        val_hash = hashlib.sha256(val_json.encode("utf-8")).hexdigest()
                        cursor.execute("SELECT seq, entry_hash FROM state_audit_log ORDER BY seq DESC LIMIT 1")
                        last_row = cursor.fetchone()
                        if last_row and last_row["entry_hash"]:
                            next_seq = int(last_row["seq"]) + 1
                            prev_h = str(last_row["entry_hash"])
                        else:
                            next_seq = 1
                            prev_h = "0" * 64
                        entry_hash = self._compute_entry_hash(next_seq, now, key, val_hash, prev_h)

                        self._insert_audit_entry_sync(
                            cursor, next_seq, now, key, val_hash, prev_h, entry_hash, val_json, confidence, reason
                        )
                self._prev_hash = entry_hash

    async def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            conn = self._ensure_conn()
            with contextlib.closing(conn.cursor()) as cursor:
                cursor.execute("SELECT key, value, confidence, timestamp, metadata FROM state_variables WHERE key = ?", (key,))
                row = cursor.fetchone()
                if not row:
                    cursor.execute("SELECT key, value, confidence, timestamp, metadata FROM state_variables WHERE LOWER(key) = LOWER(?) LIMIT 1", (key,))
                    row = cursor.fetchone()
                if not row:
                    return None
                return self._row_to_state_dict(row)

    async def get_states(self, keys: List[str]) -> Dict[str, Any]:
        if not keys:
            return {}

        async with self._lock:
            conn = self._ensure_conn()
            placeholders = ",".join("?" for _ in keys)
            with contextlib.closing(conn.cursor()) as cursor:
                cursor.execute(f"SELECT key, value, confidence, timestamp, metadata FROM state_variables WHERE key IN ({placeholders})", keys)
                rows = cursor.fetchall()

                result = {}
                found_keys = set()
                for row in rows:
                    k = row["key"]
                    found_keys.add(k)
                    found_keys.add(k.lower())
                    result[k] = self._row_to_state_dict(row)

                # Case-insensitive fallback for missing keys
                missing = [k for k in keys if k not in found_keys and k.lower() not in found_keys]
                if missing:
                    missing_placeholders = ",".join("?" for _ in missing)
                    cursor.execute(f"SELECT key, value, confidence, timestamp, metadata FROM state_variables WHERE LOWER(key) IN ({missing_placeholders})", [m.lower() for m in missing])
                    for row in cursor.fetchall():
                        result[row["key"]] = self._row_to_state_dict(row)

                return result

    async def get_all_states(self) -> Dict[str, Any]:
        async with self._lock:
            conn = self._ensure_conn()
            with contextlib.closing(conn.cursor()) as cursor:
                cursor.execute("SELECT key, value, confidence, timestamp, metadata FROM state_variables")
                rows = cursor.fetchall()
                result = {}
                for row in rows:
                    try:
                        result[row["key"]] = self._row_to_state_dict(row)
                    except Exception:
                        result[row["key"]] = {
                            "key": row["key"],
                            "value": str(row["value"]) if row["value"] is not None else "",
                            "confidence": float(row["confidence"]) if row["confidence"] is not None else 1.0,
                            "timestamp": float(row["timestamp"]) if row["timestamp"] is not None else 0.0,
                            "metadata": {},
                        }
                return result

    async def delete_state(self, key: str) -> bool:
        """Asynchroniczne usunięcie zmiennej stanu ze sklepu KV z wpisem do łańcucha audytu SHA-256."""
        async with self._lock:
            with self._sync_lock:
                conn = self._ensure_conn()
                with conn:
                    with contextlib.closing(conn.cursor()) as cursor:
                        cursor.execute("DELETE FROM state_variables WHERE key = ?", (key,))
                        deleted = cursor.rowcount > 0
                        if deleted:
                            # Append delete event to SHA-256 audit hash-chain (ADR-005)
                            now = time.time()
                            val_hash = hashlib.sha256(b'"__DELETED__"').hexdigest()
                            cursor.execute("SELECT seq, entry_hash FROM state_audit_log ORDER BY seq DESC LIMIT 1")
                            last_row = cursor.fetchone()
                            if last_row and last_row["entry_hash"]:
                                next_seq = int(last_row["seq"]) + 1
                                prev_h = str(last_row["entry_hash"])
                            else:
                                next_seq = 1
                                prev_h = "0" * 64
                            entry_hash = self._compute_entry_hash(next_seq, now, key, val_hash, prev_h)
                            self._insert_audit_entry_sync(cursor, next_seq, now, key, val_hash, prev_h, entry_hash, '"__DELETED__"', 0.0, "delete")
                            self._prev_hash = entry_hash
                        return deleted

    def delete_sync(self, key: str) -> bool:
        """Synchroniczne usunięcie zmiennej stanu ze sklepu KV z wpisem do łańcucha audytu SHA-256."""
        with self._sync_lock:
            conn = self._ensure_conn()
            with conn:
                with contextlib.closing(conn.cursor()) as cursor:
                    cursor.execute("DELETE FROM state_variables WHERE key = ?", (key,))
                    deleted = cursor.rowcount > 0
                    if deleted:
                        # Append delete event to SHA-256 audit hash-chain (ADR-005)
                        now = time.time()
                        val_hash = hashlib.sha256(b'"__DELETED__"').hexdigest()
                        cursor.execute("SELECT seq, entry_hash FROM state_audit_log ORDER BY seq DESC LIMIT 1")
                        last_row = cursor.fetchone()
                        if last_row and last_row["entry_hash"]:
                            next_seq = int(last_row["seq"]) + 1
                            prev_h = str(last_row["entry_hash"])
                        else:
                            next_seq = 1
                            prev_h = "0" * 64
                        entry_hash = self._compute_entry_hash(next_seq, now, key, val_hash, prev_h)
                        self._insert_audit_entry_sync(cursor, next_seq, now, key, val_hash, prev_h, entry_hash, '"__DELETED__"', 0.0, "delete")
                        self._prev_hash = entry_hash
                    return deleted

    def set_sync(
        self,
        key: str,
        value: Any,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        reason: str = "update",
    ) -> None:
        """Synchronous version of set_state for sync ingest & UDS handlers."""
        confidence = clamp_conf(confidence)
        val_json = json.dumps(value, default=str)
        meta_json = json.dumps(metadata or {}, default=str)
        now = time.time()

        with self._sync_lock:
            conn = self._ensure_conn()
            with conn:
                with contextlib.closing(conn.cursor()) as cursor:
                    cursor.execute("""
                        INSERT INTO state_variables (key, value, confidence, timestamp, metadata)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(key) DO UPDATE SET
                            value = excluded.value,
                            confidence = excluded.confidence,
                            timestamp = excluded.timestamp,
                            metadata = excluded.metadata
                    """, (key, val_json, confidence, now, meta_json))

                    val_hash = hashlib.sha256(val_json.encode("utf-8")).hexdigest()
                    cursor.execute("SELECT seq, entry_hash FROM state_audit_log ORDER BY seq DESC LIMIT 1")
                    last_row = cursor.fetchone()
                    if last_row and last_row["entry_hash"]:
                        next_seq = int(last_row["seq"]) + 1
                        prev_h = str(last_row["entry_hash"])
                    else:
                        next_seq = 1
                        prev_h = "0" * 64
                    entry_hash = self._compute_entry_hash(next_seq, now, key, val_hash, prev_h)

                    self._insert_audit_entry_sync(
                        cursor, next_seq, now, key, val_hash, prev_h, entry_hash, val_json, confidence, reason
                    )
            self._prev_hash = entry_hash

    def get_sync(self, key: str) -> Optional[Dict[str, Any]]:
        """Synchronous version of get_state with case-insensitive fallback."""
        with self._sync_lock:
            conn = self._ensure_conn()
            with contextlib.closing(conn.cursor()) as cursor:
                cursor.execute("SELECT key, value, confidence, timestamp, metadata FROM state_variables WHERE key = ?", (key,))
                row = cursor.fetchone()
                if not row:
                    cursor.execute("SELECT key, value, confidence, timestamp, metadata FROM state_variables WHERE LOWER(key) = LOWER(?) LIMIT 1", (key,))
                    row = cursor.fetchone()
                if not row:
                    return None
                return self._row_to_state_dict(row)

    def get_all_sync(self) -> Dict[str, Any]:
        """Synchronous version of get_all_states."""
        with self._sync_lock:
            conn = self._ensure_conn()
            with contextlib.closing(conn.cursor()) as cursor:
                cursor.execute("SELECT key, value, confidence, timestamp, metadata FROM state_variables")
                rows = cursor.fetchall()
                result = {}
                for row in rows:
                    try:
                        result[row["key"]] = self._row_to_state_dict(row)
                    except Exception:
                        result[row["key"]] = {
                            "key": row["key"],
                            "value": str(row["value"]) if row["value"] is not None else "",
                            "confidence": float(row["confidence"]) if row["confidence"] is not None else 1.0,
                            "timestamp": float(row["timestamp"]) if row["timestamp"] is not None else 0.0,
                            "metadata": {},
                        }
                return result

    def get_recent_sync(self, max_age_seconds: float = 900.0, limit: int = 25) -> List[Dict[str, Any]]:
        """Zwraca najświeższe zapisane stany (np. w oknie 15 minut) posortowane chronologicznie."""
        cutoff_ts = time.time() - max_age_seconds
        with self._sync_lock:
            conn = self._ensure_conn()
            with contextlib.closing(conn.cursor()) as cursor:
                cursor.execute(
                    "SELECT key, value, confidence, timestamp, metadata FROM state_variables WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT ?",
                    (cutoff_ts, limit),
                )
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    try:
                        results.append(self._row_to_state_dict(row))
                    except Exception:
                        results.append({
                            "key": row["key"],
                            "value": str(row["value"]) if row["value"] is not None else "",
                            "confidence": float(row["confidence"]) if row["confidence"] is not None else 1.0,
                            "timestamp": float(row["timestamp"]) if row["timestamp"] is not None else 0.0,
                            "metadata": {},
                        })
                results.reverse()
                return results

    # Aliases
    set = set_sync
    get = get_sync
    get_all = get_all_sync
    get_recent = get_recent_sync

    async def close(self) -> None:
        async with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
