from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from atlas_memory.models import MemoryRecord


class VerifiedKVStore:
    """
    L2: Verified Key-Value / SQL Store (SQLite / JSONB) z obsługą transakcji Saga / 2PC
    oraz kryptograficznym SHA-256 Hash-Chain Audit Log (Source of Truth & Immutability).
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._lock = asyncio.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._prev_hash: str = "0" * 64
        self._init_db()

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            # Włączenie trybu WAL i zoptymalizowanych flag I/O dla maksymalnej współbieżności i wydajności SSD
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA temp_store=MEMORY;")
            self._conn.execute("PRAGMA cache_size=-64000;")

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
                    entry_hash TEXT NOT NULL
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

            # Inicjalizacja _prev_hash na podstawie ostatniego wpisu w bazie (jeśli istnieje)
            cursor = self._conn.cursor()
            cursor.execute("SELECT entry_hash FROM state_audit_log ORDER BY seq DESC LIMIT 1")
            row = cursor.fetchone()
            if row and row["entry_hash"]:
                self._prev_hash = row["entry_hash"]
            else:
                self._prev_hash = "0" * 64

    async def append_audit_log(self, entry: Dict[str, Any]) -> str:
        """
        Dołącza nowy wpis do kryptograficznego SHA-256 Hash-Chain.
        entry_hash = SHA-256(seq + timestamp + key + value_hash + prev_hash)
        """
        async with self._lock:
            assert self._conn is not None
            cursor = self._conn.cursor()

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

            # entry_hash = SHA-256(seq + timestamp + key + value_hash + prev_hash)
            payload_str = f"{next_seq}:{now}:{key}:{value_hash}:{prev_h}"
            entry_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

            cursor.execute("""
                INSERT INTO state_audit_log (seq, timestamp, key, value_hash, prev_hash, entry_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (next_seq, now, key, value_hash, prev_h, entry_hash))
            self._conn.commit()

            self._prev_hash = entry_hash
            return entry_hash

    async def verify_chain_integrity(self) -> Tuple[bool, int]:
        """
        Weryfikuje kryptograficzną integralność całego łańcucha audytu SHA-256.
        Zwraca (is_valid, broken_at_seq). Jeśli wszystko poprawne: (True, 0).
        """
        async with self._lock:
            assert self._conn is not None
            cursor = self._conn.cursor()
            cursor.execute("SELECT seq, timestamp, key, value_hash, prev_hash, entry_hash FROM state_audit_log ORDER BY seq ASC")
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

                payload_str = f"{seq}:{ts}:{key}:{val_h}:{prev_h}"
                computed_h = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
                if computed_h != stored_entry_h:
                    return False, seq

                expected_prev = stored_entry_h

            return True, 0


    async def create_transaction_intent(self, tx_id: str, record: MemoryRecord) -> None:
        """Zapisuje intencję transakcji przed wysłaniem do zewnętrznych baz (Kùzu/Qdrant)."""
        async with self._lock:
            assert self._conn is not None
            with self._conn:
                self._conn.execute("""
                    INSERT INTO transaction_intent_log (tx_id, subject, predicate, object, status, timestamp, error)
                    VALUES (?, ?, ?, ?, 'PENDING', ?, NULL)
                """, (tx_id, record.effective_subject, record.predicate, str(record.object), time.time()))

    async def mark_intent_committed(self, tx_id: str) -> None:
        """Potwierdza pomyślne zakończenie zapisu we wszystkich bazach."""
        async with self._lock:
            assert self._conn is not None
            with self._conn:
                self._conn.execute("UPDATE transaction_intent_log SET status = 'COMMITTED' WHERE tx_id = ?", (tx_id,))

    async def mark_intent_failed(self, tx_id: str, error: str) -> None:
        """Oznacza transakcję jako nieudaną i wymagającą kompensacji."""
        async with self._lock:
            assert self._conn is not None
            with self._conn:
                self._conn.execute("UPDATE transaction_intent_log SET status = 'FAILED', error = ? WHERE tx_id = ?", (error, tx_id))

    async def get_dangling_intents(self) -> List[Dict[str, Any]]:
        """Zwraca listę transakcji, które pozostały w stanie PENDING (np. po nagłym restarcie)."""
        async with self._lock:
            assert self._conn is not None
            cursor = self._conn.cursor()
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
        val_json = json.dumps(value, default=str)
        meta_json = json.dumps(metadata or {}, default=str)
        now = time.time()

        async with self._lock:
            assert self._conn is not None
            cursor = self._conn.cursor()
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
            cursor.execute("SELECT IFNULL(MAX(seq), 0) + 1 AS next_seq FROM state_audit_log")
            next_seq = cursor.fetchone()["next_seq"]
            prev_h = self._prev_hash
            payload_str = f"{next_seq}:{now}:{key}:{val_hash}:{prev_h}"
            entry_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

            cursor.execute("""
                INSERT INTO state_audit_log (seq, timestamp, key, value_hash, prev_hash, entry_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (next_seq, now, key, val_hash, prev_h, entry_hash))

            self._prev_hash = entry_hash
            self._conn.commit()

    async def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            assert self._conn is not None
            cursor = self._conn.cursor()
            cursor.execute("SELECT key, value, confidence, timestamp, metadata FROM state_variables WHERE key = ?", (key,))
            row = cursor.fetchone()
            if not row:
                return None

            return {
                "key": row["key"],
                "value": json.loads(row["value"]),
                "confidence": row["confidence"],
                "timestamp": row["timestamp"],
                "metadata": json.loads(row["metadata"] or "{}"),
            }

    async def get_states(self, keys: List[str]) -> Dict[str, Any]:
        if not keys:
            return {}

        async with self._lock:
            assert self._conn is not None
            placeholders = ",".join("?" for _ in keys)
            cursor = self._conn.cursor()
            cursor.execute(f"SELECT key, value, confidence, timestamp, metadata FROM state_variables WHERE key IN ({placeholders})", keys)
            rows = cursor.fetchall()

            result = {}
            for row in rows:
                result[row["key"]] = {
                    "value": json.loads(row["value"]),
                    "confidence": row["confidence"],
                    "timestamp": row["timestamp"],
                    "metadata": json.loads(row["metadata"] or "{}"),
                }
            return result

    async def get_all_states(self) -> Dict[str, Any]:
        async with self._lock:
            assert self._conn is not None
            cursor = self._conn.cursor()
            cursor.execute("SELECT key, value, confidence, timestamp, metadata FROM state_variables")
            rows = cursor.fetchall()
            return {
                row["key"]: {
                    "value": json.loads(row["value"]),
                    "confidence": row["confidence"],
                    "timestamp": row["timestamp"],
                }
                for row in rows
            }

    async def close(self) -> None:
        async with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
