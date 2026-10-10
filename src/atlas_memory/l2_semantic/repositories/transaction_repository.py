from __future__ import annotations

import contextlib
import logging
import sqlite3
import time
from typing import Any, Dict, List

from atlas_memory.core.models import MemoryRecord

logger = logging.getLogger("atlas.l2.repositories.transaction")


class TransactionIntentRepository:
    """Repository managing distributed transaction intent logs (Saga / 2PC pattern)."""

    @staticmethod
    def init_schema(conn: sqlite3.Connection) -> None:
        """Initializes the transaction intent log table."""
        conn.execute("""
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

    def create_intent(self, conn: sqlite3.Connection, tx_id: str, record: MemoryRecord) -> None:
        """Records a transaction intent in PENDING state prior to external sync."""
        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute("""
                INSERT INTO transaction_intent_log (tx_id, subject, predicate, object, status, timestamp, error)
                VALUES (?, ?, ?, ?, 'PENDING', ?, NULL)
            """, (tx_id, record.effective_subject, record.predicate, str(record.object), time.time()))

    def mark_committed(self, conn: sqlite3.Connection, tx_id: str) -> None:
        """Confirms successful transaction completion."""
        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute("UPDATE transaction_intent_log SET status = 'COMMITTED' WHERE tx_id = ?", (tx_id,))

    def mark_failed(self, conn: sqlite3.Connection, tx_id: str, error: str) -> None:
        """Marks a transaction as failed with error details."""
        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute("UPDATE transaction_intent_log SET status = 'FAILED', error = ? WHERE tx_id = ?", (error, tx_id))

    def get_dangling_intents(self, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        """Returns transactions still in PENDING state."""
        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute(
                "SELECT tx_id, subject, predicate, object, status, timestamp FROM transaction_intent_log WHERE status = 'PENDING'"
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
