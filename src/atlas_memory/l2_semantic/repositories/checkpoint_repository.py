from __future__ import annotations

import contextlib
import logging
import sqlite3
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("atlas.l2.repositories.checkpoint")


class SleepCheckpointRepository:
    """Repository managing sleep consolidation checkpoints."""

    @staticmethod
    def init_schema(conn: sqlite3.Connection) -> None:
        """Initializes the sleep consolidation checkpoints table."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sleep_consolidation_checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                last_processed_step_id INTEGER NOT NULL,
                sessions_consolidated INTEGER NOT NULL,
                sops_baked INTEGER NOT NULL,
                facts_superseded INTEGER NOT NULL,
                duration_ms REAL NOT NULL
            )
        """)

    def record_checkpoint(
        self,
        conn: sqlite3.Connection,
        checkpoint_id: str,
        last_processed_step_id: int,
        sessions_consolidated: int = 0,
        sops_baked: int = 0,
        facts_superseded: int = 0,
        duration_ms: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Records a single sleep consolidation cycle checkpoint. Returns True if created, False if existing."""
        if not checkpoint_id:
            raise ValueError("checkpoint_id is required for sleep checkpoint record")

        now = timestamp if timestamp is not None else time.time()
        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute(
                "SELECT 1 FROM sleep_consolidation_checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,),
            )
            exists = cursor.fetchone() is not None
            if exists:
                return False

            cursor.execute("""
                INSERT INTO sleep_consolidation_checkpoints (
                    checkpoint_id, timestamp, last_processed_step_id,
                    sessions_consolidated, sops_baked, facts_superseded, duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                checkpoint_id,
                now,
                int(last_processed_step_id),
                int(sessions_consolidated),
                int(sops_baked),
                int(facts_superseded),
                float(duration_ms),
            ))
            return True

    def get_checkpoint(self, conn: sqlite3.Connection, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single sleep checkpoint by ID."""
        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute(
                "SELECT * FROM sleep_consolidation_checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_checkpoints(self, conn: sqlite3.Connection, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists recent sleep consolidation checkpoints ordered newest first."""
        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute(
                "SELECT * FROM sleep_consolidation_checkpoints ORDER BY timestamp DESC LIMIT ?",
                (int(max(1, min(limit, 5000))),),
            )
            return [dict(row) for row in cursor.fetchall()]

    def count_checkpoints(self, conn: sqlite3.Connection) -> int:
        """Returns the total number of recorded sleep consolidation cycles."""
        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute("SELECT COUNT(*) AS n FROM sleep_consolidation_checkpoints")
            row = cursor.fetchone()
            return int(row["n"]) if row else 0
