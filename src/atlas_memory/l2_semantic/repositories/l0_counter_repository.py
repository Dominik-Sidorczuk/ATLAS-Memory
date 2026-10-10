from __future__ import annotations

import contextlib
import logging
import sqlite3
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("atlas.l2.repositories.l0_counter")

L0_STEP_COUNTER_TABLE = "l0_step_counters"
L0_TTT_COUNTER_NAME = "ttt"


class L0StepCounterRepository:
    """Repository managing monotonic step counters for L0 test-time training (TTT)."""

    @staticmethod
    def init_schema(conn: sqlite3.Connection) -> None:
        """Initializes the L0 step counter table."""
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {L0_STEP_COUNTER_TABLE} (
                name TEXT PRIMARY KEY,
                step_count INTEGER NOT NULL,
                total_energy REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

    def save_step_counter(
        self,
        conn: sqlite3.Connection,
        step_count: int,
        total_energy: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> None:
        """Monotonically saves L0-TTT step counter via MAX in SQLite UPSERT."""
        now = timestamp if timestamp is not None else time.time()
        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute(
                f"INSERT INTO {L0_STEP_COUNTER_TABLE} (name, step_count, total_energy, updated_at) VALUES (?, ?, ?, ?) "
                f"ON CONFLICT(name) DO UPDATE SET "
                f"step_count = MAX({L0_STEP_COUNTER_TABLE}.step_count, excluded.step_count), "
                f"total_energy = MAX({L0_STEP_COUNTER_TABLE}.total_energy, excluded.total_energy), "
                f"updated_at = excluded.updated_at",
                (L0_TTT_COUNTER_NAME, int(step_count), float(total_energy), now),
            )

    def load_step_counter(self, conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
        """Loads the current L0-TTT step counter state or returns None if absent."""
        try:
            with contextlib.closing(conn.cursor()) as cursor:
                cursor.execute(
                    f"SELECT step_count, total_energy, updated_at FROM {L0_STEP_COUNTER_TABLE} WHERE name = ?",
                    (L0_TTT_COUNTER_NAME,),
                )
                row = cursor.fetchone()
        except sqlite3.Error as counter_err:
            logger.debug("L0-TTT step counter read failed: %s", counter_err)
            return None
        if row is None:
            return None
        return {
            "step_count": int(row["step_count"]),
            "total_energy": float(row["total_energy"]),
            "updated_at": float(row["updated_at"]),
        }
