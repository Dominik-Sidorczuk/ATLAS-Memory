from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("atlas.l2.repositories.connection")


class SqliteConnectionManager:
    """Manages SQLite connection lifecycle, WAL pragmas, cross-db attachment, and backup."""

    @staticmethod
    def connect(
        db_path: str,
        audit_db_path_override: Optional[str] = None,
    ) -> Tuple[sqlite3.Connection, Optional[str], str]:
        """Creates and configures the SQLite connection with WAL pragmas and optional attached audit DB."""
        conn = sqlite3.connect(db_path, check_same_thread=False, autocommit=True)
        conn.row_factory = sqlite3.Row

        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA cache_size=-64000;")

        if db_path != ":memory:":
            if audit_db_path_override:
                audit_db_path = str(audit_db_path_override)
            else:
                base_stem = Path(db_path).stem
                audit_db_path = str(Path(db_path).parent / f"{base_stem}_audit.db")

            conn.execute(f"ATTACH DATABASE '{audit_db_path}' AS audit;")
            conn.execute("PRAGMA audit.journal_mode=WAL;")
            conn.execute("PRAGMA audit.synchronous=NORMAL;")
            conn.execute("PRAGMA audit.busy_timeout=5000;")
            target_audit = "audit."
        else:
            audit_db_path = ":memory:"
            target_audit = ""

        return conn, audit_db_path, target_audit

    @staticmethod
    def checkpoint_wal(
        conn: Optional[sqlite3.Connection],
        db_path: str,
        audit_db_path: Optional[str],
        truncate: bool = True,
    ) -> Dict[str, Any]:
        """Performs a WAL checkpoint on main and attached audit databases."""
        mode = "TRUNCATE" if truncate else "PASSIVE"
        res: Dict[str, Any] = {}
        if conn is not None and db_path != ":memory:":
            try:
                cur = conn.cursor()
                r_main = cur.execute(f"PRAGMA main.wal_checkpoint({mode});").fetchone()
                res["main"] = tuple(r_main) if r_main else ()
                if audit_db_path and audit_db_path != ":memory:":
                    r_audit = cur.execute(f"PRAGMA audit.wal_checkpoint({mode});").fetchone()
                    res["audit"] = tuple(r_audit) if r_audit else ()
                cur.close()
            except Exception as exc:
                logger.warning("wal_checkpoint failed: %s", exc)
                res["error"] = str(exc)
        return res

    @staticmethod
    def backup_audit_db(
        conn: Optional[sqlite3.Connection],
        audit_db_path: Optional[str],
        dest_path: Path | str,
        checkpoint_fn: Any,
    ) -> Path:
        """Executes a live consistent snapshot of the audit database using SQLite Online Backup API."""
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_fn(truncate=True)

        if audit_db_path == ":memory:":
            if conn is not None:
                dest_conn = sqlite3.connect(str(dest))
                try:
                    conn.backup(dest_conn)
                finally:
                    dest_conn.close()
        else:
            if audit_db_path is not None:
                src_conn = sqlite3.connect(audit_db_path)
                dest_conn = sqlite3.connect(str(dest))
                try:
                    src_conn.backup(dest_conn)
                finally:
                    dest_conn.close()
                    src_conn.close()
        return dest

    @staticmethod
    def close(
        conn: Optional[sqlite3.Connection],
        db_path: str,
        audit_db_path: Optional[str],
        checkpoint_fn: Any,
    ) -> None:
        """Closes the connection after performing a TRUNCATE checkpoint."""
        if conn is not None:
            try:
                if db_path != ":memory:":
                    checkpoint_fn(truncate=True)
            except Exception as exc:
                logger.warning("Error during WAL checkpoint on close: %s", exc)
            conn.close()
