from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import sqlite3
import time
from typing import Any, Dict, Optional, Set, Tuple

logger = logging.getLogger("atlas.l2.repositories.audit")

REDACTION_MARKER_PREFIX = "[REDACTED-"
REDACTION_REGISTRY_TABLE = "audit_redactions"
REDACTION_LEGACY_REASON = "grandfathered-pre-registry-v87"


class Sha256AuditLedgerRepository:
    """Repository managing the SHA-256 hash-chain audit ledger and authorized redactions."""

    @staticmethod
    def compute_entry_hash(seq: int, timestamp: float, key: str, value_hash: str, prev_hash: str) -> str:
        """Deterministic computation of SHA-256 hash-chain node hash."""
        payload_str = f"{seq}:{timestamp}:{key}:{value_hash}:{prev_hash}"
        return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

    def init_schema(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        audit_db_path: Optional[str],
        target_audit: str = "",
    ) -> Tuple[Set[str], str]:
        """Initializes audit ledger tables, migrations, and returns available columns and tip hash."""
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {target_audit}state_audit_log (
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

        with contextlib.closing(conn.cursor()) as registry_probe:
            registry_probe.execute(
                f"SELECT name FROM {target_audit}sqlite_master WHERE type='table' AND name='{REDACTION_REGISTRY_TABLE}'"
            )
            registry_preexisting = registry_probe.fetchone() is not None

        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {target_audit}{REDACTION_REGISTRY_TABLE} (
                seq INTEGER PRIMARY KEY,
                key TEXT NOT NULL,
                value_hash TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)

        if not registry_preexisting:
            with contextlib.closing(conn.execute(f"""
                INSERT OR IGNORE INTO {REDACTION_REGISTRY_TABLE} (seq, key, value_hash, reason, created_at)
                SELECT seq, key, value_hash, ?, COALESCE(timestamp, 0.0)
                FROM state_audit_log
                WHERE new_value LIKE '{REDACTION_MARKER_PREFIX}%'
            """, (REDACTION_LEGACY_REASON,))) as legacy_probe:
                if legacy_probe.rowcount:
                    logger.info("V87: registered %d legacy redactions (reason=%s)", legacy_probe.rowcount, REDACTION_LEGACY_REASON)

        # Migration if state_audit_log was previously in main
        if db_path != ":memory:":
            with contextlib.closing(conn.cursor()) as cur:
                cur.execute("SELECT name FROM main.sqlite_master WHERE type='table' AND name='state_audit_log'")
                if cur.fetchone() is not None:
                    cur.execute("SELECT count(*) AS c FROM audit.state_audit_log")
                    if cur.fetchone()["c"] == 0:
                        cur.execute("INSERT INTO audit.state_audit_log SELECT * FROM main.state_audit_log")
                        logger.info("Migrated state_audit_log from %s to %s", db_path, audit_db_path)
                    cur.execute("DROP TABLE main.state_audit_log")

        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute("PRAGMA table_info(state_audit_log)")
            audit_cols = {row["name"] for row in cursor.fetchall()}
            for col_name, col_def in (
                ("new_value", "TEXT"),
                ("confidence", "REAL DEFAULT 1.0"),
                ("reason", "TEXT DEFAULT 'update'"),
            ):
                if col_name not in audit_cols:
                    try:
                        cursor.execute(f"ALTER TABLE {target_audit}state_audit_log ADD COLUMN {col_name} {col_def}")
                        audit_cols.add(col_name)
                    except Exception as col_err:
                        logger.debug("Column %s already added or alter failed: %s", col_name, col_err)

            cursor.execute("SELECT entry_hash FROM state_audit_log ORDER BY seq DESC LIMIT 1")
            row = cursor.fetchone()
            if row and row["entry_hash"]:
                prev_hash = str(row["entry_hash"])
            else:
                prev_hash = "0" * 64

        return audit_cols, prev_hash

    def get_last_entry(self, conn: sqlite3.Connection) -> Tuple[int, str]:
        """Returns the next sequence number and current tip hash directly from the database."""
        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute("SELECT seq, entry_hash FROM state_audit_log ORDER BY seq DESC LIMIT 1")
            last_row = cursor.fetchone()
            if last_row and last_row["entry_hash"]:
                return int(last_row["seq"]) + 1, str(last_row["entry_hash"])
            return 1, "0" * 64

    def insert_audit_entry(
        self,
        conn: sqlite3.Connection | sqlite3.Cursor,
        audit_cols: Set[str],
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
        """Inserts a single entry into state_audit_log."""
        if isinstance(conn, sqlite3.Cursor):
            cursor = conn
            if "new_value" in audit_cols:
                cursor.execute("""
                    INSERT INTO state_audit_log (seq, timestamp, key, value_hash, prev_hash, entry_hash, new_value, confidence, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (seq, timestamp, key, val_hash, prev_h, entry_hash, val_json, confidence, reason))
            else:
                cursor.execute("""
                    INSERT INTO state_audit_log (seq, timestamp, key, value_hash, prev_hash, entry_hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (seq, timestamp, key, val_hash, prev_h, entry_hash))
        else:
            with contextlib.closing(conn.cursor()) as cursor:
                if "new_value" in audit_cols:
                    cursor.execute("""
                        INSERT INTO state_audit_log (seq, timestamp, key, value_hash, prev_hash, entry_hash, new_value, confidence, reason)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (seq, timestamp, key, val_hash, prev_h, entry_hash, val_json, confidence, reason))
                else:
                    cursor.execute("""
                        INSERT INTO state_audit_log (seq, timestamp, key, value_hash, prev_hash, entry_hash)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (seq, timestamp, key, val_hash, prev_h, entry_hash))

    def append_audit_log(
        self,
        conn: sqlite3.Connection,
        audit_cols: Set[str],
        entry: Dict[str, Any],
    ) -> str:
        """Appends a new entry to the cryptographic SHA-256 hash-chain."""
        next_seq, prev_h = self.get_last_entry(conn)
        now = float(entry.get("timestamp", time.time()))
        key = str(entry.get("key", ""))
        raw_val = entry.get("value")

        if "value_hash" in entry:
            value_hash = str(entry["value_hash"])
        else:
            val_bytes = json.dumps(raw_val, sort_keys=True).encode("utf-8") if raw_val is not None else b""
            value_hash = hashlib.sha256(val_bytes).hexdigest()

        entry_hash = self.compute_entry_hash(next_seq, now, key, value_hash, prev_h)

        if raw_val is None:
            val_json = ""
        else:
            val_json = json.dumps(raw_val, sort_keys=True)

        self.insert_audit_entry(
            conn=conn,
            audit_cols=audit_cols,
            seq=next_seq,
            timestamp=now,
            key=key,
            val_hash=value_hash,
            prev_h=prev_h,
            entry_hash=entry_hash,
            val_json=val_json,
            confidence=float(entry.get("confidence", 1.0)),
            reason=str(entry.get("reason", "append_audit")),
        )
        return entry_hash

    def redaction_is_authorized(self, conn: sqlite3.Connection, seq: int, key: str, value_hash: str) -> bool:
        """Verifies if redaction has an entry in audit_redactions (fail-closed)."""
        try:
            with contextlib.closing(conn.cursor()) as cursor:
                cursor.execute(
                    f"SELECT 1 FROM {REDACTION_REGISTRY_TABLE} WHERE seq = ? AND key = ? AND value_hash = ? LIMIT 1",
                    (seq, key, value_hash),
                )
                return cursor.fetchone() is not None
        except sqlite3.Error as reg_err:
            logger.warning("V87: redaction registry inaccessible (%s) — seq=%s rejected", reg_err, seq)
            return False

    def authorize_redaction(
        self,
        conn: sqlite3.Connection,
        seq: int,
        key: str,
        value_hash: str,
        reason: str = REDACTION_LEGACY_REASON,
    ) -> Dict[str, Any]:
        """Registers a redaction entry in audit_redactions."""
        if not str(reason).strip():
            return {"status": "error", "error": "empty_reason", "seq": seq}

        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute(
                f"INSERT OR IGNORE INTO {REDACTION_REGISTRY_TABLE} (seq, key, value_hash, reason, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (seq, key, value_hash, str(reason), time.time()),
            )
        logger.info("V87/T1: registered redaction seq=%s key=%s (reason=%s)", seq, key, reason)
        return {"status": "ok", "seq": seq, "registered": True}

    def redact_audit_entry(
        self,
        conn: sqlite3.Connection,
        seq: int,
        reason: str = "SEC-001",
        redacted_value: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Atomically replaces new_value with redaction marker and registers the redaction."""
        marker = redacted_value if redacted_value is not None else f"{REDACTION_MARKER_PREFIX}{reason} {time.strftime('%Y-%m-%d')}"
        if not marker.startswith(REDACTION_MARKER_PREFIX):
            return {"status": "error", "error": "invalid_redaction_marker", "seq": seq}
        if not str(reason).strip():
            return {"status": "error", "error": "empty_reason", "seq": seq}

        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute("SELECT seq, key, value_hash FROM state_audit_log WHERE seq = ?", (seq,))
            row = cursor.fetchone()
        if row is None:
            return {"status": "error", "error": "audit_entry_not_found", "seq": seq}

        with conn:
            conn.execute("UPDATE state_audit_log SET new_value = ? WHERE seq = ?", (marker, seq))
            conn.execute(
                f"INSERT OR REPLACE INTO {REDACTION_REGISTRY_TABLE} (seq, key, value_hash, reason, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (int(row["seq"]), row["key"], row["value_hash"], str(reason), time.time()),
            )
        logger.info("V87: authorized redaction for seq=%s key=%s (reason=%s)", seq, row["key"], reason)
        return {
            "status": "ok",
            "seq": int(row["seq"]),
            "key": row["key"],
            "value_hash": row["value_hash"],
            "reason": str(reason),
            "redacted_value": marker,
        }

    def verify_audit_log(
        self,
        conn: sqlite3.Connection,
        audit_cols: Set[str],
        deep: bool = True,
    ) -> Tuple[bool, int, str]:
        """Verify SHA-256 hash-chain integrity and optional F-10 cross-table consistency."""
        with contextlib.closing(conn.cursor()) as cursor:
            if "new_value" in audit_cols:
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
                return False, seq, expected_prev

            if deep and "new_value" in audit_cols and row["new_value"] is not None:
                nv_raw = str(row["new_value"])
                calc_val_h = hashlib.sha256(nv_raw.encode("utf-8")).hexdigest()
                if calc_val_h != val_h:
                    is_legacy_tombstone = (
                        nv_raw.strip('"') == "__DELETED__"
                        and val_h
                        in (
                            hashlib.sha256(b"__DELETED__").hexdigest(),
                            hashlib.sha256(b'"__DELETED__"').hexdigest(),
                        )
                    )
                    is_security_redaction = nv_raw.startswith(REDACTION_MARKER_PREFIX)
                    if not is_legacy_tombstone:
                        if not (
                            is_security_redaction
                            and self.redaction_is_authorized(conn, seq, key, val_h)
                        ):
                            return False, seq, expected_prev

            computed_h = self.compute_entry_hash(seq, ts, key, val_h, prev_h)
            if computed_h != stored_entry_h:
                return False, seq, expected_prev

            expected_prev = stored_entry_h

        if deep:
            with contextlib.closing(conn.cursor()) as cursor:
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='state_variables'"
                )
                if cursor.fetchone():
                    cursor.execute("PRAGMA table_info(state_variables)")
                    sv_cols = {r["name"] for r in cursor.fetchall()}
                    if "value_hash" in sv_cols:
                        cursor.execute(
                            "SELECT key, value, value_hash FROM state_variables"
                        )
                        sv_rows = cursor.fetchall()
                        for sv in sv_rows:
                            k = sv["key"]
                            v_raw = str(sv["value"]) if sv["value"] is not None else ""
                            sv_h = sv["value_hash"]
                            if sv_h:
                                if (
                                    hashlib.sha256(v_raw.encode("utf-8")).hexdigest()
                                    != sv_h
                                ):
                                    logger.warning(
                                        "Audit deep-check: state_variables value mismatch for key %s",
                                        k,
                                    )
                                    return False, -1, expected_prev

                                cursor.execute(
                                    "SELECT seq, value_hash, reason FROM state_audit_log "
                                    "WHERE key = ? ORDER BY seq DESC LIMIT 1",
                                    (k,),
                                )
                                latest_audit = cursor.fetchone()
                                if latest_audit:
                                    if latest_audit["reason"] == "delete":
                                        logger.warning(
                                            "Audit deep-check: key %s exists in state_variables "
                                            "but deleted in audit log",
                                            k,
                                        )
                                        return False, latest_audit["seq"], expected_prev
                                    if latest_audit["value_hash"] != sv_h:
                                        logger.warning(
                                            "Audit deep-check: state_variables hash != latest "
                                            "audit log hash for key %s",
                                            k,
                                        )
                                        return False, latest_audit["seq"], expected_prev

        return True, 0, expected_prev

    def heal_audit_log_chain(
        self,
        conn: sqlite3.Connection,
        audit_cols: Set[str],
    ) -> Tuple[int, bool, str]:
        """Synchronously heal SHA-256 hash-chain integrity and clamp confidence values."""
        has_confidence = "confidence" in audit_cols
        with conn:
            with contextlib.closing(conn.cursor()) as cursor:
                if has_confidence:
                    cursor.execute(
                        "SELECT seq, timestamp, key, value_hash, confidence "
                        "FROM state_audit_log ORDER BY seq ASC"
                    )
                else:
                    cursor.execute(
                        "SELECT seq, timestamp, key, value_hash "
                        "FROM state_audit_log ORDER BY seq ASC"
                    )
                rows = cursor.fetchall()

                expected_prev = "0" * 64
                for row in rows:
                    seq = row["seq"]
                    ts = row["timestamp"]
                    key = row["key"]
                    val_h = row["value_hash"]
                    prev_h = expected_prev
                    entry_h = self.compute_entry_hash(seq, ts, key, val_h, prev_h)
                    if has_confidence:
                        from atlas_memory.cognitive.models import clamp_conf
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

                try:
                    cursor.execute(
                        "UPDATE state_variables SET confidence = 1.0 WHERE confidence > 1.0"
                    )
                    cursor.execute(
                        "UPDATE state_variables SET confidence = 0.0 WHERE confidence < 0.0"
                    )
                except Exception as e:
                    logger.debug("Clamping state_variables confidence warning: %s", e)

        return len(rows), True, expected_prev


    def count_audit_entries(self, conn: sqlite3.Connection) -> int:
        """Returns the total number of entries in state_audit_log."""
        with contextlib.closing(conn.cursor()) as cursor:
            try:
                cursor.execute("SELECT COUNT(*) FROM state_audit_log")
            except Exception:
                cursor.execute("SELECT COUNT(*) FROM audit.state_audit_log")
            row = cursor.fetchone()
            return int(row[0]) if row else 0
