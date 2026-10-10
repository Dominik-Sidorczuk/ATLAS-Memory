from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Sequence, Set

from atlas_memory.l2_semantic.audit_ledger import record_swallow
from atlas_memory.l2_semantic.textnorm import normalize_stored_text

logger = logging.getLogger("atlas.l2.repositories.state")


class SqliteStateRepository:
    """Repository handling state variables, full-text search (FTS5), and state queries."""

    @staticmethod
    def init_schema(conn: sqlite3.Connection) -> None:
        """Initializes tables, indexes, FTS5 virtual table, and sync triggers for state variables."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS state_variables (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT,
                value_hash TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_state_variables_timestamp ON state_variables (timestamp DESC);")
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS state_variables_fts USING fts5(
                key,
                value,
                tokenize='unicode61'
            )
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS state_variables_ai AFTER INSERT ON state_variables BEGIN
                INSERT INTO state_variables_fts(key, value) VALUES (new.key, new.value);
            END;
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS state_variables_ad AFTER DELETE ON state_variables BEGIN
                DELETE FROM state_variables_fts WHERE key = old.key;
            END;
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS state_variables_au AFTER UPDATE ON state_variables BEGIN
                DELETE FROM state_variables_fts WHERE key = old.key;
                INSERT INTO state_variables_fts(key, value) VALUES (new.key, new.value);
            END;
        """)

        try:
            with contextlib.closing(conn.cursor()) as cur_fts:
                cur_fts.execute("SELECT count(*) AS c FROM state_variables_fts")
                if cur_fts.fetchone()["c"] == 0:
                    cur_fts.execute("SELECT count(*) AS c FROM state_variables")
                    if cur_fts.fetchone()["c"] > 0:
                        conn.execute("INSERT INTO state_variables_fts(key, value) SELECT key, value FROM state_variables")
        except Exception as fts_init_err:
            logger.debug("FTS5 sync check error: %s", fts_init_err)

        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute("PRAGMA table_info(state_variables)")
            sv_cols = {row["name"] for row in cursor.fetchall()}
            if "value_hash" not in sv_cols:
                try:
                    cursor.execute("ALTER TABLE state_variables ADD COLUMN value_hash TEXT")
                except Exception as col_err:
                    logger.debug("Failed adding value_hash to state_variables: %s", col_err)
            cursor.execute("SELECT key, value FROM state_variables WHERE value_hash IS NULL")
            for r in cursor.fetchall():
                v_raw = str(r["value"]) if r["value"] is not None else ""
                vh = hashlib.sha256(v_raw.encode("utf-8")).hexdigest()
                cursor.execute("UPDATE state_variables SET value_hash = ? WHERE key = ?", (vh, r["key"]))

    @staticmethod
    def row_to_state_dict(row: sqlite3.Row, json_loads: Optional[Any] = None) -> Dict[str, Any]:
        """Converts a state_variables sqlite3.Row to a unified dictionary with tamper detection."""
        val_raw = row["value"]
        val_raw_str = str(val_raw) if val_raw is not None else ""
        row_keys = row.keys() if hasattr(row, "keys") else []
        stored_hash = row["value_hash"] if "value_hash" in row_keys else None

        tampered = False
        if stored_hash:
            computed_h = hashlib.sha256(val_raw_str.encode("utf-8")).hexdigest()
            if computed_h != stored_hash:
                tampered = True
                logger.warning(
                    "Tamper detected for key '%s': stored_hash=%s != computed_hash=%s",
                    row["key"], stored_hash, computed_h,
                )

        _loads = json_loads or json.loads
        val = val_raw
        if isinstance(val, str):
            val = normalize_stored_text(val)
            try:
                val = _loads(val)
            except json.JSONDecodeError:
                # Plain-text value is not valid JSON by contract; fallback without error.
                logger.debug("Plain text value fallback for key '%s'", row["key"])
            except Exception as exc:
                record_swallow("l2_semantic.kv_store._row_to_state_dict", str(row["key"]), exc)
        if isinstance(val, str):
            val = normalize_stored_text(val)

        meta = row["metadata"]
        if isinstance(meta, str):
            try:
                parsed_meta = _loads(normalize_stored_text(meta)) if meta else {}
                meta = parsed_meta if isinstance(parsed_meta, dict) else {}
            except json.JSONDecodeError:
                meta = {}
            except Exception as exc:
                record_swallow("l2_semantic.kv_store._row_to_state_dict.meta", str(row["key"]), exc)
                meta = {}
        elif not isinstance(meta, dict):
            meta = {}

        state_dict: Dict[str, Any] = {
            "key": row["key"],
            "value": val,
            "confidence": row["confidence"],
            "timestamp": row["timestamp"],
            "metadata": meta,
        }
        if stored_hash is not None:
            state_dict["value_hash"] = stored_hash
        if tampered:
            state_dict["tampered"] = True
            meta["tampered"] = True

        return state_dict

    def set_state(
        self,
        conn: sqlite3.Connection,
        key: str,
        val_json: str,
        confidence: float,
        timestamp: float,
        meta_json: str,
        val_hash: str,
    ) -> None:
        """Upserts a variable into state_variables."""
        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute("""
                INSERT INTO state_variables (key, value, confidence, timestamp, metadata, value_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    confidence = excluded.confidence,
                    timestamp = excluded.timestamp,
                    metadata = excluded.metadata,
                    value_hash = excluded.value_hash
            """, (key, val_json, confidence, timestamp, meta_json, val_hash))

    def get_state(self, conn: sqlite3.Connection, key: str) -> Optional[Dict[str, Any]]:
        """Retrieves a state variable by exact key with case-insensitive fallback."""
        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute(
                "SELECT key, value, confidence, timestamp, metadata, value_hash FROM state_variables WHERE key = ?",
                (key,),
            )
            row = cursor.fetchone()
            if not row:
                cursor.execute(
                    "SELECT key, value, confidence, timestamp, metadata, value_hash FROM state_variables WHERE LOWER(key) = LOWER(?) LIMIT 1",
                    (key,),
                )
                row = cursor.fetchone()
            if not row:
                return None
            return self.row_to_state_dict(row)

    def get_states(self, conn: sqlite3.Connection, keys: List[str]) -> Dict[str, Any]:
        """Retrieves multiple state variables by keys with case-insensitive fallback."""
        if not keys:
            return {}

        placeholders = ",".join("?" for _ in keys)
        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute(
                f"SELECT key, value, confidence, timestamp, metadata, value_hash FROM state_variables WHERE key IN ({placeholders})",
                keys,
            )
            rows = cursor.fetchall()

            result: Dict[str, Any] = {}
            found_keys: Set[str] = set()
            for row in rows:
                k = row["key"]
                found_keys.add(k)
                found_keys.add(k.lower())
                result[k] = self.row_to_state_dict(row)

            # Case-insensitive fallback for missing keys
            missing = [k for k in keys if k not in found_keys and k.lower() not in found_keys]
            if missing:
                missing_placeholders = ",".join("?" for _ in missing)
                cursor.execute(
                    f"SELECT key, value, confidence, timestamp, metadata, value_hash FROM state_variables WHERE LOWER(key) IN ({missing_placeholders})",
                    [m.lower() for m in missing],
                )
                for row in cursor.fetchall():
                    result[row["key"]] = self.row_to_state_dict(row)

            return result

    def delete_state(self, conn: sqlite3.Connection, key: str) -> bool:
        """Deletes a state variable from state_variables. Returns True if deleted."""
        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute("DELETE FROM state_variables WHERE key = ?", (key,))
            return cursor.rowcount > 0

    def count(self, conn: sqlite3.Connection) -> int:
        """Returns the total number of records in state_variables (O(1))."""
        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute("SELECT count(*) FROM state_variables")
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def filter_existing_keys(self, conn: sqlite3.Connection, keys: Sequence[str]) -> Set[str]:
        """Returns the subset of keys that exist in state_variables."""
        if not keys:
            return set()
        clean_keys = [str(k) for k in keys if k]
        if not clean_keys:
            return set()

        found: Set[str] = set()
        for i in range(0, len(clean_keys), 500):
            chunk = clean_keys[i : i + 500]
            placeholders = ",".join("?" for _ in chunk)
            with contextlib.closing(conn.cursor()) as cursor:
                cursor.execute(f"SELECT key FROM state_variables WHERE key IN ({placeholders})", chunk)
                for row in cursor.fetchall():
                    found.add(row["key"] if isinstance(row, sqlite3.Row) else row[0])
            missing = [k for k in chunk if k not in found and k.lower() not in {x.lower() for x in found}]
            if missing:
                missing_placeholders = ",".join("?" for _ in missing)
                with contextlib.closing(conn.cursor()) as cursor:
                    cursor.execute(
                        f"SELECT key FROM state_variables WHERE LOWER(key) IN ({missing_placeholders})",
                        [m.lower() for m in missing],
                    )
                    for row in cursor.fetchall():
                        found.add(row["key"] if isinstance(row, sqlite3.Row) else row[0])
        return found

    def search_candidates(
        self,
        conn: sqlite3.Connection,
        query: str = "",
        limit: int = 100,
        words: Optional[List[str]] = None,
        stems: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Searches candidates using PK, prefix, and FTS5 full-text indexing."""
        if not query and not words and not stems:
            return {}

        cap = max(1, min(int(limit), 500))
        results: Dict[str, Dict[str, Any]] = {}
        with contextlib.closing(conn.cursor()) as cursor:
            q_clean = query.strip()
            if q_clean:
                # 1. Exact key match
                cursor.execute(
                    "SELECT key, value, confidence, timestamp, metadata, value_hash FROM state_variables WHERE key = ?",
                    (q_clean,),
                )
                row = cursor.fetchone()
                if row:
                    results[row["key"]] = self.row_to_state_dict(row)

                # 2. Prefix or namespace match
                if len(q_clean) >= 3 and ":" in q_clean:
                    cursor.execute(
                        "SELECT key, value, confidence, timestamp, metadata, value_hash FROM state_variables WHERE key LIKE ? LIMIT ?",
                        (f"{q_clean}%", cap),
                    )
                    for r in cursor.fetchall():
                        if r["key"] not in results:
                            results[r["key"]] = self.row_to_state_dict(r)

            # 3. FTS5 full-text search
            search_tokens: List[str] = []
            if words:
                search_tokens.extend([w for w in words if len(w) >= 3 and not any(c in w for c in '":*^()')])
            if stems:
                search_tokens.extend([s for s in stems if len(s) >= 4 and not any(c in s for c in '":*^()')])
            if not search_tokens and q_clean:
                search_tokens = [w for w in re.findall(r"\w+", q_clean) if len(w) >= 3 and not any(c in w for c in '":*^()')]

            if search_tokens and len(results) < cap:
                fts_query = " OR ".join(f'"{t}"*' for t in search_tokens[:8])
                try:
                    cursor.execute(
                        """
                        SELECT s.key, s.value, s.confidence, s.timestamp, s.metadata, s.value_hash
                        FROM state_variables s
                        JOIN state_variables_fts f ON s.key = f.key
                        WHERE state_variables_fts MATCH ?
                        LIMIT ?
                        """,
                        (fts_query, cap),
                    )
                    for r in cursor.fetchall():
                        if r["key"] not in results and len(results) < cap:
                            results[r["key"]] = self.row_to_state_dict(r)
                except Exception as fts_err:
                    logger.debug("FTS5 query '%s' error: %s", fts_query, fts_err)

            return results

    def get_arbitration_candidates(
        self,
        conn: sqlite3.Connection,
        key: str,
        value: Any = None,
        limit: int = 100,
    ) -> Dict[str, Dict[str, Any]]:
        """Filters candidates for ConflictArbiter via exact, entity root prefix, and FTS5 search."""
        cap = max(1, min(int(limit), 200))
        results: Dict[str, Dict[str, Any]] = {}
        with contextlib.closing(conn.cursor()) as cursor:
            # 1. Exact key
            cursor.execute(
                "SELECT key, value, confidence, timestamp, metadata, value_hash FROM state_variables WHERE key = ?",
                (key,),
            )
            row = cursor.fetchone()
            if row:
                results[row["key"]] = self.row_to_state_dict(row)

            # 2. Namespace / entity root
            root = key.split(":")[0] if ":" in key else key
            if root and len(root) >= 2:
                cursor.execute(
                    "SELECT key, value, confidence, timestamp, metadata, value_hash FROM state_variables WHERE key LIKE ? LIMIT ?",
                    (f"{root}:%", cap),
                )
                for r in cursor.fetchall():
                    if r["key"] not in results:
                        results[r["key"]] = self.row_to_state_dict(r)

            # 3. FTS5 search on value and key tokens
            combined_text = f"{key} {value or ''}"
            tokens = [t for t in re.findall(r"\w+", combined_text) if len(t) >= 4 and not any(c in t for c in '":*^()')]
            if tokens and len(results) < cap:
                fts_q = " OR ".join(f'"{t}"*' for t in tokens[:6])
                try:
                    cursor.execute(
                        """
                        SELECT s.key, s.value, s.confidence, s.timestamp, s.metadata, s.value_hash
                        FROM state_variables s
                        JOIN state_variables_fts f ON s.key = f.key
                        WHERE state_variables_fts MATCH ?
                        LIMIT ?
                        """,
                        (fts_q, cap),
                    )
                    for r in cursor.fetchall():
                        if r["key"] not in results and len(results) < cap:
                            results[r["key"]] = self.row_to_state_dict(r)
                except Exception as fts_err:
                    logger.debug("FTS5 arbitration query error: %s", fts_err)

            return results

    def get_all(
        self,
        conn: sqlite3.Connection,
        limit: int = 1000,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Paginated retrieval of all state variables."""
        eff_limit = max(1, min(int(limit), 5000))
        eff_offset = max(0, int(offset))
        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute(
                "SELECT key, value, confidence, timestamp, metadata, value_hash FROM state_variables ORDER BY key LIMIT ? OFFSET ?",
                (eff_limit, eff_offset),
            )
            rows = cursor.fetchall()
            result: Dict[str, Any] = {}
            for row in rows:
                try:
                    result[row["key"]] = self.row_to_state_dict(row)
                except Exception:
                    result[row["key"]] = {
                        "key": row["key"],
                        "value": str(row["value"]) if row["value"] is not None else "",
                        "confidence": float(row["confidence"]) if row["confidence"] is not None else 1.0,
                        "timestamp": float(row["timestamp"]) if row["timestamp"] is not None else 0.0,
                        "metadata": {},
                    }
            return result

    def get_recent(
        self,
        conn: sqlite3.Connection,
        max_age_seconds: float = 900.0,
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        """Retrieves recent state variables sorted chronologically."""
        cutoff_ts = time.time() - max_age_seconds
        with contextlib.closing(conn.cursor()) as cursor:
            cursor.execute(
                "SELECT key, value, confidence, timestamp, metadata, value_hash FROM state_variables WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT ?",
                (cutoff_ts, limit),
            )
            rows = cursor.fetchall()
            results: List[Dict[str, Any]] = []
            for row in rows:
                try:
                    results.append(self.row_to_state_dict(row))
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

    def purge_stale_probe_keys(self, conn: sqlite3.Connection, prefixes: List[str]) -> int:
        """Purges test and probe keys matching given prefix patterns."""
        if not prefixes:
            raise ValueError("prefixes list must be explicitly provided to purge_stale_probe_keys")
        total_deleted = 0
        with contextlib.closing(conn.cursor()) as cursor:
            for prefix in prefixes:
                if "%" in prefix:
                    cursor.execute("DELETE FROM state_variables WHERE key LIKE ?", (prefix,))
                else:
                    cursor.execute("DELETE FROM state_variables WHERE key = ?", (prefix,))
                total_deleted += cursor.rowcount
        return total_deleted
