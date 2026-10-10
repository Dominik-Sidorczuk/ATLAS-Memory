"""ATLAS L1 Working Memory: Hot Working Memory Buffer Policy (HotKVBuffer / HotBufferPolicy).

Manages active L1 working memory records within a dynamic recency window (TTL 900s),
enforcing session isolation, conversational churn filtering, deduplication,
and atomic supersession purging.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set

from atlas_memory.core.models import (
    HERMES_DEFAULT_SESSION,
    HERMES_GLOBAL_SESSION,
    HotBufferEntry,
)

logger = logging.getLogger(__name__)


def is_conversational_churn(key: str, content: str) -> bool:
    """Detects raw conversation turns, prompt noise, or tracebacks."""
    s_lower = key.lower()
    c_lower = str(content).lower()
    noise_indicators = (
        "[user]", "[assistant]", "mission order", "### session",
        "traceback (most recent call last)", "conversation___user",
        "chciałbym, abyś", "test 1: test", "test aktywnego użycia",
    )
    if any(ind in s_lower or ind in c_lower for ind in noise_indicators):
        return True
    if s_lower.startswith(("fact:conversation___user", "fact:conversation", "conversation___user", "conversation:")):
        return True
    return False


PERSISTENT_KEY_PREFIXES = (
    "fact:", "rule:", "preference:", "constraint:", "config:",
    "system:", "user:", "hermes:", "prompt-", "mnemosyne:",
)


def extract_effective_session(key: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Extracts session identifier from record key prefix or metadata fallback.

    Precedence order (v138):
    (1) Key prefix 'session:' or 'session_' (preserves full ID regex).
    (2) Persistent key prefixes (PERSISTENT_KEY_PREFIXES -> HERMES_GLOBAL_SESSION).
    (3) Metadata session fallback (session_id / source_session_id).
    (4) None.
    """
    k_lower = key.lower()

    if k_lower.startswith("session:"):
        parts = key.split(":", 2)
        if len(parts) >= 2 and parts[0].lower() == "session":
            return parts[1]
    elif k_lower.startswith("session_"):
        if ":" in key:
            return key[len("session_"):].split(":", 1)[0]
        m = re.match(r"^session_(\d{8}_\d{6}_[a-f0-9]+)", key, re.I)
        if m:
            return m.group(1)
        parts = key.split("_", 2)
        if len(parts) >= 3 and parts[0].lower() == "session":
            return parts[1]
        return key[len("session_"):]

    if any(k_lower.startswith(p) for p in PERSISTENT_KEY_PREFIXES):
        return HERMES_GLOBAL_SESSION

    meta = metadata or {}
    rec_session = meta.get("session_id") or meta.get("source_session_id")
    if rec_session:
        rec_s_str = str(rec_session)
        if rec_s_str in ("hermes_persistent", "global", HERMES_GLOBAL_SESSION):
            return HERMES_GLOBAL_SESSION
        if rec_s_str not in (HERMES_DEFAULT_SESSION, "default", ""):
            return rec_s_str
        return str(rec_session)

    return None


def is_session_accessible(effective_session: Optional[str], requested_session_id: Optional[str]) -> bool:
    """Evaluates session isolation: shared sessions accessible to all, private require match."""
    if effective_session is None or effective_session == HERMES_GLOBAL_SESSION:
        return True

    clean_req = str(requested_session_id).replace("session_", "").strip() if requested_session_id is not None else ""
    clean_eff = str(effective_session).replace("session_", "").strip()

    if clean_eff in (HERMES_DEFAULT_SESSION, "default") or not clean_eff:
        return clean_req in (HERMES_DEFAULT_SESSION, "default", "")

    return bool(clean_req and clean_req == clean_eff)


class HotBufferPolicy:
    """Active L1 working memory hot buffer policy (FIFO ring buffer, default capacity 25, TTL 900s)."""

    def __init__(self, maxlen: int = 25, ttl_seconds: float = 900.0) -> None:
        self.maxlen = maxlen
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._buffer: deque[HotBufferEntry] = deque(maxlen=maxlen)
        self.superseded_history: Set[str] = set()

    def append(self, entry: HotBufferEntry | Dict[str, Any]) -> None:
        """Appends an entry to the hot buffer, removing existing entry with same key if present."""
        with self._lock:
            if isinstance(entry, HotBufferEntry):
                ent = entry
            elif isinstance(entry, dict):
                ent = HotBufferEntry(
                    key=entry["key"],
                    value=entry.get("value", entry.get("object", "")),
                    confidence=float(entry.get("confidence", 1.0)),
                    source_type=entry.get("source_type", "user_explicit"),
                    timestamp=float(entry.get("timestamp", time.time())),
                    is_superseded=bool(entry.get("is_superseded", False)),
                    metadata=dict(entry.get("metadata") or {}),
                )
            else:
                return
            if ent.is_superseded:
                self.superseded_history.add(ent.key)
            self.purge(ent.key)
            self._buffer.append(ent)

    def update(
        self,
        key: str,
        value: Any,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        source_type: str = "user_explicit",
        timestamp: Optional[float] = None,
    ) -> HotBufferEntry:
        """Appends or updates a key in the working memory hot buffer."""
        with self._lock:
            now = time.time() if timestamp is None else timestamp
            meta = dict(metadata or {})
            is_sup = bool(meta.get("is_superseded", False))
            if is_sup:
                self.superseded_history.add(key)
            entry = HotBufferEntry(
                key=key,
                value=value,
                confidence=confidence,
                source_type=source_type,
                timestamp=now,
                is_superseded=is_sup,
                metadata=meta,
            )
            self.purge(key)
            self._buffer.append(entry)
            return entry

    def purge(self, key: str) -> int:
        """Purges any entries matching key (e.g. when superseded by belief revision)."""
        with self._lock:
            initial_len = len(self._buffer)
            remaining = [e for e in self._buffer if e.key != key]
            self._buffer = deque(remaining, maxlen=self.maxlen)
            purged = initial_len - len(self._buffer)
            if purged > 0:
                logger.debug("[HotBuffer] Purged %d entry for key='%s'", purged, key)
            return purged

    def mark_superseded(self, key: str) -> int:
        """Marks any entries with key as superseded and purges them from active buffer."""
        with self._lock:
            self.superseded_history.add(key)
            count = 0
            for entry in self._buffer:
                if entry.key == key:
                    entry.is_superseded = True
                    count += 1
            self.purge(key)
            return count

    def reconcile_superseded(self, superseded_keys: Any) -> int:
        """V109 Pillar 1: Batch-marks and purges all superseded keys from L1 hot buffer."""
        with self._lock:
            total_purged = 0
            for k in list(superseded_keys or []):
                if k:
                    self.superseded_history.add(str(k))
                    total_purged += self.mark_superseded(str(k))
            return total_purged

    def get_session_entries(self, session_id: Optional[str] = None) -> List[HotBufferEntry]:
        """V109 Pillar 1: Returns unreconciled L1 entries associated with session_id."""
        with self._lock:
            out: List[HotBufferEntry] = []
            for it in self._buffer:
                if it.is_superseded or it.metadata.get("is_superseded"):
                    continue
                consolidated_sessions = it.metadata.get("l2_consolidated_sessions")
                if consolidated_sessions is not None:
                    if session_id and session_id in consolidated_sessions:
                        continue
                elif it.metadata.get("l2_consolidated"):
                    continue
                raw_sess = str(it.metadata.get("session_id") or it.metadata.get("source_session_id") or "")
                eff_sess = extract_effective_session(it.key, it.metadata) or ""
                if session_id:
                    if (
                        raw_sess == session_id
                        or eff_sess == session_id
                        or (
                            session_id != "idle_sess"
                            and not it.metadata.get("seeded_from_kv")
                            and raw_sess in ("", "default", HERMES_DEFAULT_SESSION, HERMES_GLOBAL_SESSION)
                        )
                    ):
                        out.append(it)
                else:
                    out.append(it)
            return out

    def get_active(
        self,
        session_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> List[HotBufferEntry]:
        """Returns active, non-superseded, non-churn hot buffer entries within TTL window."""
        with self._lock:
            current_time = time.time() if now is None else now
            cutoff = current_time - self.ttl_seconds

            valid_items: List[HotBufferEntry] = []
            for it in self._buffer:
                if it.timestamp < cutoff:
                    continue
                if it.is_superseded or it.metadata.get("is_superseded"):
                    continue
                if is_conversational_churn(it.key, str(it.value)):
                    continue
                eff_sess = extract_effective_session(it.key, it.metadata)
                if session_id and not is_session_accessible(eff_sess, session_id):
                    continue
                valid_items.append(it)

            seen_keys: Set[str] = set()
            deduped: List[HotBufferEntry] = []
            for it in reversed(valid_items):
                if it.key not in seen_keys:
                    seen_keys.add(it.key)
                    deduped.append(it)
            deduped.reverse()
            return deduped

    def seed_from_kv(
        self,
        kv_store: Any,
        max_age_seconds: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> int:
        """Seeds hot buffer from persistent KV store."""
        with self._lock:
            if kv_store is None:
                return 0
            age = max_age_seconds if max_age_seconds is not None else self.ttl_seconds
            lim = limit if limit is not None else self.maxlen

            items: List[Dict[str, Any]] = []
            if hasattr(kv_store, "get_recent_sync"):
                try:
                    raw_items = kv_store.get_recent_sync(max_age_seconds=age, limit=lim)
                    if isinstance(raw_items, list):
                        items = raw_items
                except Exception as exc:
                    logger.warning("[HotBuffer] seed_from_kv error: %s", exc)
                    return 0

            if not items:
                return 0

            seeded = 0
            for it in items:
                meta = dict(it.get("metadata") or {})
                if meta.get("is_superseded"):
                    continue
                self.append({
                    "key": it["key"],
                    "value": it.get("value", ""),
                    "timestamp": float(it.get("timestamp", time.time())),
                    "confidence": float(it.get("confidence", 1.0)),
                    "metadata": meta,
                    "source_type": meta.get("source_type", "user_explicit"),
                })
                seeded += 1
            logger.info("[HotBuffer] Seeded %d items from KVStore", seeded)
            return seeded

    def clear(self) -> None:
        """Clears all entries in buffer."""
        with self._lock:
            self._buffer.clear()
            self.superseded_history.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self):
        return iter(self._buffer)


HotKVBuffer = HotBufferPolicy

__all__ = [
    "HotBufferEntry",
    "HotBufferPolicy",
    "HotKVBuffer",
    "PERSISTENT_KEY_PREFIXES",
    "extract_effective_session",
    "is_conversational_churn",
    "is_session_accessible",
]
