"""
ATLAS Cognitive Layer: Hot Working Memory Buffer Policy (V27+ Architecture).

Manages active working memory records within a dynamic recency window (TTL 900s),
enforcing session isolation, conversational churn filtering, deduplication,
and atomic supersession purging.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set

from atlas_memory.cognitive.models import (
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


def extract_effective_session(key: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Extracts session identifier from record metadata or key prefix."""
    meta = metadata or {}
    rec_session = meta.get("session_id") or meta.get("source_session_id")
    if rec_session:
        return str(rec_session)

    k_lower = key.lower()
    if k_lower.startswith("session_"):
        parts = key.split("_", 2)
        if len(parts) >= 3 and parts[0] == "session":
            return parts[1]
    elif k_lower.startswith("session:"):
        parts = key.split(":", 2)
        if len(parts) >= 3 and parts[0] == "session":
            return parts[1]
    return None


def is_session_accessible(effective_session: Optional[str], requested_session_id: Optional[str]) -> bool:
    """Evaluates session isolation: shared sessions accessible to all, private require match."""
    if str(effective_session) == HERMES_GLOBAL_SESSION:
        return True

    clean_req = str(requested_session_id).replace("session_", "").strip() if requested_session_id is not None else ""
    clean_eff = str(effective_session).replace("session_", "").strip() if effective_session is not None else ""

    # Default environment (unspecified or hermes_default/default) is only accessible to default/empty requests
    if clean_eff in (HERMES_DEFAULT_SESSION, "default") or not clean_eff:
        return clean_req in (HERMES_DEFAULT_SESSION, "default", "")

    # Private session: require exact match
    return bool(clean_req and clean_req == clean_eff)


class HotBufferPolicy:
    """
    Active working memory hot buffer policy.

    Maintains a FIFO bounded ring buffer (default capacity 25) of recent state
    variables and user assertions within a dynamic TTL window (default 900s / 15min).
    """

    def __init__(self, maxlen: int = 25, ttl_seconds: float = 900.0) -> None:
        self.maxlen = maxlen
        self.ttl_seconds = ttl_seconds
        self._buffer: deque[HotBufferEntry] = deque(maxlen=maxlen)

    def append(self, entry: HotBufferEntry | Dict[str, Any]) -> None:
        """Appends an entry to the hot buffer."""
        if isinstance(entry, HotBufferEntry):
            self._buffer.append(entry)
        elif isinstance(entry, dict):
            self._buffer.append(
                HotBufferEntry(
                    key=entry["key"],
                    value=entry.get("value", entry.get("object", "")),
                    confidence=float(entry.get("confidence", 1.0)),
                    source_type=entry.get("source_type", "user_explicit"),
                    timestamp=float(entry.get("timestamp", time.time())),
                    is_superseded=bool(entry.get("is_superseded", False)),
                    metadata=dict(entry.get("metadata") or {}),
                )
            )

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
        now = time.time() if timestamp is None else timestamp
        meta = dict(metadata or {})
        entry = HotBufferEntry(
            key=key,
            value=value,
            confidence=confidence,
            source_type=source_type,
            timestamp=now,
            is_superseded=bool(meta.get("is_superseded", False)),
            metadata=meta,
        )
        self._buffer.append(entry)
        return entry

    def purge(self, key: str) -> int:
        """Purges any entries matching key (e.g. when superseded by belief revision)."""
        initial_len = len(self._buffer)
        remaining = [e for e in self._buffer if e.key != key]
        self._buffer = deque(remaining, maxlen=self.maxlen)
        purged = initial_len - len(self._buffer)
        if purged > 0:
            logger.debug("[HotBuffer] Purged %d entry for key='%s'", purged, key)
        return purged

    def mark_superseded(self, key: str) -> int:
        """Marks any entries with key as superseded and purges them from active buffer."""
        count = 0
        for entry in self._buffer:
            if entry.key == key:
                entry.is_superseded = True
                count += 1
        self.purge(key)
        return count

    def get_active(
        self,
        session_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> List[HotBufferEntry]:
        """
        Returns active, non-superseded, non-churn hot buffer entries within TTL window,
        deduplicated in reverse chronological order (most recent version of each key wins).
        """
        current_time = time.time() if now is None else now
        cutoff = current_time - self.ttl_seconds

        # Filter valid items within TTL window
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

        # Deduplicate keys: reverse order to pick the newest entry per key
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
        if kv_store is None:
            return 0
        age = max_age_seconds if max_age_seconds is not None else self.ttl_seconds
        lim = limit if limit is not None else self.maxlen

        items: List[Dict[str, Any]] = []
        if hasattr(kv_store, "get_recent_sync"):
            try:
                items = kv_store.get_recent_sync(max_age_seconds=age, limit=lim)
            except Exception as exc:
                logger.warning("[HotBuffer] seed_from_kv error: %s", exc)
                return 0
        elif hasattr(kv_store, "get_all_sync"):
            try:
                all_items = kv_store.get_all_sync()
                now = time.time()
                items = [
                    {"key": k, **v} for k, v in all_items.items()
                    if (now - float(v.get("timestamp", 0.0))) <= age
                ][:lim]
            except Exception as exc:
                logger.warning("[HotBuffer] seed_from_kv fallback error: %s", exc)
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
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self):
        return iter(self._buffer)
