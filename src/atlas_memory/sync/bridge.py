"""StateSyncBridge: Storage-backed distributed memory bridge for ATLAS.

Connects the distributed synchronization protocol directly to physical storage:
- Reads mutations from VerifiedKVStore state_audit_log (SQLite WAL).
- Applies incoming remote mutations through EpistemicWritePath (belief revision, Kùzu graph, Qdrant vectors).
- Maintains causality tracking via CompactVectorClock.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from typing import Any, Dict, List, Union

from atlas_memory.core.models import MemoryRecord
from atlas_memory.sync.crdt import CompactVectorClock, SyncDelta, VectorClock

logger = logging.getLogger(__name__)


class StateSyncBridge:
    """Most replikacji stanu łączący protokół rozproszony z VerifiedKVStore i EpistemicWritePath."""

    def __init__(self, kv_store: Any, write_path: Any, node_id: str) -> None:
        self.kv_store = kv_store
        self.write_path = write_path
        self.node_id = str(node_id)
        self.clock = CompactVectorClock(clocks={self.node_id: 0})

    def export_delta_from_audit(self, since_seq: int = 0, limit: int = 200) -> Dict[str, Any]:
        """Eksportuje rzeczywiste mutacje stanu z bazy SQLite state_audit_log jako podpisaną deltę."""
        if self.kv_store is None:
            return {
                "source_node": self.node_id,
                "vector_clock": dict(self.clock.clocks),
                "max_seq": since_seq,
                "mutations": [],
            }

        safe_limit = max(1, min(int(limit), 1000))
        mutations: List[Dict[str, Any]] = []
        max_seq = int(since_seq)

        # Bezpieczny dostęp do bazy SQLite chroniony blokadą KV
        lock = getattr(self.kv_store, "_sync_lock", None)
        if lock is not None:
            with lock:
                rows = self._query_audit_rows(since_seq, safe_limit)
        else:
            rows = self._query_audit_rows(since_seq, safe_limit)

        for row in rows:
            seq_val = int(row["seq"])
            if seq_val > max_seq:
                max_seq = seq_val

            val_str = row["new_value"] if "new_value" in row.keys() else None
            conf_val = float(row["confidence"]) if "confidence" in row.keys() and row["confidence"] is not None else 1.0
            reason_val = str(row["reason"]) if "reason" in row.keys() and row["reason"] is not None else "sync"

            mutations.append({
                "seq": seq_val,
                "key": str(row["key"]),
                "value": val_str,
                "confidence": conf_val,
                "reason": reason_val,
                "timestamp": float(row["timestamp"]) if "timestamp" in row.keys() else 0.0,
            })

        self.clock.increment(self.node_id)
        return {
            "source_node": self.node_id,
            "vector_clock": dict(self.clock.clocks),
            "max_seq": max_seq,
            "mutations": mutations,
        }

    def _query_audit_rows(self, since_seq: int, limit: int) -> List[Any]:
        """Pomocnicze zapytanie do state_audit_log z uwzględnieniem podłączenia bazy audit."""
        try:
            conn = self.kv_store._ensure_conn()
            with contextlib.closing(conn.cursor()) as cursor:
                cursor.execute(
                    "SELECT seq, timestamp, key, new_value, confidence, reason "
                    "FROM state_audit_log WHERE seq > ? ORDER BY seq ASC LIMIT ?",
                    (since_seq, limit),
                )
                return cursor.fetchall() or []
        except Exception as exc:
            logger.debug("StateSyncBridge failed querying audit rows: %s", exc)
            return []

    async def apply_remote_delta(self, delta: Union[Dict[str, Any], SyncDelta]) -> int:
        """Aplikuje otrzymaną deltę bezpośrednio przez autorytet EpistemicWritePath."""
        if isinstance(delta, SyncDelta):
            delta_dict = delta.payload
            source_node = delta.source_node
            peer_vc = delta.vector_clock
        elif isinstance(delta, dict):
            delta_dict = delta
            source_node = delta.get("source_node", "peer")
            peer_vc = delta.get("vector_clock", {})
        else:
            return 0

        mutations = delta_dict.get("mutations", [])
        applied_count = 0

        # V88/T5: odrzucanie delt przezaśniętych (stale). Delta jest "stale", gdy
        # zegar własny węzła DOMINUJE zegar peera (peer_vc <= self.clock component-wise):
        # oznacza to dostarczenie zdarzeń już odnotowanych — bez tej ochrony replay
        # tej samej delty był stosowany wielokrotnie, nadpisując świeższy stan.
        peer_clock = None
        if peer_vc:
            peer_clock = peer_vc if isinstance(peer_vc, VectorClock) else VectorClock.model_validate(peer_vc)
            peer_clocks = getattr(peer_clock, "clocks", {}) or {}
            own = self.clock.clocks
            dominates = bool(peer_clocks) and all(
                own.get(k, 0) >= v for k, v in peer_clocks.items()
            )
            if dominates:
                logger.warning(
                    "STALE_DELTA rejected from node '%s' (peer clock %s dominated by local %s)",
                    source_node,
                    peer_clocks,
                    own,
                )
                return 0

        for m in mutations:
            key = str(m.get("key", "")).strip()
            if not key:
                continue

            incoming_conf = float(m.get("confidence", 0.9))
            incoming_prec = float(m.get("precision", 1.0))
            if "metadata" in m and isinstance(m["metadata"], dict) and "precision" in m["metadata"]:
                try:
                    incoming_prec = float(m["metadata"]["precision"])
                except (ValueError, TypeError):
                    pass
            incoming_lambda = incoming_prec * incoming_conf
            incoming_ts = float(m.get("timestamp", 0.0))

            # V155 Filar III: Weryfikacja potencjału informacyjnego Fristona (Lambda = Pi * conf).
            # Odporność na sztuczny dryf zegara fizycznego OS: znacznik z przyszłości nie może nadpisać wyższej precyzji.
            if self.kv_store is not None and hasattr(self.kv_store, "get_sync"):
                existing = self.kv_store.get_sync(key)
                if existing is not None:
                    local_conf = float(existing.get("confidence", 1.0))
                    local_meta = existing.get("metadata") or {}
                    local_prec = 1.0
                    if isinstance(local_meta, dict) and "precision" in local_meta:
                        try:
                            local_prec = float(local_meta["precision"])
                        except (ValueError, TypeError):
                            pass
                    elif "precision" in existing:
                        try:
                            local_prec = float(existing["precision"])
                        except (ValueError, TypeError):
                            pass
                    local_lambda = local_prec * local_conf

                    if incoming_lambda < local_lambda - 1e-6:
                        logger.warning(
                            "REJECTED_LOW_PRECISION_MUTATION: key='%s' incoming Lambda=%.3f < local Lambda=%.3f "
                            "(incoming Pi=%.2f, conf=%.2f vs local Pi=%.2f, conf=%.2f, ts=%.1f vs local_ts=%.1f)",
                            key, incoming_lambda, local_lambda, incoming_prec, incoming_conf,
                            local_prec, local_conf, incoming_ts, float(existing.get("timestamp", 0.0))
                        )
                        continue

            val_raw = m.get("value")
            is_deleted = (
                val_raw in ('"__DELETED__"', '__DELETED__')
                or m.get("reason") in ("delete", "agent_delete")
            )

            if is_deleted:
                if self.kv_store is not None:
                    if hasattr(self.kv_store, "delete_sync"):
                        self.kv_store.delete_sync(key)
                    elif hasattr(self.kv_store, "delete_state"):
                        res = self.kv_store.delete_state(key)
                        if hasattr(res, "__await__"):
                            await res
                    elif hasattr(self.kv_store, "delete_state_sync"):
                        self.kv_store.delete_state_sync(key)

                # Purge from vector store to avoid stale/orphan peer vectors
                if self.write_path is not None:
                    v_store = getattr(self.write_path, "vector_store", None)
                    if v_store is not None:
                        del_sync = getattr(v_store, "delete_by_upsert_key_sync", None)
                        if callable(del_sync):
                            del_sync(f"kv:{key}")
                            del_sync(key)
                applied_count += 1
            else:
                val: Any
                if isinstance(val_raw, str):
                    try:
                        val = json.loads(val_raw)
                    except (json.JSONDecodeError, TypeError):
                        val = val_raw
                else:
                    val = val_raw

                conf = float(m.get("confidence", 0.9))
                reason = f"sync_from_{source_node}"

                # Infer subject/predicate/object for graph wiring
                val_text = str(val)
                parts = key.split(":", 2)
                if len(parts) == 3:
                    sub_val, pred_val, obj_val = parts[0], parts[1], str(parts[2] if not val_text else val_text)
                elif len(parts) == 2:
                    sub_val, pred_val, obj_val = parts[0], parts[1], val_text
                else:
                    sub_val, pred_val, obj_val = key, "has_state", val_text

                meta_dict = {
                    "source_type": "peer_replication",
                    "origin": f"sync_{source_node}",
                    "subject": sub_val,
                    "predicate": pred_val,
                    "object": obj_val,
                    "kv_key": key,
                    "precision": incoming_prec,
                }

                # Construct MemoryRecord for Qdrant vector indexing
                try:
                                v_rec = MemoryRecord(
                        subject=f"kv:{key}",
                        predicate="state_is",
                        object=val_text,
                        confidence=conf,
                        timestamp=time.time(),
                        is_state_variable=True,
                        metadata=meta_dict,
                    )
                except Exception:
                    v_rec = None

                if self.write_path is not None:
                    if hasattr(self.write_path, "write_async"):
                        await self.write_path.write_async(
                            key=key,
                            value=val,
                            confidence=conf,
                            metadata=meta_dict,
                            source_type="peer_replication",
                            reason=reason,
                            vector_record=v_rec,
                        )
                    elif hasattr(self.write_path, "write"):
                        self.write_path.write(
                            key=key,
                            value=val,
                            confidence=conf,
                            metadata=meta_dict,
                            source_type="peer_replication",
                            reason=reason,
                            vector_record=v_rec,
                        )
                elif self.kv_store is not None and hasattr(self.kv_store, "set_sync"):
                    self.kv_store.set_sync(
                        key=key,
                        value=val,
                        confidence=conf,
                        metadata=meta_dict,
                        reason=reason,
                    )
                applied_count += 1

        # Synchronizacja zegara logicznego (V88/T5): odbiór jest ZDARZENIEM LOKALNYM,
        # więc po component-wise max inkrementujemy własny licznik (reguła Lamporta).
        # Dzięki temu kolejny replay tej samej delty jest rozpoznawany jako stale.
        if peer_clock is not None:
            self.clock.receive(peer_clock, self.node_id)

        return applied_count
