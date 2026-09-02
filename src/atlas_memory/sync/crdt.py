"""Delta-State Conflict-Free Replicated Data Types (CRDT) for Memory Synchronization."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field, PrivateAttr


class VectorClock(BaseModel):
    """Vector Clock for tracking causal relationships across distributed agents."""

    clocks: Dict[str, int] = Field(default_factory=dict)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def increment(self, node_id: str) -> None:
        """Increments the logical clock for the given node."""
        with self._lock:
            self.clocks[node_id] = self.clocks.get(node_id, 0) + 1

    def merge(self, other: VectorClock) -> VectorClock:
        """Merges another vector clock by taking the component-wise maximum."""
        with self._lock:
            all_keys = set(self.clocks.keys()) | set(other.clocks.keys())
            merged = {k: max(self.clocks.get(k, 0), other.clocks.get(k, 0)) for k in all_keys}
            self.clocks = merged
            return VectorClock(clocks=merged)

    def happens_before(self, other: VectorClock) -> bool:
        """Returns True if this vector clock strictly happens before the other clock."""
        all_keys = set(self.clocks.keys()) | set(other.clocks.keys())
        has_strictly_less = False
        for k in all_keys:
            v_self = self.clocks.get(k, 0)
            v_other = other.clocks.get(k, 0)
            if v_self > v_other:
                return False
            if v_self < v_other:
                has_strictly_less = True
        return has_strictly_less

    def concurrent(self, other: VectorClock) -> bool:
        """Returns True if the two vector clocks are concurrent (neither dominates)."""
        if self.clocks == other.clocks:
            return False
        return not self.happens_before(other) and not other.happens_before(self)


class LWWElementSet(BaseModel):
    """Last-Write-Wins Element Set (LWW-Element-Set) with deterministic tie-breaking."""

    adds: Dict[str, Tuple[float, str]] = Field(default_factory=dict)
    removes: Dict[str, Tuple[float, str]] = Field(default_factory=dict)
    tombstone_ttl_seconds: Optional[float] = Field(
        default=None,
        description="TTL for tombstones (removes). None = no GC.",
    )

    def gc(self, now: Optional[float] = None) -> int:
        """Removes tombstones older than tombstone_ttl_seconds.

        If tombstone_ttl_seconds is None, GC is disabled and returns 0.
        Returns the number of tombstone entries purged.
        """
        if self.tombstone_ttl_seconds is None:
            return 0

        current_time = time.time() if now is None else now
        cutoff = current_time - self.tombstone_ttl_seconds
        removed_keys = [elem for elem, (ts, _) in self.removes.items() if ts < cutoff]

        for elem in removed_keys:
            del self.removes[elem]
            # Also clean up corresponding stale add entry if it is older than or equal to the tombstone
            if elem in self.adds:
                add_ts, _ = self.adds[elem]
                if add_ts < cutoff:
                    del self.adds[elem]

        return len(removed_keys)

    def add(self, element: str, timestamp: Optional[float] = None, node_id: str = "") -> None:
        """Adds an element with timestamp and node_id for tie-breaking."""
        ts = time.time() if timestamp is None else timestamp
        meta = (ts, node_id)
        if element not in self.adds or meta > self.adds[element]:
            self.adds[element] = meta

    def remove(self, element: str, timestamp: Optional[float] = None, node_id: str = "") -> None:
        """Removes an element with timestamp and node_id for tie-breaking."""
        ts = time.time() if timestamp is None else timestamp
        meta = (ts, node_id)
        if element not in self.removes or meta > self.removes[element]:
            self.removes[element] = meta

    def lookup(self, element: str) -> bool:
        """Determines if the element is currently present using LWW rules."""
        if element not in self.adds:
            return False
        if element not in self.removes:
            return True

        add_meta = self.adds[element]
        rem_meta = self.removes[element]

        # Higher timestamp wins; if timestamps equal, lexicographical node_id tie-breaks
        return add_meta > rem_meta

    def merge(self, other: LWWElementSet) -> LWWElementSet:
        """Merges with another LWWElementSet monotonically."""
        new_adds = dict(self.adds)
        for elem, meta in other.adds.items():
            if elem not in new_adds or meta > new_adds[elem]:
                new_adds[elem] = meta

        new_removes = dict(self.removes)
        for elem, meta in other.removes.items():
            if elem not in new_removes or meta > new_removes[elem]:
                new_removes[elem] = meta

        self.adds = new_adds
        self.removes = new_removes
        return LWWElementSet(adds=new_adds, removes=new_removes)


class SyncDelta(BaseModel):
    """Payload delta representing a state mutation between nodes."""

    source_node: str = Field(...)
    vector_clock: VectorClock = Field(default_factory=VectorClock)
    payload: Dict[str, Any] = Field(default_factory=dict)
    encrypted_payload: Optional[bytes] = Field(default=None)


class DeltaCRDT:
    """State-based Delta-CRDT container for agent memory graphs and key-value state."""

    def __init__(self, node_id: str) -> None:
        self.node_id: str = node_id
        self.clock: VectorClock = VectorClock(clocks={node_id: 0})
        self.state: Dict[str, LWWElementSet] = {}

    def get_set(self, key: str) -> LWWElementSet:
        """Gets or initializes an LWWElementSet for the given key."""
        if key not in self.state:
            self.state[key] = LWWElementSet()
        return self.state[key]

    def export_delta(self, since_vector: Optional[VectorClock] = None) -> SyncDelta:
        """Exports state delta since the given vector clock."""
        payload_state: Dict[str, Dict[str, Any]] = {}
        for key, elem_set in self.state.items():
            payload_state[key] = elem_set.model_dump()

        return SyncDelta(
            source_node=self.node_id,
            vector_clock=VectorClock(clocks=dict(self.clock.clocks)),
            payload={"state": payload_state},
        )

    def apply_delta(self, delta: SyncDelta) -> VectorClock:
        """Applies an incoming delta by merging element sets and vector clocks."""
        state_payload = delta.payload.get("state", {})
        for key, set_data in state_payload.items():
            incoming_set = LWWElementSet.model_validate(set_data)
            if key in self.state:
                self.state[key].merge(incoming_set)
            else:
                self.state[key] = incoming_set

        self.clock.merge(delta.vector_clock)
        return self.clock
