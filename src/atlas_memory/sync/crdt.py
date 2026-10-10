"""Delta-State Conflict-Free Replicated Data Types (CRDT) for Memory Synchronization (V83 Refactor).

Replaced heavy Pydantic models with lightweight, slotted dataclasses (Python 3.14 free-threaded optimized).
Includes automatic tombstone garbage collection and canonical model serialization helpers.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union


@dataclass(slots=True)
class CompactVectorClock:
    """Vector Clock for tracking causal relationships across distributed agents."""

    clocks: Dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def increment(self, node_id: str) -> None:
        """Increments the logical clock for the given node."""
        with self._lock:
            self.clocks[node_id] = self.clocks.get(node_id, 0) + 1

    def merge(self, other: Union[CompactVectorClock, Dict[str, int], Any]) -> CompactVectorClock:
        """Merges another vector clock by taking the component-wise maximum."""
        other_clocks = other.clocks if hasattr(other, "clocks") else (other if isinstance(other, dict) else {})
        with self._lock:
            all_keys = self.clocks.keys() | other_clocks.keys()
            merged = {k: max(self.clocks.get(k, 0), other_clocks.get(k, 0)) for k in all_keys}
            self.clocks = merged
            return CompactVectorClock(clocks=dict(merged))

    def receive(self, other: Union[CompactVectorClock, Dict[str, int], Any], node_id: str) -> CompactVectorClock:
        """Rejestruje ODBIÓR jako zdarzenie lokalne (reguła Lamporta: merge + inkrement).

        `merge()` liczy wyłącznie component-wise max, więc po odebraniu zdarzeń peera
        zegar odbiorcy był *równy* zegarowi nadawcy i relacja przyczynowa
        (`happens_before`) nie była reprezentowalna — replay delty był nieodróżnialny
        od świeżej dostawy. `receive()` po maksimum inkrementuje własny licznik węzła,
        dzięki czemu zegar odbiorcy ściśle dominuje zegar dostarczonej delty.
        """
        other_clocks = other.clocks if hasattr(other, "clocks") else (other if isinstance(other, dict) else {})
        with self._lock:
            all_keys = self.clocks.keys() | other_clocks.keys()
            merged = {k: max(self.clocks.get(k, 0), other_clocks.get(k, 0)) for k in all_keys}
            merged[node_id] = merged.get(node_id, 0) + 1
            self.clocks = merged
            return CompactVectorClock(clocks=dict(merged))

    def happens_before(self, other: Union[CompactVectorClock, Any]) -> bool:
        """Returns True if this vector clock strictly happens before the other clock."""
        other_clocks = other.clocks if hasattr(other, "clocks") else (other if isinstance(other, dict) else {})
        all_keys = self.clocks.keys() | other_clocks.keys()
        has_strictly_less = False
        for k in all_keys:
            v_self = self.clocks.get(k, 0)
            v_other = other_clocks.get(k, 0)
            if v_self > v_other:
                return False
            if v_self < v_other:
                has_strictly_less = True
        return has_strictly_less

    def concurrent(self, other: Union[CompactVectorClock, Any]) -> bool:
        """Returns True if the two vector clocks are concurrent (neither dominates)."""
        other_clocks = other.clocks if hasattr(other, "clocks") else (other if isinstance(other, dict) else {})
        if self.clocks == other_clocks:
            return False
        return not self.happens_before(other) and not other.happens_before(self)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        """Pydantic core schema generation allowing seamless use inside Pydantic models."""
        from pydantic_core import core_schema
        return core_schema.chain_schema([
            core_schema.any_schema(),
            core_schema.no_info_plain_validator_function(cls.model_validate),
        ])

    def model_dump(self) -> Dict[str, Any]:
        """Canonical dictionary serialization."""
        return {"clocks": dict(self.clocks)}

    def model_dump_json(self) -> str:
        """Canonical JSON string serialization."""
        return json.dumps(self.model_dump())

    @classmethod
    def model_validate(cls, data: Any) -> CompactVectorClock:
        """Canonical validation from dict or object."""
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            clocks = data.get("clocks", data)
            return cls(clocks=dict(clocks) if isinstance(clocks, dict) else {})
        return cls()

    @classmethod
    def model_validate_json(cls, json_str: Union[str, bytes]) -> CompactVectorClock:
        """Canonical deserialization from JSON."""
        data = json.loads(json_str)
        return cls.model_validate(data)


# Canonical alias
VectorClock = CompactVectorClock


def _normalize_crdt_meta(raw_meta: Any) -> Tuple[float, int, float, str]:
    """Normalizuje metadane CRDT do krotki (lambda_potential, clock_rank, timestamp, node_id).

    Dla danych wstecznie kompatybilnych (timestamp, node_id):
    lambda_potential = 1.0, clock_rank = 0.
    """
    if isinstance(raw_meta, (tuple, list)):
        if len(raw_meta) >= 4:
            return (float(raw_meta[0]), int(raw_meta[1]), float(raw_meta[2]), str(raw_meta[3]))
        elif len(raw_meta) == 2:
            return (1.0, 0, float(raw_meta[0]), str(raw_meta[1]))
        elif len(raw_meta) == 3:
            return (float(raw_meta[0]), 0, float(raw_meta[1]), str(raw_meta[2]))
    return (1.0, 0, 0.0, "")


def _crdt_meta_dominates(a: Tuple[Any, ...], b: Optional[Tuple[Any, ...]]) -> bool:
    """Relacja częściowego porządku oparta na potencjale informacyjnym Fristona (V155 Filar III).

    Lambda = Pi_channel * confidence.
    1. Wyższy potencjał Lambda bezwzględnie dominuje (odporność na sztuczny dryf zegara fizycznego OS).
    2. Przy równości potencjałów decyduje logiczny zegar/ranking (VectorClock).
    3. Dopiero przy równej przyczynowości decyduje fizyczny timestamp i node_id.
    """
    if b is None:
        return True
    lam_a, rank_a, ts_a, node_a = _normalize_crdt_meta(a)
    lam_b, rank_b, ts_b, node_b = _normalize_crdt_meta(b)

    # 1. Potencjał informacyjny Fristona (tolerancja 1e-6)
    if lam_a > lam_b + 1e-6:
        return True
    if lam_a < lam_b - 1e-6:
        return False

    # 2. Równość potencjałów Fristona -> rozstrzyga VectorClock rank
    if rank_a > rank_b:
        return True
    if rank_a < rank_b:
        return False

    # 3. Przy równej przyczynowości rozstrzyga fizyczny timestamp
    if ts_a > ts_b:
        return True
    if ts_a < ts_b:
        return False

    # 4. Deterministyczny tie-breaker węzła
    return node_a >= node_b


@dataclass(slots=True)
class CompactLWWSet:
    """Precision-CRDT Element Set with Friston Information Potential partial order and auto GC (V155 Filar III)."""

    adds: Dict[str, Tuple[Any, ...]] = field(default_factory=dict)
    removes: Dict[str, Tuple[Any, ...]] = field(default_factory=dict)
    tombstone_ttl_seconds: Optional[float] = None

    def gc(self, now: Optional[float] = None) -> int:
        """Removes tombstones older than tombstone_ttl_seconds.

        If tombstone_ttl_seconds is None, GC is disabled and returns 0.
        Returns the number of tombstone entries purged.
        """
        if self.tombstone_ttl_seconds is None:
            return 0

        current_time = time.time() if now is None else now
        cutoff = current_time - self.tombstone_ttl_seconds
        removed_keys = []
        for elem, meta in self.removes.items():
            norm = _normalize_crdt_meta(meta)
            if norm[2] < cutoff:
                removed_keys.append(elem)

        for elem in removed_keys:
            del self.removes[elem]
            if elem in self.adds:
                norm_add = _normalize_crdt_meta(self.adds[elem])
                if norm_add[2] < cutoff:
                    del self.adds[elem]

        return len(removed_keys)

    def auto_gc(self, now: Optional[float] = None) -> int:
        """Alias for gc() to support automatic garbage collection during sync."""
        return self.gc(now)

    def add(
        self,
        element: str,
        timestamp: Optional[float] = None,
        node_id: str = "",
        precision: float = 1.0,
        confidence: float = 1.0,
        vector_clock: Optional[Union[CompactVectorClock, int]] = None,
    ) -> None:
        """Adds an element with Friston precision potential, vector clock, timestamp, and node_id."""
        ts = time.time() if timestamp is None else float(timestamp)
        prec = float(precision)
        conf = float(confidence)
        lambda_pot = prec * conf
        clock_rank = 0
        if vector_clock is not None:
            if isinstance(vector_clock, CompactVectorClock):
                clock_rank = sum(vector_clock.clocks.values())
            else:
                clock_rank = int(vector_clock)
        meta = (float(lambda_pot), int(clock_rank), float(ts), str(node_id))
        if element not in self.adds or _crdt_meta_dominates(meta, self.adds[element]):
            self.adds[element] = meta

    def remove(
        self,
        element: str,
        timestamp: Optional[float] = None,
        node_id: str = "",
        precision: float = 1.0,
        confidence: float = 1.0,
        vector_clock: Optional[Union[CompactVectorClock, int]] = None,
    ) -> None:
        """Removes an element with Friston precision potential, vector clock, timestamp, and node_id."""
        ts = time.time() if timestamp is None else float(timestamp)
        prec = float(precision)
        conf = float(confidence)
        lambda_pot = prec * conf
        clock_rank = 0
        if vector_clock is not None:
            if isinstance(vector_clock, CompactVectorClock):
                clock_rank = sum(vector_clock.clocks.values())
            else:
                clock_rank = int(vector_clock)
        meta = (float(lambda_pot), int(clock_rank), float(ts), str(node_id))
        if element not in self.removes or _crdt_meta_dominates(meta, self.removes[element]):
            self.removes[element] = meta

    def lookup(self, element: str) -> bool:
        """Determines if the element is currently present using Friston precision partial order."""
        if element not in self.adds:
            return False
        if element not in self.removes:
            return True

        add_meta = self.adds[element]
        rem_meta = self.removes[element]
        return _crdt_meta_dominates(add_meta, rem_meta)

    def merge(self, other: CompactLWWSet) -> CompactLWWSet:
        """Merges with another LWWElementSet monotonically using Friston precision partial order."""
        new_adds = dict(self.adds)
        for elem, meta in other.adds.items():
            if elem not in new_adds or _crdt_meta_dominates(meta, new_adds[elem]):
                new_adds[elem] = meta

        new_removes = dict(self.removes)
        for elem, meta in other.removes.items():
            if elem not in new_removes or _crdt_meta_dominates(meta, new_removes[elem]):
                new_removes[elem] = meta

        self.adds = new_adds
        self.removes = new_removes
        return CompactLWWSet(
            adds=new_adds,
            removes=new_removes,
            tombstone_ttl_seconds=self.tombstone_ttl_seconds,
        )

    def model_dump(self) -> Dict[str, Any]:
        """Canonical dictionary serialization."""
        return {
            "adds": {k: list(v) for k, v in self.adds.items()},
            "removes": {k: list(v) for k, v in self.removes.items()},
            "tombstone_ttl_seconds": self.tombstone_ttl_seconds,
        }

    def model_dump_json(self) -> str:
        """Canonical JSON string serialization."""
        return json.dumps(self.model_dump())

    @classmethod
    def model_validate(cls, data: Any) -> CompactLWWSet:
        """Canonical validation from dict or instance."""
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            return cls()

        raw_adds = data.get("adds", {})
        raw_removes = data.get("removes", {})
        parsed_adds: Dict[str, Tuple[Any, ...]] = {}
        for k, v in raw_adds.items():
            if isinstance(v, (list, tuple)):
                parsed_adds[k] = _normalize_crdt_meta(v)

        parsed_removes: Dict[str, Tuple[Any, ...]] = {}
        for k, v in raw_removes.items():
            if isinstance(v, (list, tuple)):
                parsed_removes[k] = _normalize_crdt_meta(v)

        ttl = data.get("tombstone_ttl_seconds")
        return cls(
            adds=parsed_adds,
            removes=parsed_removes,
            tombstone_ttl_seconds=float(ttl) if ttl is not None else None,
        )

    @classmethod
    def model_validate_json(cls, json_str: Union[str, bytes]) -> CompactLWWSet:
        """Canonical deserialization from JSON."""
        return cls.model_validate(json.loads(json_str))


# Canonical alias
LWWElementSet = CompactLWWSet


@dataclass(slots=True)
class SyncDelta:
    """Payload delta representing a state mutation between nodes."""

    source_node: str
    vector_clock: VectorClock = field(default_factory=VectorClock)
    payload: Dict[str, Any] = field(default_factory=dict)
    encrypted_payload: Optional[bytes] = None

    def model_dump(self) -> Dict[str, Any]:
        """Canonical dictionary serialization."""
        d: Dict[str, Any] = {
            "source_node": self.source_node,
            "vector_clock": self.vector_clock.model_dump(),
            "payload": self.payload,
        }
        if self.encrypted_payload is not None:
            d["encrypted_payload"] = self.encrypted_payload.hex()
        return d

    def model_dump_json(self) -> str:
        """Canonical JSON string serialization."""
        return json.dumps(self.model_dump())

    @classmethod
    def model_validate(cls, data: Any) -> SyncDelta:
        """Canonical validation from dictionary."""
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict for SyncDelta validation, got {type(data)}")

        vc_data = data.get("vector_clock", {})
        vc = VectorClock.model_validate(vc_data)
        enc = data.get("encrypted_payload")
        enc_bytes = bytes.fromhex(enc) if isinstance(enc, str) else enc

        return cls(
            source_node=str(data.get("source_node", "")),
            vector_clock=vc,
            payload=data.get("payload", {}),
            encrypted_payload=enc_bytes,
        )

    @classmethod
    def model_validate_json(cls, json_str: Union[str, bytes]) -> SyncDelta:
        """Canonical deserialization from JSON string."""
        return cls.model_validate(json.loads(json_str))


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

    def auto_gc(self, now: Optional[float] = None) -> int:
        """Sweeps through all managed element sets and purges expired tombstones."""
        total_purged = 0
        for elem_set in self.state.values():
            total_purged += elem_set.auto_gc(now)
        return total_purged

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
        self.auto_gc()
        return self.clock
