"""V124 (F-039) T1 — `what_if` must consult the RAM causal engine first.

Audit finding: daemon's ``_handle_what_if`` delegated straight to
``server/causal_query.execute_what_if_rpc``, whose gate structure lets a
disk-only Kùzu presence check short-circuit the live ``RetroCausalEngine``
instance held in RAM (``daemon.causal_engine``). The fix makes the handler:
  1. consult ``self.causal_engine.causal_what_if`` FIRST (RAM path) when an
     engine exists;
  2. fall back to the disk path (causal_query) when the engine is missing OR
     raises — the fallback must not re-invoke the broken engine.

Doctrine: tmp_path fixtures only; no socket server is started; zero writes
to ~/.hermes/atlas/*.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from atlas_memory.server.atlas_daemon import AtlasDaemon


class RamGraph:
    """RAM (NetworkX-backed) graph: node present in memory, no disk bridge."""

    def __init__(self) -> None:
        self.nx_graph: Dict[str, Any] = {"X": {}}  # marker: RAM graph


class DiskGraph:
    """Disk (Kùzu) graph with the historic disk-only accessors."""

    def __init__(self, on_disk: bool = True) -> None:
        self.on_disk = on_disk
        self.relations_calls: int = 0

    def entity_exists(self, name: str) -> bool:  # on-disk presence gate
        return self.on_disk

    def get_relations_on_disk(self, entity: str, limit: int = 20) -> List[Dict[str, Any]]:
        self.relations_calls += 1
        return [
            {
                "subject": entity,
                "predicate": "depends_on",
                "object": f"{entity}_disk_child",
                "confidence": 0.5,
            }
        ]


class StubEngine:
    """Stub of RetroCausalEngine with a call counter."""

    def __init__(self, graph: Any, raise_: Optional[Exception] = None) -> None:
        self.graph = graph
        self.causal_what_if_calls: int = 0
        self.raise_ = raise_
        self.last_kwargs: Dict[str, Any] = {}

    async def causal_what_if(self, entity: str, action: str, depth: int = 2) -> List[Dict[str, Any]]:
        self.causal_what_if_calls += 1
        self.last_kwargs = {"entity": entity, "action": action, "depth": depth}
        if self.raise_ is not None:
            raise self.raise_
        return [
            {
                "source_entity": entity,
                "simulated_action": action,
                "affected_target": f"{entity}_engine_child",
                "cumulative_confidence": 0.9,
            }
        ]


def _make_daemon(tmp_path: Path) -> AtlasDaemon:
    return AtlasDaemon(
        socket_path=tmp_path / "v124t1.sock",
        pid_path=tmp_path / "v124t1.pid",
        engine=None,
    )


@pytest.mark.asyncio
async def test_what_if_uses_ram_engine_first(tmp_path: Path) -> None:
    """Engine present -> RAM path wins even when the disk gate misses the entity.

    RED premise (old code): execute_what_if_rpc checks entity existence on
    ``graph_client`` (disk) and returns ``entity_not_found`` WITHOUT ever
    touching ``self.causal_engine`` — an entity that lives ONLY in the RAM
    engine's graph was invisible until this fix.
    """
    daemon = _make_daemon(tmp_path)
    disk = DiskGraph(on_disk=False)
    daemon.graph_client = disk
    engine = StubEngine(graph=RamGraph())
    daemon.causal_engine = engine

    resp = await daemon._handle_what_if({"entity": "X", "action": "restart", "depth": 3})

    assert engine.causal_what_if_calls == 1
    assert engine.last_kwargs == {"entity": "X", "action": "restart", "depth": 3}
    assert resp["status"] == "ok"
    assert resp["entity"] == "X"
    assert resp["action"] == "restart"
    assert resp["paths_count"] == 1
    assert resp["causal_paths"][0]["affected_target"] == "X_engine_child"


@pytest.mark.asyncio
async def test_what_if_engine_exception_falls_back_to_disk(tmp_path: Path) -> None:
    """Engine raises -> disk response, engine tried exactly once."""
    daemon = _make_daemon(tmp_path)
    disk = DiskGraph()
    daemon.graph_client = disk
    engine = StubEngine(graph=RamGraph(), raise_=RuntimeError("RAM unavailable"))
    daemon.causal_engine = engine

    resp = await daemon._handle_what_if({"entity": "X", "action": "restart"})

    assert engine.causal_what_if_calls == 1
    assert resp["status"] == "ok"
    assert resp["source"] == "kuzu_disk"
    assert resp["paths_count"] == 1
    assert resp["causal_paths"][0]["affected_target"] == "X_disk_child"
    assert disk.relations_calls == 1


@pytest.mark.asyncio
async def test_what_if_without_engine_uses_disk_path(tmp_path: Path) -> None:
    """No engine at all -> unchanged disk behavior (regression guard)."""
    daemon = _make_daemon(tmp_path)
    disk = DiskGraph()
    daemon.graph_client = disk
    daemon.causal_engine = None

    resp = await daemon._handle_what_if({"entity": "X", "action": "restart"})

    assert resp["status"] == "ok"
    assert resp["source"] == "kuzu_disk"
    assert disk.relations_calls == 1


@pytest.mark.asyncio
async def test_what_if_ram_empty_paths_falls_back_to_disk_fill(tmp_path: Path) -> None:
    """Engine returning [] must keep the V72.2 disk-fill semantics."""
    class EmptyEngine(StubEngine):
        async def causal_what_if(self, entity, action, depth=2):  # noqa: ANN001, ANN003
            self.causal_what_if_calls += 1
            return []

    daemon = _make_daemon(tmp_path)
    disk = DiskGraph(on_disk=True)
    daemon.graph_client = disk
    daemon.causal_engine = EmptyEngine(graph=RamGraph())

    resp = await daemon._handle_what_if({"entity": "X", "action": "restart"})

    assert resp["status"] == "ok"
    assert resp["source"] == "kuzu_disk"
    assert resp["paths_count"] == 1
    # Broken-engine re-invocation guard: the empty engine must not be asked twice.
    assert daemon.causal_engine.causal_what_if_calls == 1
    # Canonical ok envelope keys must be present even on the fill path.
    assert resp["causal_path"] is not None


@pytest.mark.asyncio
async def test_what_if_ram_empty_and_absent_on_disk_is_entity_not_found(tmp_path: Path) -> None:
    """Empty RAM + disk gate says 'no' -> honest entity_not_found (no silent zero paths)."""
    class EmptyEngine(StubEngine):
        async def causal_what_if(self, entity, action, depth=2):  # noqa: ANN001, ANN003
            self.causal_what_if_calls += 1
            return []

    daemon = _make_daemon(tmp_path)
    daemon.graph_client = DiskGraph(on_disk=False)
    daemon.causal_engine = EmptyEngine(graph=RamGraph())

    resp = await daemon._handle_what_if({"entity": "ghost", "action": "restart"})

    assert resp["status"] == "entity_not_found"
    assert resp["paths_count"] == 0
