"""V124 (F-041) T3 — KuzuGraphStore.create_entity must match the protocol contract.

AUDIT: ``l2_semantic/kuzu_graph.py`` declares
``create_entity(self, name, kind="entity")`` while the
``DualNodeGraphStoreProtocol`` contract (``core/protocols.py``) requires
``create_entity(self, name, entity_type="entity")``. Keyword-contract callers
(``create_entity(name="X", entity_type=...)``) hit TypeError.

Repo-wide call-site review before the fix:
- src: l2_semantic/kuzu/schema_driver.py:179 — positional call (safe),
- tests: tests/unit/test_write_path_dual_node.py:32 — kwarg ``kind=`` (fixed
  in the same commit), no other callers of create_entity with ``kind=`` exist.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from atlas_memory.core.protocols import DualNodeGraphStoreProtocol
from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore


@pytest.fixture()
def store(tmp_path: Path) -> KuzuGraphStore:
    db_path = tmp_path / "v124t3_kuzu" / "kuzu_db"
    return KuzuGraphStore(db_path=str(db_path))


def test_kuzu_store_is_dual_node_graph_store_protocol(store: KuzuGraphStore) -> None:
    """KuzuGraphStore must satisfy the DualNodeGraphStoreProtocol contract."""
    assert isinstance(store, DualNodeGraphStoreProtocol)


def test_create_entity_contract_signature(store: KuzuGraphStore) -> None:
    """Signature must expose entity_type (protocol), NOT the legacy kind kwarg."""
    params = inspect.signature(store.create_entity).parameters
    assert "name" in params
    assert "entity_type" in params
    assert "kind" not in params


def test_create_entity_entity_type_kwarg(store: KuzuGraphStore) -> None:
    """Contract kwarg call: create_entity(name='X', entity_type='tool')."""
    store.create_entity(name="X_v124t3", entity_type="tool")
    assert "X_v124t3" in store.nx_graph
    assert store.nx_graph.nodes["X_v124t3"].get("type") == "tool"


def test_create_entity_positional_call_survives(store: KuzuGraphStore) -> None:
    """Old positional call style must not break after the rename."""
    store.create_entity("Y_v124t3", "entity")
    assert "Y_v124t3" in store.nx_graph
    assert store.nx_graph.nodes["Y_v124t3"].get("type") == "entity"
