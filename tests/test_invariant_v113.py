"""V113 (F-029-class): Invariant enforcement tests — `InvariantViolation` instead of `assert`.

Verifies that production invariants survive `python -O` / `PYTHONOPTIMIZE=2`:
  T1  schema_driver W1 entity/relation-count invariants raise InvariantViolation
      with structured fields (module, name, observed, expected, hint) when counts
      decrease during `migrate_dual_node_schema_and_populate`.
  T2  `VerifiedKVStore.ensure_connection()` (public API) raises InvariantViolation
      when `_conn` is None (daemon `_handle_delete` legacy else-path).
  T3  PPR igraph backend with a missing `ig_graph` raises InvariantViolation
      (name='ppr_igraph_missing') instead of an optimized-out assert.
  T4  `-O` gate: the patched functions contain no bare `assert`, and a subprocess
      running under PYTHONOPTIMIZE=2 still raises InvariantViolation.
  T5  Regression: happy paths (migration, ensure_connection, PPR numpy backend)
      still pass unchanged.

AUDYT=FIX doctrine: these tests gate enforcement only; no production behavior
besides assert→raise was altered.
"""
from __future__ import annotations

import inspect
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

from atlas_memory.cognitive.retrieval import ppr as ppr_module
from atlas_memory.cognitive.retrieval.ppr import HippoRAG2PPR
from atlas_memory.core.exceptions import InvariantViolation
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.server.atlas_daemon import AtlasDaemon

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"

_PASSAGES = [
    {
        "key": "hermes:memory_md:V113 probe",
        "text": "Demon ATLAS używa gniazda UDS do transportu RPC i bazy Kuzu dla grafu.",
        "gist": "ATLAS daemon UDS Kuzu",
    },
    {
        "key": "fact:atlas:v113_arch",
        "text": "Architektura warstw L2 opiera się na NetworkX oraz SQLite verified KV store.",
    },
]


def _seed_store(tmp_path: Path, name: str) -> KuzuGraphStore:
    store = KuzuGraphStore(db_path=str(tmp_path / name))
    store.add_entity("atlas_daemon", entity_type="service")
    store.add_entity("uds_socket", entity_type="transport")
    store.add_relation("atlas_daemon", "listens_on", "uds_socket", confidence=0.9)
    return store


# ---------------------------------------------------------------------------
# T1 — schema_driver W1 entity/relation invariants raise InvariantViolation
# ---------------------------------------------------------------------------


def test_t1_entity_count_decrease_raises_invariant_violation(tmp_path: Path, monkeypatch) -> None:
    store = _seed_store(tmp_path, "kuzu_t1_ent")
    try:
        # Deterministic simulation of a migration that loses Entity nodes: the
        # invariant snapshot (`entities_after`) is taken right after
        # `build_synonym_edges`, so that is the last intercept point before the gate.
        def fake_build_synonym_edges(entities, **kw):
            non_passage = [
                n for n, d in store.nx_graph.nodes(data=True) if d.get("type") != "passage"
            ]
            store.nx_graph.remove_nodes_from(non_passage)
            return []

        monkeypatch.setattr(
            "atlas_memory.l2_semantic.kuzu.schema_driver.build_synonym_edges",
            fake_build_synonym_edges,
        )

        with pytest.raises(InvariantViolation) as excinfo:
            store.migrate_dual_node_schema_and_populate(_PASSAGES)

        exc = excinfo.value
        assert exc.module == "l2_semantic.kuzu.schema_driver"
        assert exc.name == "w1_entity_count_monotonic"
        assert exc.observed < exc.expected
        assert "Entity count" in exc.hint or "Entity" in str(exc)
    finally:
        store.close()


def test_t1b_relation_count_decrease_raises_invariant_violation(tmp_path: Path, monkeypatch) -> None:
    store = _seed_store(tmp_path, "kuzu_t1_rel")
    try:
        orig_add_passage = store.add_passage

        def sabotage(passage_id: str, source_key: str, text: str, *args, **kwargs) -> None:
            # Drop the one RELATION edge; pad an entity so the entity gate passes first.
            if store.nx_graph.has_edge("atlas_daemon", "uds_socket", "listens_on"):
                store.nx_graph.remove_edge("atlas_daemon", "uds_socket", key="listens_on")
            store.nx_graph.add_node("v113_pad_entity", type="entity")
            orig_add_passage(passage_id, source_key, text)

        monkeypatch.setattr(store, "add_passage", sabotage)

        with pytest.raises(InvariantViolation) as excinfo:
            store.migrate_dual_node_schema_and_populate(_PASSAGES)

        exc = excinfo.value
        assert exc.name == "w1_relation_count_monotonic"
        assert exc.observed < exc.expected
    finally:
        store.close()


# ---------------------------------------------------------------------------
# T2 — ensure_connection() public API + daemon legacy delete path
# ---------------------------------------------------------------------------


class _LegacyKVStub:
    """KV store WITHOUT `delete_state` and WITHOUT a live `_conn` — the daemon's
    legacy else-branch surface. Uses the real `ensure_connection` implementation."""

    _conn = None
    ensure_connection = VerifiedKVStore.ensure_connection


def test_t2_ensure_connection_raises_when_conn_closed(tmp_path: Path) -> None:
    kv = VerifiedKVStore(db_path=str(tmp_path / "atlas.db"))
    assert kv.ensure_connection() is kv._conn  # healthy handle passthrough
    kv._conn = None
    with pytest.raises(InvariantViolation) as excinfo:
        kv.ensure_connection()
    exc = excinfo.value
    assert exc.module == "l2_semantic.kv_store"
    assert exc.name == "kv_connection_open"
    assert exc.observed == 0 and exc.expected == 1
    assert exc.hint


@pytest.mark.asyncio
async def test_t2_daemon_delete_legacy_path_raises_invariant(tmp_path: Path) -> None:
    engine = HybridMemoryEngine.create_default(
        db_path=str(tmp_path / "atlas.db"),
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    daemon = AtlasDaemon(engine=engine)
    engine.kv = _LegacyKVStub()  # no delete_state → else-branch → ensure_connection
    with pytest.raises(InvariantViolation) as excinfo:
        await daemon._handle_delete({"key": "v113:orphan_probe"})
    assert excinfo.value.name == "kv_connection_open"


# ---------------------------------------------------------------------------
# T3 — PPR igraph backend missing graph raises InvariantViolation
# ---------------------------------------------------------------------------


def test_t3_ppr_missing_igraph_raises_invariant(monkeypatch) -> None:
    ppr = HippoRAG2PPR()  # default backend: igraph
    graph = nx.MultiDiGraph()
    graph.add_edge("alpha", "beta", predicate="depends_on")
    store = SimpleNamespace(nx_graph=graph)

    monkeypatch.setattr(
        ppr,
        "_build_graph_structures",
        lambda _gs: (["alpha", "beta"], {"alpha": 0, "beta": 1}, {(0, 1): 1.0}, None),
    )
    monkeypatch.setattr(
        ppr_module, "score_query_to_triples",
        lambda **kw: ({"alpha": 1.0}, {}, {}),
    )
    monkeypatch.setattr(
        ppr_module, "expand_associative_seeds_2hop",
        lambda **kw: {"alpha": 1.0},
    )

    with pytest.raises(InvariantViolation) as excinfo:
        ppr.run("alpha", store)

    exc = excinfo.value
    assert exc.module == "cognitive.retrieval.ppr"
    assert exc.name == "ppr_igraph_missing"
    assert "igraph" in exc.hint.lower()


# ---------------------------------------------------------------------------
# T4 — `python -O` gate: no bare asserts left; raise still fires with -O
# ---------------------------------------------------------------------------

_BARE_ASSERT = re.compile(r"(?m)^[ \t]*assert[ \t]")


def test_t4_patched_functions_contain_no_bare_asserts() -> None:
    from atlas_memory.l2_semantic.kuzu.schema_driver import KuzuSchemaDriverMixin

    for func in (
        KuzuSchemaDriverMixin.migrate_dual_node_schema_and_populate,
        AtlasDaemon._handle_delete,
        VerifiedKVStore.ensure_connection,
        HippoRAG2PPR.run,
    ):
        src = inspect.getsource(func)
        assert not _BARE_ASSERT.search(src), (
            f"{func.__qualname__} still contains a bare `assert` — stripped under -O"
        )


def test_t4_invariant_violation_survives_python_o() -> None:
    script = (
        "import sys\n"
        "assert not __debug__, 'subprocess must run with -O'\n"
        "from atlas_memory.l2_semantic.kv_store import VerifiedKVStore\n"
        "from atlas_memory.core.exceptions import InvariantViolation\n"
        "kv = VerifiedKVStore(db_path=':memory:')\n"
        "kv._conn = None\n"
        "try:\n"
        "    kv.ensure_connection()\n"
        "except InvariantViolation as exc:\n"
        "    assert exc.name == 'kv_connection_open', exc\n"
        "    print('V113-O-STILL-RAISES')\n"
        "else:\n"
        "    sys.exit(1)\n"
    )
    res = subprocess.run(
        [sys.executable, "-OO", "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        env={"PYTHONPATH": str(SRC_ROOT), "PATH": "/usr/bin:/bin"},
    )
    assert res.returncode == 0, f"subprocess failed: {res.stderr[-500:]}"
    assert "V113-O-STILL-RAISES" in res.stdout


# ---------------------------------------------------------------------------
# T5 — regression: happy paths unchanged
# ---------------------------------------------------------------------------


def test_t5_happy_migration_and_kv_and_ppr_pass(tmp_path: Path) -> None:
    store = _seed_store(tmp_path, "kuzu_t5")
    try:
        report = store.migrate_dual_node_schema_and_populate(_PASSAGES)
        assert report["entities_after"] >= report["entities_before"]
        assert report["relations_after"] >= report["relations_before"]
        assert report["passages_added"] == 2

        ppr = HippoRAG2PPR()
        result = ppr.run("atlas daemon", store, backend="numpy")
        assert result is not None
    finally:
        store.close()

    kv = VerifiedKVStore(db_path=str(tmp_path / "atlas_t5.db"))
    conn = kv.ensure_connection()
    with conn:
        conn.execute("SELECT 1")
    assert conn.execute("SELECT 1").fetchone()[0] == 1
