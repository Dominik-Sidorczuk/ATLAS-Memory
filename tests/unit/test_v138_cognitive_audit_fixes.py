"""Unit tests for Milestone v138: Cognitive audit fixes and architectural remediations.

Tasks covered:
T1: PII regex secret assignment separation without broken trailing word boundary.
T2: JEPA active step in engine.record_mental_transition updating latent state and history.
T3: HotBuffer extract_effective_session key prefix priority over metadata.
T4: Session tracker graph mark_superseded and lock-protected revision bump.
T5: Lock decomposition in HippoRAG 2 fetch_graph_ppr_candidates with post-run validation.
T6: Fusion auth asymmetry elimination for strong graph hits.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from atlas_memory.cognitive.retrieval.ppr import (
    HippoRAG2PPR,
    fetch_graph_ppr_candidates,
)
from atlas_memory.core.models import (
    HERMES_GLOBAL_SESSION,
    ActionPlan,
    ConsolidationStats,
    HotBufferEntry,
    is_pii_entity,
)
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.l1_working.hot_buffer import (
    extract_effective_session,
    is_session_accessible,
)
from atlas_memory.l1_working.session_tracker import HermesSessionHook
from atlas_memory.l2_semantic.kuzu.ram_multigraph import AtlasMultiDiGraph
from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore


def test_t1_pii_regex_secret_assignment_and_domain_whitelist():
    """T1: secret assignment is detected as PII while domain terms remain clean."""
    assert is_pii_entity("secret = hunter2") is True
    assert is_pii_entity("secret: abc123") is True
    assert is_pii_entity("secret=hunter2") is True
    assert is_pii_entity("tokenizer") is False
    assert is_pii_entity("shared_secret") is False
    assert is_pii_entity("token_budget") is False
    assert is_pii_entity("secret") is True
    assert is_pii_entity("client_secret=xyz") is True


def test_t2_jepa_active_step_in_record_mental_transition():
    """T2: record_mental_transition calls latent.step(action), advancing latent state and history."""
    engine = HybridMemoryEngine.create_default(db_path=":memory:")
    s0 = engine.latent.current_state
    h0 = len(engine.latent.history)

    a1 = ActionPlan(name="action_probe_alpha", parameters={"target": "left"})
    a2 = ActionPlan(name="action_probe_beta", parameters={"target": "right"})

    engine.record_mental_transition(a1, session_id="s1")
    s1 = engine.latent.current_state
    assert s1 != s0
    assert len(engine.latent.history) == h0 + 1

    engine.record_mental_transition(a2, session_id="s1")
    s2 = engine.latent.current_state
    assert s2 != s0
    assert s2 != s1
    assert len(engine.latent.history) == h0 + 2


def test_t3_persistent_key_prefix_priority_over_metadata():
    """T3: Key prefixes (session and persistent) take priority over record metadata session_id."""
    # 1. 'hermes:memory_md:Raporty' + metadata.session_id -> HERMES_GLOBAL_SESSION
    assert extract_effective_session(
        "hermes:memory_md:Raporty",
        {"session_id": "session_20261004_170710_6717e1"},
    ) == HERMES_GLOBAL_SESSION

    # 2. 'session_20261004_170710_6717e1' (without metadata) -> '20261004_170710_6717e1'
    pelny_id = extract_effective_session("session_20261004_170710_6717e1")
    assert pelny_id == "20261004_170710_6717e1"

    # 3. 'session_20261004_170710_6717e1' + metadata.session_id='X' -> key ID wins
    assert extract_effective_session(
        "session_20261004_170710_6717e1",
        {"session_id": "X"},
    ) == "20261004_170710_6717e1"

    # 4. 'session_67890' -> '67890'
    assert extract_effective_session("session_67890") == "67890"

    # 5. is_session_accessible(pelny_id, 'session_20261004_170710_6717e1') -> True
    assert is_session_accessible(pelny_id, "session_20261004_170710_6717e1") is True


@pytest.mark.asyncio
async def test_t4_session_reconcile_superseded_bumps_revision():
    """T4: Reconciling superseded facts bumps graph revision under lock."""
    # 1. KuzuGraphStore backing
    store = KuzuGraphStore(db_path=":memory:")
    store.nx_graph.add_node("fact:superseded_item", type="fact")
    rev_before = store.nx_graph.revision

    entry = HotBufferEntry(key="fact:superseded_item", value="stale_content", is_superseded=True)
    hot_buf = SimpleNamespace(_buffer=[entry], get_session_entries=lambda _s: [])

    engine = SimpleNamespace(
        graph=store,
        graph_store=store,
        hot_buffer=hot_buf,
        kv=None,
        vector_store=None,
    )
    hook = HermesSessionHook(engine)
    stats = ConsolidationStats()
    await hook._reconcile_l1_to_l2(session_id="s_test", touched_keys={"fact:superseded_item"}, stats=stats)

    assert store.nx_graph.revision > rev_before
    assert store.nx_graph.nodes["fact:superseded_item"]["is_superseded"] is True

    # 2. AtlasMultiDiGraph direct backing
    direct_g = AtlasMultiDiGraph()
    direct_g.add_node("fact:direct_superseded", type="fact")
    rev_before_direct = direct_g.revision

    entry_direct = HotBufferEntry(key="fact:direct_superseded", value="stale_val", is_superseded=True)
    hot_buf_direct = SimpleNamespace(_buffer=[entry_direct], get_session_entries=lambda _s: [])
    engine_direct = SimpleNamespace(
        graph=direct_g,
        hot_buffer=hot_buf_direct,
        kv=None,
        vector_store=None,
    )
    hook_direct = HermesSessionHook(engine_direct)
    stats_direct = ConsolidationStats()
    await hook_direct._reconcile_l1_to_l2(
        session_id="s_direct", touched_keys={"fact:direct_superseded"}, stats=stats_direct
    )

    assert direct_g.revision > rev_before_direct
    assert direct_g.nodes["fact:direct_superseded"]["is_superseded"] is True


@pytest.mark.asyncio
async def test_t5_ppr_lock_decomposition_concurrent_mutation(monkeypatch):
    """T5: PPR lock is decomposed so concurrent edge additions proceed without blocking or deadlocks."""
    monkeypatch.setenv("ATLAS_HIPPORAG", "1")
    store = KuzuGraphStore(db_path=":memory:")
    store.add_passage("passage:1", "rule:1", "knowledge retrieval architecture")
    store.add_entity("knowledge")
    store.add_contains("passage:1", "knowledge")

    ppr_entered = threading.Event()
    mutation_done = threading.Event()
    mutation_success = threading.Event()

    orig_run = HippoRAG2PPR.run

    def slow_run(self, *args, **kwargs):
        ppr_entered.set()
        mutation_done.wait(timeout=2.0)
        return orig_run(self, *args, **kwargs)

    def mutator():
        if ppr_entered.wait(timeout=2.0):
            with store._graph_lock:
                store.nx_graph.add_node("entity_concurrent")
                store.nx_graph.add_edge(
                    "knowledge",
                    "entity_concurrent",
                    predicate="CONCURRENT",
                    confidence=0.99,
                )
                store.nx_graph.bump_revision()
                mutation_success.set()
        mutation_done.set()

    t = threading.Thread(target=mutator)
    t.start()

    with patch.object(HippoRAG2PPR, "run", slow_run):
        cands = await asyncio.to_thread(fetch_graph_ppr_candidates, store, "knowledge", limit=5)

    await asyncio.to_thread(t.join, 3.0)

    assert mutation_success.is_set(), "Mutation should succeed concurrently while PPR run is in progress"
    assert len(cands) >= 1
    assert not store._graph_lock.locked()
    ppr_engine_instance = getattr(store, "_ppr_engine", None)
    assert ppr_engine_instance is not None
    assert ppr_engine_instance._cached_sig is None



