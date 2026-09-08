"""
Unit Tests for Causal Reasoning & Topological Graph Diffusion.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import networkx as nx
import pytest

from atlas_memory.causal.retro_causal_edge import (
    EMAIL_PATTERN,
    RetroCausalEngine,
    is_pii_entity,
)
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.ingest.mnemosyne_ingest import MnemosyneIngestEngine
from atlas_memory.l1_working.jepa_latent import JEPALatentBuffer
from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore
from atlas_memory.models import MemoryRecord
from atlas_memory.server.atlas_daemon import AtlasDaemon


class MockGraphForCausal:
    """Mock grafu relacji dla testów przyczynowo-skutkowych."""
    def __init__(self):
        self.relations = []

    def add(self, subj: str, pred: str, obj: str, conf: float = 1.0):
        self.relations.append({
            "subject": subj,
            "predicate": pred,
            "object": obj,
            "confidence": conf,
        })

    async def get_subgraph_relations(self, entities, max_depth=2):
        return {"relations": self.relations}


@pytest.mark.asyncio
async def test_causal_what_if_1_hop_dependency():
    g = MockGraphForCausal()
    g.add("entity_nas_01", "hosts", "NFS_Share", 0.95)
    engine = RetroCausalEngine(graph_client=g)

    paths = await engine.causal_what_if("entity_nas_01", "restart_service", depth=1)
    assert len(paths) == 1
    assert paths[0].affected_target == "NFS_Share"
    assert paths[0].cumulative_confidence == 0.95
    assert paths[0].risk_level == "CRITICAL"


@pytest.mark.asyncio
async def test_causal_what_if_2_hop_propagation():
    g = MockGraphForCausal()
    g.add("entity_nas_01", "hosts", "NFS_Share", 0.9)
    g.add("NFS_Share", "mounts", "DatabaseCluster", 0.8)
    engine = RetroCausalEngine(graph_client=g)

    paths = await engine.causal_what_if("entity_nas_01", "zmien_MTU", depth=2)
    assert len(paths) == 2
    targets = [p.affected_target for p in paths]
    assert "NFS_Share" in targets
    assert "DatabaseCluster" in targets


@pytest.mark.asyncio
async def test_confidence_multiplication_propagation():
    g = MockGraphForCausal()
    g.add("A", "links_to", "B", 0.9)
    g.add("B", "links_to", "C", 0.8)
    engine = RetroCausalEngine(graph_client=g)

    paths = await engine.causal_what_if("A", "modify_A", depth=2)
    path_c = next(p for p in paths if p.affected_target == "C")
    assert path_c.cumulative_confidence == pytest.approx(0.72, rel=1e-3)


@pytest.mark.asyncio
async def test_cycle_avoidance_in_causal_graph():
    g = MockGraphForCausal()
    g.add("ServiceA", "depends_on", "ServiceB", 0.9)
    g.add("ServiceB", "depends_on", "ServiceA", 0.9)  # Cykl
    engine = RetroCausalEngine(graph_client=g)

    paths = await engine.causal_what_if("ServiceA", "upgrade", depth=3)
    # Powinno zakończyć eksplorację bez pętli nieskończonej
    assert len(paths) >= 1
    assert all(len(p.steps) <= 2 for p in paths)


@pytest.mark.asyncio
async def test_critical_predicate_risk_classification():
    g = MockGraphForCausal()
    g.add("ServerX", "powers", "MainSwitch", 0.95)
    g.add("ServerX", "documented_in", "WikiPage", 0.95)
    engine = RetroCausalEngine(graph_client=g)

    paths = await engine.causal_what_if("ServerX", "poweroff", depth=1)
    p_switch = next(p for p in paths if p.affected_target == "MainSwitch")
    p_wiki = next(p for p in paths if p.affected_target == "WikiPage")

    assert p_switch.risk_level == "CRITICAL"
    assert p_wiki.risk_level in ["LOW", "MODERATE"]


@pytest.mark.asyncio
async def test_jepa_latent_divergence_integration():
    g = MockGraphForCausal()
    g.add("entity_nas_01", "depends_on", "Router01", 0.9)
    latent = JEPALatentBuffer(state_dim=32, action_dim=16)
    engine = RetroCausalEngine(graph_client=g, latent_buffer=latent)

    paths = await engine.causal_what_if("entity_nas_01", "reboot_network", depth=1)
    assert len(paths) == 1
    assert paths[0].jepa_latent_divergence is not None
    assert paths[0].jepa_latent_divergence > 0.0


@pytest.mark.asyncio
async def test_evaluate_what_if_safety_score():
    g = MockGraphForCausal()
    g.add("NAS", "depends_on", "PowerGrid", 0.99)
    g.add("NAS", "critical_for", "ProdApp", 0.95)
    engine = RetroCausalEngine(graph_client=g)

    res = await engine.evaluate_what_if("NAS", "pull_cable", depth=1)
    assert res.highest_risk_target in ["PowerGrid", "ProdApp"]
    assert res.overall_safety_score < 0.6  # Niskie bezpieczeństwo z powodu krytycznych zależności


@pytest.mark.asyncio
async def test_empty_graph_graceful_handling():
    g = MockGraphForCausal()
    engine = RetroCausalEngine(graph_client=g)

    paths = await engine.causal_what_if("NonExistent", "act", depth=2)
    assert paths == []
    res = await engine.evaluate_what_if("NonExistent", "act", depth=2)
    assert res.overall_safety_score == 1.0


@pytest.mark.asyncio
async def test_unconnected_entity_handling():
    g = MockGraphForCausal()
    g.add("OtherA", "links_to", "OtherB", 1.0)
    engine = RetroCausalEngine(graph_client=g)

    paths = await engine.causal_what_if("IsolatedNode", "delete", depth=2)
    assert len(paths) == 0


@pytest.mark.asyncio
async def test_multi_branch_impact_sorting():
    g = MockGraphForCausal()
    g.add("Hub", "depends_on", "CriticalDB", 0.99)
    g.add("Hub", "links_to", "LogSink", 0.4)
    engine = RetroCausalEngine(graph_client=g)

    paths = await engine.causal_what_if("Hub", "test_action", depth=1)
    assert len(paths) == 2
    # Pierwszy element powinien mieć wyższy priorytet ryzyka
    assert paths[0].affected_target == "CriticalDB"
    assert paths[0].risk_level == "CRITICAL"

import pytest

from atlas_memory.causal.models import (
    DiffusionResult,
)


class MockGraphClient:
    """Mock graph store with adjacency list dictionary."""

    def __init__(self, relations=None, multi_hop_paths=None):
        self.relations = relations or []
        self.multi_hop_paths = multi_hop_paths or []

    async def get_subgraph_relations(self, entities, max_depth=2):
        return {"relations": self.relations}

    async def execute_multi_hop_cypher(self, source, target, max_hops=5):
        return self.multi_hop_paths


@pytest.mark.asyncio
async def test_diffusion_linear_chain():
    """A -> B -> C -> D, diffusion from A, decay 0.7 -> impact scores decrease exponentially."""
    relations = [
        {"subject": "A", "predicate": "leads_to", "object": "B", "confidence": 1.0},
        {"subject": "B", "predicate": "leads_to", "object": "C", "confidence": 1.0},
        {"subject": "C", "predicate": "leads_to", "object": "D", "confidence": 1.0},
    ]
    client = MockGraphClient(relations=relations)
    engine = RetroCausalEngine(graph_client=client)

    res = await engine.causal_diffusion_analysis("A", max_depth=3, decay_factor=0.7)

    assert isinstance(res, DiffusionResult)
    assert len(res.nodes) == 3
    node_map = {n.node_id: n for n in res.nodes}

    assert "B" in node_map
    assert "C" in node_map
    assert "D" in node_map

    # B: depth 1 -> 0.7^1 = 0.7
    # C: depth 2 -> 0.7^2 = 0.49
    # D: depth 3 -> 0.7^3 = 0.343
    assert node_map["B"].impact_score == pytest.approx(0.7, rel=1e-3)
    assert node_map["C"].impact_score == pytest.approx(0.49, rel=1e-3)
    assert node_map["D"].impact_score == pytest.approx(0.343, rel=1e-3)

    assert node_map["B"].impact_score > node_map["C"].impact_score > node_map["D"].impact_score
    assert res.total_impact == pytest.approx(0.7 + 0.49 + 0.343, rel=1e-3)
    assert res.max_depth_reached == 3


@pytest.mark.asyncio
async def test_diffusion_branching():
    """A -> [B, C], B -> D, C -> D -> D gets impact from both paths (sum)."""
    relations = [
        {"subject": "A", "predicate": "causes", "object": "B", "confidence": 1.0},
        {"subject": "A", "predicate": "causes", "object": "C", "confidence": 1.0},
        {"subject": "B", "predicate": "leads_to", "object": "D", "confidence": 1.0},
        {"subject": "C", "predicate": "leads_to", "object": "D", "confidence": 1.0},
    ]
    client = MockGraphClient(relations=relations)
    engine = RetroCausalEngine(graph_client=client)

    res = await engine.causal_diffusion_analysis("A", max_depth=3, decay_factor=0.7)

    node_map = {n.node_id: n for n in res.nodes}
    assert "D" in node_map
    # D reached via B (depth 2: 0.49) and via C (depth 2: 0.49) -> 0.49 + 0.49 = 0.98
    assert node_map["D"].impact_score == pytest.approx(0.98, rel=1e-3)
    assert node_map["B"].impact_score == pytest.approx(0.7, rel=1e-3)
    assert node_map["C"].impact_score == pytest.approx(0.7, rel=1e-3)


@pytest.mark.asyncio
async def test_diffusion_cycle_handling():
    """A -> B -> C -> A (cycle) -> terminates cleanly at max_depth without infinite loop."""
    relations = [
        {"subject": "A", "predicate": "rel", "object": "B", "confidence": 1.0},
        {"subject": "B", "predicate": "rel", "object": "C", "confidence": 1.0},
        {"subject": "C", "predicate": "rel", "object": "A", "confidence": 1.0},
        {"subject": "C", "predicate": "rel", "object": "D", "confidence": 1.0},
    ]
    client = MockGraphClient(relations=relations)
    engine = RetroCausalEngine(graph_client=client)

    res = await engine.causal_diffusion_analysis("A", max_depth=3, decay_factor=0.7)

    assert isinstance(res, DiffusionResult)
    assert res.max_depth_reached <= 3
    node_ids = {n.node_id for n in res.nodes}
    assert "B" in node_ids
    assert "C" in node_ids
    assert "D" in node_ids


@pytest.mark.asyncio
async def test_cpof_hub_detection():
    """Graph with central hub H -> CPoF severity > 0.8 for hub H."""
    relations = [
        {"subject": "Root", "predicate": "routes_to", "object": "Hub", "confidence": 1.0},
        {"subject": "Hub", "predicate": "powers", "object": "S1", "confidence": 1.0},
        {"subject": "Hub", "predicate": "powers", "object": "S2", "confidence": 1.0},
        {"subject": "Hub", "predicate": "powers", "object": "S3", "confidence": 1.0},
        {"subject": "Hub", "predicate": "powers", "object": "S4", "confidence": 1.0},
        {"subject": "Hub", "predicate": "powers", "object": "S5", "confidence": 1.0},
    ]
    client = MockGraphClient(relations=relations)
    engine = RetroCausalEngine(graph_client=client)

    cpofs = await engine.detect_cpof("Root")

    assert len(cpofs) >= 1
    top_cpof = cpofs[0]
    assert top_cpof.node_id == "Hub"
    assert top_cpof.severity > 0.8
    assert set(top_cpof.affected_nodes) == {"S1", "S2", "S3", "S4", "S5"}


@pytest.mark.asyncio
async def test_cpof_no_cpof_in_tree():
    """Balanced tree with no single point of failure -> CPoF severity < 0.5 for all nodes."""
    relations = [
        {"subject": "Root", "predicate": "rel", "object": "A", "confidence": 1.0},
        {"subject": "Root", "predicate": "rel", "object": "B", "confidence": 1.0},
        {"subject": "A", "predicate": "rel", "object": "A1", "confidence": 1.0},
        {"subject": "A", "predicate": "rel", "object": "A2", "confidence": 1.0},
        {"subject": "B", "predicate": "rel", "object": "B1", "confidence": 1.0},
        {"subject": "B", "predicate": "rel", "object": "B2", "confidence": 1.0},
    ]
    client = MockGraphClient(relations=relations)
    engine = RetroCausalEngine(graph_client=client)

    cpofs = await engine.detect_cpof("Root")

    # In balanced tree, removing A affects only A1, A2 (2/6 = 33.3% < 50%)
    assert len(cpofs) == 0


@pytest.mark.asyncio
async def test_multi_hop_cypher_simple():
    """A -> B -> C, query(A, C, max_hops=3) -> 1 path."""
    raw_mock_paths = [
        {
            "source": "A",
            "target": "C",
            "nodes": ["A", "B", "C"],
            "edges": [
                {"source": "A", "predicate": "depends_on", "target": "B", "confidence": 0.9},
                {"source": "B", "predicate": "depends_on", "target": "C", "confidence": 0.8},
            ],
        }
    ]
    client = MockGraphClient(multi_hop_paths=raw_mock_paths)
    engine = RetroCausalEngine(graph_client=client)

    paths = await engine.multi_hop_cypher_query("A", "C", max_hops=3)

    assert len(paths) == 1
    p = paths[0]
    assert p.source_entity == "A"
    assert p.affected_target == "C"
    assert len(p.steps) == 2
    assert p.cumulative_confidence == pytest.approx(0.72, rel=1e-3)


@pytest.mark.asyncio
async def test_multi_hop_cypher_exclude_cycles():
    """Exclude cyclic paths when exclude_cycles=True."""
    raw_mock_paths = [
        {
            "source": "A",
            "target": "C",
            "nodes": ["A", "B", "C"],
            "edges": [
                {"source": "A", "predicate": "leads_to", "target": "B", "confidence": 0.9},
                {"source": "B", "predicate": "leads_to", "target": "C", "confidence": 0.9},
            ],
        },
        {
            "source": "A",
            "target": "C",
            "nodes": ["A", "B", "C", "A", "B", "C"],
            "edges": [
                {"source": "A", "predicate": "leads_to", "target": "B", "confidence": 0.9},
                {"source": "B", "predicate": "leads_to", "target": "C", "confidence": 0.9},
                {"source": "C", "predicate": "leads_to", "target": "A", "confidence": 0.9},
                {"source": "A", "predicate": "leads_to", "target": "B", "confidence": 0.9},
                {"source": "B", "predicate": "leads_to", "target": "C", "confidence": 0.9},
            ],
        },
    ]
    client = MockGraphClient(multi_hop_paths=raw_mock_paths)
    engine = RetroCausalEngine(graph_client=client)

    paths = await engine.multi_hop_cypher_query("A", "C", max_hops=5, exclude_cycles=True)

    assert len(paths) == 1
    assert len(paths[0].steps) == 2


@pytest.mark.asyncio
async def test_multi_hop_cypher_max_hops_limit():
    """A -> B -> C -> D -> E (4 hops). Query with max_hops=2 returns 0 paths."""
    kuzu_store = KuzuGraphStore(db_path=None)
    # Add chain A -> B -> C -> D -> E
    nodes = ["A", "B", "C", "D", "E"]
    for i in range(len(nodes) - 1):
        rec = MemoryRecord(
            subject=nodes[i],
            predicate="leads_to",
            object=nodes[i + 1],
            confidence=1.0,
            timestamp=1000.0,
        )
        await kuzu_store.add_record(rec)

    engine = RetroCausalEngine(graph_client=kuzu_store)

    paths_limited = await engine.multi_hop_cypher_query("A", "E", max_hops=2)
    assert len(paths_limited) == 0

    paths_full = await engine.multi_hop_cypher_query("A", "E", max_hops=5)
    assert len(paths_full) == 1
    assert len(paths_full[0].steps) == 4


@pytest.mark.asyncio
async def test_v37_recalibrate_graph_with_annealer():
    """Test RetroCausalEngine graph auto-recalibration via CausalAnnealer."""
    class MockGraph:
        def __init__(self):
            self.relations = [
                {"subject": "A", "predicate": "depends_on", "object": "B", "confidence": 0.8},
                {"subject": "B", "predicate": "depends_on", "object": "C", "confidence": 0.9},
            ]
            self.updated = []

        async def get_subgraph_relations(self, entities, max_depth=2):
            return {"relations": self.relations, "active_roots": entities}

        def add_relation(self, src, pred, tgt, confidence=1.0):
            self.updated.append((src, pred, tgt, confidence))

    g = MockGraph()
    engine = RetroCausalEngine(graph_client=g)

    res = await engine.recalibrate_graph_with_annealer(target_entity="A", max_depth=2)

    assert res["status"] == "ok"
    assert res["updated_edges"] == 2
    assert len(g.updated) == 2


@pytest.mark.asyncio
async def test_runtime_set_builds_kuzu_graph_multi_hop():
    """Verify runtime 'set' automatically builds graph edges and enables multi-hop what_if queries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sock_path = Path(tmpdir) / "test_atlas.sock"
        pid_path = Path(tmpdir) / "test_atlas.pid"
        db_path = Path(tmpdir) / "test_atlas.db"

        engine = HybridMemoryEngine.create_default(db_path=str(db_path))
        daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path, engine=engine)

        await daemon._handle_set({
            "key": "test:hop:person",
            "predicate": "works_with",
            "target": "test:hop:atlas_uses",
            "confidence": 0.95,
        })
        await daemon._handle_set({
            "key": "test:hop:atlas_uses",
            "predicate": "connects_to",
            "target": "test:hop:qdrant_port",
            "confidence": 0.90,
        })

        res = await daemon._handle_what_if({
            "entity": "test:hop:person",
            "action": "remove_Dominik_access",
            "depth": 3,
        })

        paths = res.get("causal_paths", [])
        assert len(paths) >= 2
        targets = [p["affected_target"] for p in paths]
        assert "test:hop:atlas_uses" in targets
        assert "test:hop:qdrant_port" in targets


@pytest.mark.asyncio
async def test_anneal_rpc_execution():
    """Verify anneal RPC handler executes cleanly and recalibrates edges."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sock_path = Path(tmpdir) / "test_atlas.sock"
        pid_path = Path(tmpdir) / "test_atlas.pid"
        db_path = Path(tmpdir) / "test_atlas.db"

        engine = HybridMemoryEngine.create_default(db_path=str(db_path))
        daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path, engine=engine)

        await daemon._handle_set({
            "key": "anneal_root",
            "predicate": "causes",
            "target": "anneal_target",
            "confidence": 0.9,
        })

        res = await daemon._handle_anneal({"target_entity": "anneal_root"})
        assert res.get("status") == "ok"
        assert res.get("converged") is not None


@pytest.mark.asyncio
async def test_what_if_entity_not_found(tmp_path):
    """Test that what_if returns status entity_not_found with diagnostic message for missing entity."""
    daemon = AtlasDaemon(socket_path=tmp_path / "daemon.sock", pid_path=tmp_path / "daemon.pid")

    mock_graph = MagicMock(spec=["nx_graph"])
    mock_graph.nx_graph = MagicMock()
    mock_graph.nx_graph.nodes = ["DatabaseCluster", "RedisCache"]
    mock_graph.nx_graph.__contains__.side_effect = lambda x: x in ["DatabaseCluster", "RedisCache"]

    mock_causal = MagicMock()
    mock_causal.causal_what_if = AsyncMock(return_value=[])
    mock_causal.graph = mock_graph

    daemon.graph_client = mock_graph
    daemon.causal_engine = mock_causal

    res = await daemon._handle_what_if({"entity": "QwixztlBrakEncji123", "action": "test", "depth": 3})

    assert res["status"] == "entity_not_found"
    assert res["paths_count"] == 0
    assert "not found in causal knowledge graph" in res["message"]


@pytest.mark.asyncio
async def test_multihop_graph_expansion_depth():
    """Verify that multi-hop extracted relations allow what_if to expand deeper at depth=3 than depth=1."""
    graph = KuzuGraphStore(db_path=":memory:")

    note_subject = "Peptide research"
    note_content = (
        "Peptide research (Klow Stack, SlavicLabs, GHK-Cu+TB-500+BPC-157+KPV): subQ, 4°C. "
        "Rekomendacja = protokół BEZPIECZNEGO UŻYCIA po screeningu onkologicznym (colonoscopy+CEA+FIT+bloodwork). "
        "Fakty: KPV transport PepT1, GHK-Cu limit 2-3mg."
    )

    added_edges = MnemosyneIngestEngine.extract_and_insert_multihop_relations(graph, note_subject, note_content)
    assert added_edges >= 5

    causal_engine = RetroCausalEngine(graph_client=graph)
    paths_depth_1 = await causal_engine.causal_what_if("Peptide research", action="remove screening", depth=1)
    paths_depth_3 = await causal_engine.causal_what_if("Peptide research", action="remove screening", depth=3)

    assert len(paths_depth_1) > 0
    assert len(paths_depth_3) > len(paths_depth_1)
    targets_depth_3 = {p.affected_target for p in paths_depth_3}
    assert any("colonoscopy" in t.lower() or "fit" in t.lower() or "cea" in t.lower() or "ghk" in t.lower() for t in targets_depth_3)


@pytest.mark.asyncio
async def test_f2_multihop_depth_expansion():
    """Finding F-2: Multi-hop graph depth expansion retains distinct paths."""
    class LocalMockGraph:
        def __init__(self):
            self.nx_graph = nx.DiGraph()
            self.relations = []

        def add_relation(self, subj: str, pred: str, obj: str, confidence: float = 1.0):
            self.nx_graph.add_edge(subj, obj, predicate=pred, confidence=confidence)
            self.relations.append({
                "subject": subj,
                "predicate": pred,
                "object": obj,
                "confidence": confidence,
            })

        async def get_subgraph_relations(self, entities, max_depth=2):
            return {"relations": self.relations, "active_roots": entities}

    graph_client = LocalMockGraph()
    graph_client.add_relation("Daemon", "depends_on", "Socket", confidence=1.0)
    graph_client.add_relation("Socket", "routes_to", "Worker", confidence=1.0)
    graph_client.add_relation("Worker", "affects", "Latency", confidence=1.0)
    graph_client.add_relation("Daemon", "interacts_with", "Cache", confidence=1.0)
    graph_client.add_relation("Cache", "influences", "Latency", confidence=1.0)

    engine = RetroCausalEngine(graph_client=graph_client)
    paths = await engine.causal_what_if("Daemon", "failure", depth=3)

    assert len(paths) >= 2
    step_counts = [len(p.steps) for p in paths]
    assert any(c >= 2 for c in step_counts)


def test_f5_graph_auto_insertion_trimming():
    """Finding F-5: Short trivial strings are skipped from deep relation extraction."""
    class LocalMockGraph:
        def __init__(self):
            self.nx_graph = nx.DiGraph()

        def add_relation(self, subj: str, pred: str, obj: str, confidence: float = 1.0):
            self.nx_graph.add_edge(subj, obj, predicate=pred, confidence=confidence)

    graph_client = LocalMockGraph()

    MnemosyneIngestEngine.extract_and_insert_multihop_relations(
        graph_client,
        "test_subj",
        "ok",
    )
    assert len(graph_client.nx_graph.edges) == 0

    MnemosyneIngestEngine.extract_and_insert_multihop_relations(
        graph_client,
        "KlowStack",
        "Rekomendacja = protokół bezpiecznego użycia po screeningu",
    )
    for node in graph_client.nx_graph.nodes:
        assert node.lower() != "bezpiecznego"


@pytest.mark.asyncio
async def test_retro_causal_no_relations_skips_anneal(tmp_path: Path):
    """Verifies that target entities with no relations skip anneal without touching the rest of the graph."""
    db_path = str(tmp_path / "test_atlas.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )

    if hasattr(engine, "causal_engine") and engine.causal_engine is not None:
        res = await engine.causal_engine.recalibrate_graph_with_annealer(target_entity="NonExistentEntity_12345")
        assert res["status"] == "no_relations"
        assert res["updated_edges"] == 0


def test_f6_retro_causal_ignores_stated_memory_prose():
    """
    F6 (H2): Verifies that prose and documentation predicates (stated_memory, raw_content,
    documentation, source_doc) are ignored when building the causal diffusion adjacency list.
    """
    relations: List[Dict[str, Any]] = [
        {"subject": "database_timeout", "predicate": "causes", "object": "service_503", "confidence": 0.95},
        {"subject": "database_timeout", "predicate": "stated_memory", "object": "User discussed server lag yesterday in chat", "confidence": 1.0},
        {"subject": "service_503", "predicate": "raw_content", "object": "HTTP/1.1 503 Service Unavailable ... full trace ...", "confidence": 1.0},
        {"subject": "service_503", "predicate": "documentation", "object": "RFC 7231 Section 6.6.4 Specification details", "confidence": 1.0},
        {"subject": "service_503", "predicate": "triggers", "object": "circuit_breaker_trip", "confidence": 0.88},
        {"subject": "circuit_breaker_trip", "predicate": "source_doc", "object": "Architecture guide chapter 4", "confidence": 1.0},
    ]

    adj = RetroCausalEngine._build_adjacency(relations)

    all_predicates = [edge["predicate"] for edges in adj.values() for edge in edges]
    assert "stated_memory" not in all_predicates
    assert "raw_content" not in all_predicates
    assert "documentation" not in all_predicates
    assert "source_doc" not in all_predicates

    assert len(adj["database_timeout"]) == 1
    assert adj["database_timeout"][0]["predicate"] == "causes"
    assert adj["database_timeout"][0]["target"] == "service_503"

    assert len(adj["service_503"]) == 1
    assert adj["service_503"][0]["predicate"] == "triggers"
    assert adj["service_503"][0]["target"] == "circuit_breaker_trip"

    assert "circuit_breaker_trip" not in adj


def test_t5_causal_what_if_filters_and_redacts_pii():
    """
    T5: Verifies that PII entities (emails, tokens, passwords) are filtered out
    from causal graph edges and that emails in descriptions are redacted.
    """
    assert is_pii_entity("john.doe@company.org")
    assert is_pii_entity("ghp_api_key_secret_value")
    assert is_pii_entity("clientprofile_auth_token")
    assert not is_pii_entity("database_cluster")
    assert not is_pii_entity("kubernetes_service")

    relations = [
        {"subject": "service_a", "predicate": "routes_to", "object": "service_b", "confidence": 0.9},
        {"subject": "service_b", "predicate": "sends_email_to", "object": "admin@domain.com", "confidence": 0.8},
        {"subject": "api_key_bearer", "predicate": "authenticates", "object": "database", "confidence": 0.95},
    ]

    adj = RetroCausalEngine._build_adjacency(relations)
    assert "service_a" in adj
    assert len(adj["service_a"]) == 1
    assert adj["service_a"][0]["target"] == "service_b"

    assert "service_b" not in adj
    assert "api_key_bearer" not in adj

    sample_desc = "Notification sent to alert_admin@system.local regarding outage"
    redacted = EMAIL_PATTERN.sub("[REDACTED_PII]", sample_desc)
    assert "alert_admin@system.local" not in redacted
    assert "[REDACTED_PII]" in redacted

