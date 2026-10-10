"""FIX-3 (F-004): igraph vs Reference NumPy PPR Equivalence (L1 <= 1e-9) & 15k-Node Scale Benchmark.

Requirements:
  1. Compare PPR probability vectors from `ATLAS_PPR_BACKEND=igraph` vs `ATLAS_PPR_BACKEND=numpy`
     on the same 100-node graph: L1 distance <= 1e-9.
  2. Benchmark `ig.Graph.personalized_pagerank` on a 15,000-node dual-node graph:
     p95 latency < 50.0 ms/query and RSS delta < 100.0 MB.
"""

from __future__ import annotations

import resource
import time
from pathlib import Path

import numpy as np

from atlas_memory.cognitive.retrieval.ppr import HippoRAG2PPR
from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore


def test_igraph_matches_numpy_ppr_100_nodes_l1_le_1e9(tmp_path: Path, monkeypatch) -> None:
    store = KuzuGraphStore(db_path=str(tmp_path / "kuzu_100_nodes"))
    try:
        # Build a 100-node dual-node graph (20 Passage nodes + 80 Entity nodes)
        for p_idx in range(20):
            pid = f"passage:fact:doc_{p_idx}"
            skey = f"fact:doc_{p_idx}"
            store.add_passage(pid, skey, f"Dokument techniczny numer {p_idx} o kalibracji grafu.", ingest_at="2026-08-01")
            for e_offset in range(4):
                ent_id = f"entity_{p_idx * 4 + e_offset}"
                store.add_contains(pid, ent_id, weight=1.0, persist_to_kuzu=False)
                if e_offset > 0:
                    prev_ent = f"entity_{p_idx * 4 + e_offset - 1}"
                    store.add_synonym(prev_ent, ent_id, sim=0.88)
            if p_idx > 0:
                store.add_relation(f"entity_{p_idx * 4}", "depends_on", f"entity_{(p_idx - 1) * 4}", confidence=0.85)

        assert store.nx_graph.number_of_nodes() == 100

        monkeypatch.setenv("ATLAS_PPR_BACKEND", "igraph")
        ppr_ig = HippoRAG2PPR(damping=0.85)
        res_ig = ppr_ig.run("entity_0 entity_4", store, top_k=10)

        monkeypatch.setenv("ATLAS_PPR_BACKEND", "numpy")
        ppr_np = HippoRAG2PPR(damping=0.85)
        res_np = ppr_np.run("entity_0 entity_4", store, top_k=10)

        all_nodes = sorted(set(res_ig.node_scores.keys()) | set(res_np.node_scores.keys()))
        vec_ig = np.array([res_ig.node_scores.get(n, 0.0) for n in all_nodes], dtype=np.float64)
        vec_np = np.array([res_np.node_scores.get(n, 0.0) for n in all_nodes], dtype=np.float64)
        l1_diff = float(np.sum(np.abs(vec_ig - vec_np)))
        assert l1_diff <= 1e-9, f"Expected L1 <= 1e-9 between igraph and numpy PPR, got {l1_diff:.3e}"
        assert [c["key"] for c in res_ig.candidates] == [c["key"] for c in res_np.candidates]
    finally:
        store.close()


def test_igraph_15k_nodes_p95_latency_and_rss_budget(monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_PPR_BACKEND", "igraph")
    store = KuzuGraphStore(db_path=":memory:")
    store.conn = None  # In-memory 15k-node graph benchmark
    try:
        rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Populate 15,000 nodes (500 Passages + 14,500 Entities) and ~22,000 edges
        for p_idx in range(500):
            pid = f"passage:fact:scale_{p_idx}"
            store.nx_graph.add_node(pid, type="passage", source_key=f"fact:scale_{p_idx}", text=f"Passage {p_idx} kalibracja")
            for k in range(29):
                ent_name = f"ent_{p_idx}_{k}"
                store.nx_graph.add_node(ent_name, type="entity")
                store.nx_graph.add_edge(pid, ent_name, key="CONTAINS", predicate="CONTAINS", weight=1.0)
                if k > 0 and k % 5 == 0:
                    store.nx_graph.add_edge(ent_name, f"ent_{p_idx}_{k - 1}", key="SYNONYM", predicate="SYNONYM", sim=0.9)

        assert store.nx_graph.number_of_nodes() == 15000

        ppr = HippoRAG2PPR(damping=0.85)
        # Warmup build of O(E) igraph + inverted token index
        ppr.run("ent_0_0 kalibracja", store, top_k=10)

        latencies_ms: list[float] = []
        for q_i in range(20):
            t0 = time.perf_counter()
            res = ppr.run(f"ent_{q_i * 10}_0 kalibracja", store, top_k=10)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)
            assert res.candidates

        rss_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_delta_mb = max(0.0, float(rss_after_kb - rss_before_kb) / 1024.0)
        p95_ms = float(np.percentile(latencies_ms, 95))
        assert p95_ms < 50.0, f"15k-node PPR p95 latency {p95_ms:.2f}ms exceeded 50ms budget"
        assert rss_delta_mb < 100.0, f"15k-node PPR RSS delta {rss_delta_mb:.2f}MB exceeded 100MB budget"
    finally:
        store.close()
