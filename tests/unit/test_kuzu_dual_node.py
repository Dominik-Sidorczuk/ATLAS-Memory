"""V104 / W1 Unit & Contract Tests: Kùzu Dual-Node Schema & ADD-ONLY Migration Invariants.

Verifies:
1. Existing `Entity` and `RELATION` counts are strictly preserved (non-decreasing) across
   `migrate_dual_node_schema_and_populate`.
2. `Passage`, `CONTAINS`, and `SYNONYM` tables are populated with >=95% `CONTAINS` coverage.
3. `hydrate_from_disk_sync` faithfully restores `Passage`, `CONTAINS`, and `SYNONYM` into RAM.
"""

from __future__ import annotations

from pathlib import Path

from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore


def test_w1_dual_node_migration_preserves_entities_and_relations(tmp_path: Path) -> None:
    db_dir = tmp_path / "kuzu_w1"
    store = KuzuGraphStore(db_path=str(db_dir))
    try:
        store.add_entity("atlas_daemon", entity_type="service")
        store.add_entity("uds_socket", entity_type="transport")
        store.add_relation("atlas_daemon", "listens_on", "uds_socket", confidence=0.95)

        passages = [
            {
                "key": "hermes:memory_md:Obsidian MCP",
                "text": "Obsidian MCP UŻYWAĆ natywnych narzędzi mcp obsidian vault write na porcie 27124.",
                "gist": "Obsidian MCP vault_write port 27124",
            },
            {
                "key": "fact:atlas:arch",
                "text": "Architektura ATLAS opiera się na Kùzu Graph DB, Qdrant Vector Store i gnieździe UDS.",
            },
            {
                "key": "rule:git:branch_strategy",
                "text": "Każda faza Loop Engineering pracuje na izolowanym worktree i gałęzi z tagiem.",
            },
        ]

        report = store.migrate_dual_node_schema_and_populate(passages, synonym_threshold=0.80)

        assert report["entities_before"] == 2
        assert report["entities_after"] > report["entities_before"]
        assert report["relations_before"] == 1
        assert report["relations_after"] == report["relations_before"]
        assert report["passages_added"] == 3
        assert report["contains_coverage"] >= 0.95
        assert report["contains_edges_added"] >= 15

        # Original relation must remain intact
        edge_data = store.nx_graph.get_edge_data("atlas_daemon", "uds_socket", key="listens_on")
        assert edge_data is not None
        assert abs(float(edge_data["confidence"]) - 0.95) < 1e-6
    finally:
        store.close()


def test_w1_dual_node_disk_hydration_roundtrip(tmp_path: Path) -> None:
    db_dir = tmp_path / "kuzu_w1_hydrate"
    store1 = KuzuGraphStore(db_path=str(db_dir))
    try:
        store1.add_relation("node_alpha", "depends_on", "node_beta", confidence=0.9)
        store1.add_passage("passage:fact:alpha", "fact:alpha", "Alpha service depends on Beta database cluster.")
        store1.add_contains("passage:fact:alpha", "node_alpha", weight=1.0)
        store1.add_synonym("node_alpha", "alpha_svc", sim=0.92)
    finally:
        store1.close()

    store2 = KuzuGraphStore(db_path=str(db_dir))
    try:
        stats = store2.hydrate_from_disk_sync()
        assert stats["loaded_nodes"] >= 3
        assert "passage:fact:alpha" in store2.nx_graph
        assert store2.nx_graph.nodes["passage:fact:alpha"].get("type") == "passage"
        assert store2.nx_graph.nodes["passage:fact:alpha"].get("source_key") == "fact:alpha"
        assert store2.nx_graph.get_edge_data("passage:fact:alpha", "node_alpha", key="CONTAINS") is not None
        assert store2.nx_graph.get_edge_data("node_alpha", "alpha_svc", key="SYNONYM") is not None
    finally:
        store2.close()
