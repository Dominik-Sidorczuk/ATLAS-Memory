from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from atlas_memory.causal.retro_causal_edge import RetroCausalEngine
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.server.atlas_daemon import AtlasDaemon


@pytest.mark.asyncio
async def test_prefetch_veracity_hierarchy_and_confidence_preservation():
    """Verify prefetch preserves exact source_type, non-null confidence, and veracity ranking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sock_path = Path(tmpdir) / "test_atlas.sock"
        pid_path = Path(tmpdir) / "test_atlas.pid"
        db_path = Path(tmpdir) / "test_atlas.db"

        engine = HybridMemoryEngine.create_default(db_path=str(db_path))
        daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path, engine=engine)

        # 1. Insert 3 state variables with distinct veracity sources and confidences
        await daemon._handle_set({
            "key": "test:veracity:agent",
            "value": "inferred_value_by_agent",
            "confidence": 0.50,
            "source_type": "agent_inference",
        })
        await daemon._handle_set({
            "key": "test:veracity:tool",
            "value": "observation_from_tool",
            "confidence": 0.85,
            "source_type": "tool_output",
        })
        await daemon._handle_set({
            "key": "test:veracity:user",
            "value": "explicit_user_instruction",
            "confidence": 1.0,
            "source_type": "user_explicit",
        })
        # Add a chat dialogue turn that should be filtered out
        await daemon._handle_set({
            "key": "fact:conversation___user__hello",
            "value": "hello there raw chatter",
            "confidence": 0.7,
            "source_type": "external_doc",
        })

        # 2. Call prefetch for query "veracity"
        res = await daemon._handle_prefetch({"query": "veracity"})
        records = res.get("records", [])

        assert len(records) >= 3
        # Chat leak must be filtered out
        assert not any("conversation___user" in r["subject"] for r in records)

        # Map by key
        by_key = {r["subject"]: r for r in records}
        assert "test:veracity:user" in by_key
        assert "test:veracity:tool" in by_key
        assert "test:veracity:agent" in by_key

        # Check source_type fidelity
        assert by_key["test:veracity:user"]["source_type"] == "user_explicit"
        assert by_key["test:veracity:tool"]["source_type"] == "tool_output"
        assert by_key["test:veracity:agent"]["source_type"] == "agent_inference"

        # Check confidence fidelity (NEVER null)
        assert by_key["test:veracity:user"]["confidence"] == 1.0
        assert by_key["test:veracity:tool"]["confidence"] == 0.85
        assert by_key["test:veracity:agent"]["confidence"] == 0.50

        # Check Veracity-first ranking order: user (top) -> tool -> agent
        keys_in_order = [r["subject"] for r in records if r["subject"] in by_key]
        assert keys_in_order.index("test:veracity:user") < keys_in_order.index("test:veracity:agent")


@pytest.mark.asyncio
async def test_what_if_medical_dose_and_destructive_action_critical_keywords():
    """Verify what_if triggers CRITICAL risk for actions with dosage or destructive keywords."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "kuzu_test"
        from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore
        kuzu_store = KuzuGraphStore(db_path=str(db_path))

        # Add nodes and causal dependency
        kuzu_store.add_entity("ghk_cu_dose", entity_type="Concept")
        kuzu_store.add_entity("copper_toxicity", entity_type="Observation")
        kuzu_store.add_relation("ghk_cu_dose", "causes", "copper_toxicity", confidence=0.95)

        engine = RetroCausalEngine(graph_client=kuzu_store)

        # Action containing dosing keyword "set_15mg_per_dose"
        res = await engine.causal_what_if(
            entity="ghk_cu_dose",
            action="set_15mg_per_dose",
        )

        assert len(res) >= 1
        top_path = res[0]
        # Should be evaluated as CRITICAL due to action keyword + critical predicate "causes"
        assert top_path.risk_level == "CRITICAL"
