"""Tests for Adversarial Protocol Omniverse V155: Neuro-Symbolic Ingest & Precision-CRDT Lattice.

Proves the 4 pillars and mutation review gates:
- M1 (Dependency Ingest): Ingestion of 'a -> b' in code/logs without grammatical verbal structure does not create relations.
- M2 (Tool Envelope): Inspection command results ('grep -i fatal error') with exit_code: 0 are marked success=True.
- M3 (Precision-CRDT): Overwrite of Pi=2.5 by node with Pi=1.0 and artificial future timestamp is strictly blocked.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from atlas_memory.cognitive.write_path import EpistemicWritePath
from atlas_memory.hermes.trajectory_extractor import (
    ToolExecutionResult,
    _judge_result,
    extract_trajectory,
)
from atlas_memory.ingest.domain_rules import DomainRelationExtractor
from atlas_memory.ingest.mnemosyne_ingest import MnemosyneIngestEngine
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.sync.bridge import StateSyncBridge
from atlas_memory.sync.crdt import CompactLWWSet, CompactVectorClock

# ==============================================================================
# M1 (Dependency Ingest Gate)
# ==============================================================================

def test_m1_dependency_ingest_no_spurious_arrow_relations():
    """M1: Injection of 'a -> b' in code or logs without verbal syntax MUST NOT create graph edges."""
    mock_graph = MagicMock()
    mock_graph.add_relation = MagicMock()
    mock_graph.add_entity = MagicMock()

    content_with_code_arrows = (
        "def process_pipeline():\n"
        "    input_tensor -> output_tensor\n"
        "    step1 => step2 => step3\n"
        "    [LOG 12:00:01] event_a → event_b\n"
    )

    engine = MnemosyneIngestEngine()
    added = engine.extract_and_insert_multihop_relations(mock_graph, "CodeSubject", content_with_code_arrows)

    # Raw arrow splitting is eradicated; no spurious edges created
    assert added == 0
    assert mock_graph.add_relation.call_count == 0


def test_m1_dependency_ingest_extracts_grammatical_relations_with_parser_confidence():
    """M1 (Positive): Grammatical sentence with subject-verb-object creates relations weighted by parser confidence."""
    mock_graph = MagicMock()
    mock_graph.add_relation = MagicMock()
    mock_graph.add_entity = MagicMock()

    content = "Serwer bazy danych wymaga dedykowanej pamięci RAM."

    mock_wp = MagicMock()
    mock_wp.commit_relation.return_value = True

    engine = MnemosyneIngestEngine()
    added = engine.extract_and_insert_multihop_relations(
        mock_graph, "BazaDanych", content, write_path=mock_wp
    )

    assert added >= 1
    assert mock_wp.commit_relation.called
    calls = mock_wp.commit_relation.call_args_list
    # Relacja z parsera zależnościowego spaCy (SVO)
    spacy_calls = [c for c in calls if c[0][1] == "wymagać"]
    assert len(spacy_calls) >= 1
    assert spacy_calls[0][1]["confidence"] in (0.95, 0.80)


def test_m1_domain_rules_calibrated_confidence():
    """M1 (Domain Rules): Domain patterns with calibrated weights do not emit arbitrary 1.0 confidence weights."""
    collected_edges = []

    def fake_add_edge(u: str, pred: str, v: str, conf: float):
        collected_edges.append((u, pred, v, conf))

    custom_patterns = [
        (r"(?:wymaga|requires)\s+([A-Za-z0-9_-]{3,})", "{subject}", "requires", 0.90),
    ]
    extractor = DomainRelationExtractor(dep_patterns=custom_patterns)
    content = "Aplikacja wymaga HTTPS dla bezpiecznego połączenia."
    extractor.extract_and_apply("App", content, fake_add_edge, set())

    assert len(collected_edges) >= 1
    for _, _, _, conf in collected_edges:
        assert conf == 0.90


# ==============================================================================
# M2 (Tool Envelope Gate)
# ==============================================================================

def test_m2_tool_envelope_grep_fatal_error_success():
    """M2: grep -i 'fatal error' daemon.log with exit_code: 0 MUST be judged as success=True."""
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_grep_999",
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "arguments": json.dumps({"command": 'grep -i "fatal error" daemon.log'}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_grep_999",
            "content": (
                "2026-10-10 12:00:00 [FATAL ERROR] NullPointerException in thread 42\n"
                "Traceback (most recent call last):\n"
                "  File 'daemon.py', line 123, in run\n"
                "Exception: fatal error occurred"
            ),
            "exit_code": 0,
            "is_error": False,
        },
    ]

    steps = extract_trajectory(messages, session_id="test_sess_m2")
    assert len(steps) == 1
    step = steps[0]
    assert step["tool_name"] == "run_command"
    assert step["success"] is True
    assert step["error"] is None


def test_m2_tool_envelope_explicit_error_and_nonzero_exit():
    """M2 (Negative): Non-zero exit code or is_error: True are strictly marked failure."""
    # 1. Non-zero exit code
    envelope_fail = ToolExecutionResult.from_raw(
        text="some output",
        tool_name="run_command",
        raw_tool_msg={"exit_code": 1, "is_error": False},
    )
    ok, err = _judge_result(envelope_fail)
    assert ok is False
    assert err == "exit_code=1"

    # 2. Protocol flagged is_error: True
    envelope_flagged = ToolExecutionResult.from_raw(
        text="failed to allocate buffer",
        tool_name="memory_alloc",
        raw_tool_msg={"is_error": True},
    )
    ok2, err2 = _judge_result(envelope_flagged)
    assert ok2 is False
    assert "failed to allocate buffer" in err2


# ==============================================================================
# M3 (Precision-CRDT Gate)
# ==============================================================================

def test_m3_precision_crdt_rejects_lower_precision_with_future_timestamp():
    """M3: Overwrite of Pi=2.5 by node Pi=1.0 with artificial future timestamp is strictly blocked."""
    crdt_set = CompactLWWSet()

    ts_now = time.time()
    # High precision node adds entry
    crdt_set.add("critical_state_key", timestamp=ts_now, node_id="node_a", precision=2.5, confidence=1.0)
    assert crdt_set.lookup("critical_state_key") is True
    assert crdt_set.adds["critical_state_key"][0] == 2.5  # Lambda = 2.5

    # Low precision node tries to delete or overwrite entry with artificial 100-year future timestamp
    future_ts = ts_now + 3153600000.0
    crdt_set.remove("critical_state_key", timestamp=future_ts, node_id="node_b_malicious", precision=1.0, confidence=1.0)

    # Element MUST remain present because Lambda=2.5 dominates Lambda=1.0
    assert crdt_set.lookup("critical_state_key") is True

    # Similarly on merge:
    evil_set = CompactLWWSet()
    evil_set.add("critical_state_key", timestamp=future_ts, node_id="node_b_malicious", precision=1.0, confidence=1.0)
    merged = crdt_set.merge(evil_set)
    assert merged.adds["critical_state_key"][0] == 2.5


def test_m3_precision_crdt_tie_breaking_by_vector_clock():
    """M3 (Tie-break): When Friston potentials are equal, VectorClock decides causality."""
    crdt_set = CompactLWWSet()

    vc1 = CompactVectorClock(clocks={"node_a": 1})
    vc2 = CompactVectorClock(clocks={"node_a": 1, "node_b": 2})  # vc2 dominates vc1

    # Equal precision Lambda=1.5
    crdt_set.add("cluster_leader", timestamp=100.0, node_id="node_a", precision=1.5, confidence=1.0, vector_clock=vc1)
    assert crdt_set.adds["cluster_leader"][1] == 1  # clock_rank = 1

    # Higher logical clock rank overwrites despite same timestamp
    crdt_set.add("cluster_leader", timestamp=100.0, node_id="node_b", precision=1.5, confidence=1.0, vector_clock=vc2)
    assert crdt_set.adds["cluster_leader"][1] == 3  # clock_rank = 3
    assert crdt_set.adds["cluster_leader"][3] == "node_b"


@pytest.mark.asyncio
async def test_m3_state_sync_bridge_blocks_low_precision_mutation_with_future_timestamp(tmp_path: Path):
    """M3 (Bridge): StateSyncBridge strictly blocks low precision mutation despite future timestamp."""
    kv = VerifiedKVStore(db_path=str(tmp_path / "test_m3.db"))
    wp = EpistemicWritePath(kv_store=kv)
    bridge = StateSyncBridge(kv_store=kv, write_path=wp, node_id="local_node")

    now = time.time()
    await wp.write_async(
        key="system:clock_policy",
        value="strict_monotonic",
        confidence=1.0,
        metadata={"precision": 2.5},
    )

    local_val = kv.get_sync("system:clock_policy")
    assert local_val is not None
    assert local_val["value"] == "strict_monotonic"

    # Malicious or desynchronized peer with lower precision (1.0) and future timestamp (+10 years)
    future_delta = {
        "source_node": "remote_node_lagged",
        "vector_clock": {"remote_node_lagged": 5},
        "mutations": [
            {
                "key": "system:clock_policy",
                "value": "lax_wall_clock",
                "confidence": 1.0,
                "precision": 1.0,
                "timestamp": now + 315360000.0,
            }
        ],
    }

    applied = await bridge.apply_remote_delta(future_delta)
    assert applied == 0  # Mutation was rejected

    # Local value must remain intact
    updated_val = kv.get_sync("system:clock_policy")
    assert updated_val["value"] == "strict_monotonic"

    kv.close_sync()
