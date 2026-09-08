"""
Unit tests for ATLAS V56 Architectural Consolidation (Vectors D-A .. D-E).
- V1 (D-A): Knowledge graph choke point via EpistemicWritePath (F6 prose, F3 chains, PII filtering)
- V2 (D-B): Centralized clamp_conf and session constants in cognitive/models.py
- V3 (D-C): Thin delegation in AtlasMemoryProvider (zero duplicated ranking loops)
- V4 (D-E): AtlasDaemon launcher singleton enforcement with exit code != 0
- V5: ADR-003 integrity and grep invariant verification
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from atlas_memory.cognitive.models import (
    HERMES_DEFAULT_SESSION,
    HERMES_GLOBAL_SESSION,
    clamp_conf,
)
from atlas_memory.cognitive.write_path import EpistemicWritePath
from atlas_memory.models import (
    HERMES_DEFAULT_SESSION as RE_HERMES_DEFAULT_SESSION,
)
from atlas_memory.models import (
    HERMES_GLOBAL_SESSION as RE_HERMES_GLOBAL_SESSION,
)
from atlas_memory.models import (
    clamp_conf as re_clamp_conf,
)


# ---------------------------------------------------------------------------
# V2 (D-B): Test clamp_conf and Domain Session Constants
# ---------------------------------------------------------------------------
def test_v56_clamp_conf() -> None:
    """Verifies clamp_conf behavior across normal, boundary, and pathological inputs."""
    # Within valid range
    assert clamp_conf(0.0) == 0.0
    assert clamp_conf(0.5) == 0.5
    assert clamp_conf(1.0) == 1.0
    assert clamp_conf(0.85) == pytest.approx(0.85)

    # Below lower bound
    assert clamp_conf(-0.01) == 0.0
    assert clamp_conf(-5.0) == 0.0
    assert clamp_conf(-999.0) == 0.0

    # Above upper bound
    assert clamp_conf(1.01) == 1.0
    assert clamp_conf(2.0) == 1.0
    assert clamp_conf(999.0) == 1.0

    # String representations
    assert clamp_conf("0.75") == 0.75
    assert clamp_conf("1.5") == 1.0
    assert clamp_conf("-2.0") == 0.0

    # None and fallback invalid types default safely to 1.0
    assert clamp_conf(None) == 1.0
    assert clamp_conf("invalid_string") == 1.0
    assert clamp_conf(object()) == 1.0
    assert clamp_conf([]) == 1.0

    # Re-exported function matches behavior
    assert re_clamp_conf(0.42) == pytest.approx(0.42)
    assert re_clamp_conf(-1.0) == 0.0


def test_v56_constants_presence() -> None:
    """Verifies domain constants exist and match expected architectural values."""
    assert HERMES_DEFAULT_SESSION == "hermes_default"
    assert HERMES_GLOBAL_SESSION == "global"
    assert RE_HERMES_DEFAULT_SESSION == "hermes_default"
    assert RE_HERMES_GLOBAL_SESSION == "global"


# ---------------------------------------------------------------------------
# V1 (D-A): Knowledge Graph Choke Point & Prose Rejection
# ---------------------------------------------------------------------------
def test_v56_graph_choke_point_rejects_prose_and_pii() -> None:
    """Verifies EpistemicWritePath.commit_relation enforces F6 prose and PII filters."""
    mock_graph = MagicMock()
    mock_graph.add_relation = MagicMock()
    write_path = EpistemicWritePath(graph_store=mock_graph)

    # 1. Ignored prose predicates
    assert write_path.commit_relation("UserPreference", "stated_memory", "Some memory text") is False
    assert write_path.commit_relation("UserPreference", "raw_content", "Raw text data") is False
    assert write_path.commit_relation("UserPreference", "documentation", "API docs") is False
    assert write_path.commit_relation("UserPreference", "source_doc", "README.md") is False

    # 2. Ignored prose sentences (count(" ") > 6 with punctuation)
    prose_sentence = "To jest przykładowe zdanie w języku naturalnym, które zawiera interpunkcję."
    assert write_path.commit_relation("Note", "observed_fact", prose_sentence) is False

    # 3. Ignored PII entities
    assert write_path.commit_relation("user@example.com", "owns", "account_123") is False
    assert write_path.commit_relation("account_123", "api_key", "secret_token_val") is False

    # 4. Ignored self-loops
    assert write_path.commit_relation("ConceptA", "relates_to", "ConceptA") is False

    # 5. Valid domain relation is accepted and passed to graph_store
    assert write_path.commit_relation("ServiceA", "depends_on", "DatabaseB", confidence=0.9) is True
    mock_graph.add_relation.assert_called_once()
    call_args = mock_graph.add_relation.call_args[0]
    assert call_args[0] == "ServiceA"
    assert call_args[1] == "depends_on"
    assert call_args[2] == "DatabaseB"
    assert mock_graph.add_relation.call_args[1]["confidence"] == pytest.approx(0.9)


def test_v56_mnemosyne_ingest_prose_does_not_create_graph_node() -> None:
    """Verifies that mnemosyne_ingest prose lines do not create graph edges."""
    from atlas_memory.ingest.mnemosyne_ingest import MnemosyneIngestEngine
    mock_graph = MagicMock()
    mock_graph.add_relation = MagicMock()
    mock_graph.add_entity = MagicMock()

    content = "To jest całe zdanie z kropką na końcu. Kolejne zdanie prozy."
    engine = MnemosyneIngestEngine()
    added = engine.extract_and_insert_multihop_relations(mock_graph, "UserSubject", content)

    # Prose sentence from content.split(".")[0] must be filtered out by EpistemicWritePath choke point
    assert mock_graph.add_relation.call_count == 0
    assert added == 0


# ---------------------------------------------------------------------------
# V1 & V3 Grep Verification Invariants
# ---------------------------------------------------------------------------
def test_v56_grep_add_relation_choke_point() -> None:
    """Verifies that add_relation( occurs ONLY in kuzu_graph.py and write_path.py in src/."""
    src_dir = Path(__file__).resolve().parent.parent.parent / "src"
    allowed_files = {
        Path("atlas_memory/l2_semantic/kuzu_graph.py"),
        Path("atlas_memory/cognitive/write_path.py"),
    }

    violating_files = []
    for py_file in src_dir.rglob("*.py"):
        rel_path = py_file.relative_to(src_dir)
        if rel_path in allowed_files:
            continue
        code = py_file.read_text(encoding="utf-8")
        if "add_relation(" in code:
            violating_files.append(str(rel_path))

    assert not violating_files, (
        f"Found direct add_relation( calls outside choke point in: {violating_files}. "
        f"All graph writes must go through EpistemicWritePath!"
    )


def test_v56_provider_thin_delegation_grep() -> None:
    """Verifies that atlas_provider.py contains no duplicated ranking logic."""
    provider_file = Path(__file__).resolve().parent.parent.parent / "src" / "atlas_memory" / "hermes" / "atlas_provider.py"
    code = provider_file.read_text(encoding="utf-8")
    matches = re.findall(r"veracity|rank|\[:limit\]", code)
    assert not matches, (
        f"atlas_provider.py contains forbidden duplicated ranking patterns: {matches}. "
        f"Provider must be a thin delegation wrapper around RetrievalPipeline."
    )


# ---------------------------------------------------------------------------
# V4 (D-E): Daemon Launcher Singleton Exit Code
# ---------------------------------------------------------------------------
def test_v56_daemon_launcher_singleton_exit_code(tmp_path: Path) -> None:
    """Verifies scripts/atlas_daemon_launcher.py exits with code != 0 when daemon is already active."""
    launcher_script = Path(__file__).resolve().parent.parent.parent / "scripts" / "atlas_daemon_launcher.py"

    # Test 1: Real running daemon test if current live socket is active
    live_sock = Path.home() / ".hermes" / "atlas.sock"
    if live_sock.exists():
        res = subprocess.run(
            [sys.executable, str(launcher_script)],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        assert res.returncode != 0, f"Expected non-zero exit code when daemon is running, got {res.returncode}"
        assert "daemon already running" in (res.stdout + res.stderr)

    # Test 2: Isolated tmp socket with fake active PID
    fake_sock = tmp_path / "fake_atlas.sock"
    fake_pid_file = tmp_path / "fake_atlas.pid"
    # Write current test process PID as simulated running daemon
    fake_pid_file.write_text(str(os.getpid()), encoding="utf-8")

    env = dict(os.environ)
    env["ATLAS_SOCKET_PATH"] = str(fake_sock)
    env["ATLAS_PID_PATH"] = str(fake_pid_file)

    res_fake = subprocess.run(
        [sys.executable, str(launcher_script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=5.0,
    )
    assert res_fake.returncode != 0, f"Expected non-zero exit code, got {res_fake.returncode}"
    assert "daemon already running" in (res_fake.stdout + res_fake.stderr)
