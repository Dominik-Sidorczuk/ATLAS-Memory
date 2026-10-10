"""Unit tests verifying Layer 0 Core Remediation (V134).

Verifies the 10 core architectural and logic fixes:
1. `protocols.py`: Elimination of Layer 0 coupling in `fetch_graph_ppr_candidates` (lazy delegation to cognitive layer) and clean import isolation.
2. `protocols.py`: Safe `normalize_vector_hit` immune to `TypeError` on `sparse_score=None` and preserving `sparse_score` on object/ScoredPoint hits.
3. `models.py`: Word-boundary and domain whitelist in `is_pii_entity` protecting LLM terminology (token, tokenizer, max_tokens, shared_secret, etc.).
4. `models.py`: Extensible `EpistemicSource` with SENSOR_OBSERVATION, CRDT_SYNC, HOT_BUFFER and string union in `MemoryRecord`.
5. `exceptions.py`: `InvariantViolation` inheriting from `AtlasError`.
6. `linguistics.py`: Protection of English technical terms (node, code, cache, base, pipeline) and loanwords (cookie, zombie, calorie).
7. `config.py`: Inline comment stripping in `_load_env_file_dict` and `AtlasConfig.from_env()`.
8. `config.py`: `AtlasConfig` paths dynamically rooted in `HERMES_HOME`.
9. `models.py`: Thread-safe concurrency in `_CounterMetric` and `_MetricsRegistry`.
10. `protocols.py`: Single-key priority over full database dump in `verify_kv_keys_batch`.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from atlas_memory.core.config import AtlasConfig, _load_env_file_dict
from atlas_memory.core.exceptions import AtlasError, InvariantViolation
from atlas_memory.core.linguistics import _bilingual_stem, _heuristic_stem_fallback, spacy_lemmatize_word
from atlas_memory.core.models import (
    EpistemicSource,
    MemoryRecord,
    _CounterMetric,
    _MetricsRegistry,
    is_pii_entity,
)
from atlas_memory.core.protocols import (
    fetch_graph_ppr_candidates,
    normalize_vector_hit,
    verify_kv_keys_batch,
)


# --- 1. Elimination of Layer 0 Coupling & Static Import Isolation ---
def test_core_module_import_isolation():
    """Verify core and protocols can be imported without pulling cognitive modules."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import atlas_memory.core; "
            "import atlas_memory.core.protocols; "
            "cogs = [m for m in sys.modules if 'atlas_memory.cognitive' in m]; "
            "assert not cogs, f'Found leaked cognitive modules: {cogs}'"
        ),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Core import isolation failed: {res.stderr}"


def test_fetch_graph_ppr_candidates_delegation():
    mock_engine = MagicMock()
    mock_res = MagicMock()
    mock_res.candidates = [{"key": "test_key", "score": 0.95}]
    mock_engine.run.return_value = mock_res

    mock_gs = MagicMock()
    mock_gs.nx_graph = MagicMock()
    mock_gs.nx_graph.nodes = ["node1"]
    mock_gs._ppr_engine = mock_engine

    with patch("atlas_memory.cognitive.retrieval.ppr.is_hipporag_enabled", return_value=True):
        cands = fetch_graph_ppr_candidates(mock_gs, query="test query", limit=5)
        assert len(cands) == 1
        assert cands[0]["key"] == "test_key"
        mock_engine.run.assert_called_once()


# --- 2. Safe normalize_vector_hit ---
def test_normalize_vector_hit_dict_with_none_sparse():
    hit = {
        "id": "doc:123",
        "text": "sample text",
        "score": 0.88,
        "sparse_score": None,
        "via_sparse": False,
        "via_dense": True,
        "metadata": {"kv_key": "custom_key"},
    }
    norm = normalize_vector_hit(hit)
    assert norm["key"] == "custom_key"
    assert norm["score"] == 0.88
    assert norm["sparse_score"] is None
    assert norm["via_dense"] is True


def test_normalize_vector_hit_object_with_sparse_score():
    class MockScoredPoint:
        def __init__(self):
            self.id = "point_42"
            self.score = 0.91
            self.sparse_score = 0.85
            self.via_sparse = True
            self.via_dense = True
            self.payload = {"object": "payload content", "metadata": {"kv_key": "kv:secret_value"}}

    point = MockScoredPoint()
    norm = normalize_vector_hit(point)
    assert norm["key"] == "secret_value"
    assert norm["value"] == "payload content"
    assert norm["score"] == 0.91
    assert norm["sparse_score"] == pytest.approx(0.85)
    assert norm["via_sparse"] is True


# --- 3. is_pii_entity Domain Whitelist & Precision ---
def test_is_pii_entity_whitelists_domain_terms():
    assert is_pii_entity("tokenizer") is False
    assert is_pii_entity("max_tokens") is False
    assert is_pii_entity("token_budget") is False
    assert is_pii_entity("shared_secret") is False
    assert is_pii_entity("token") is False
    assert is_pii_entity("tokens") is False


def test_is_pii_entity_flags_actual_secrets():
    assert is_pii_entity("api_key") is True
    assert is_pii_entity("api-key") is True
    assert is_pii_entity("api_key: sk-12345") is True
    assert is_pii_entity("bearer token") is True
    assert is_pii_entity("bearer_token") is True
    assert is_pii_entity("password") is True
    assert is_pii_entity("client_secret") is True
    assert is_pii_entity("john.doe@domain.com") is True


# --- 4. Extensible EpistemicSource & MemoryRecord ---
def test_epistemic_source_extension_and_record_compatibility():
    assert EpistemicSource.SENSOR_OBSERVATION.value == "sensor_observation"
    assert EpistemicSource.CRDT_SYNC.value == "crdt_sync"
    assert EpistemicSource.HOT_BUFFER.value == "hot_buffer"

    rec1 = MemoryRecord(
        subject="cam_01",
        predicate="detected",
        object="motion",
        source_type=EpistemicSource.SENSOR_OBSERVATION,
    )
    assert rec1.source_type == EpistemicSource.SENSOR_OBSERVATION

    # Custom string source provenance should not fail validation
    rec2 = MemoryRecord(
        subject="agent_cluster",
        predicate="state",
        object="synced",
        source_type="custom_neural_sync",
    )
    assert rec2.source_type == "custom_neural_sync"


# --- 5. InvariantViolation Exception Hierarchy ---
def test_invariant_violation_caught_by_atlas_error():
    exc = InvariantViolation(module="core", name="kv_alive", observed=0, expected=1, hint="check db")
    assert isinstance(exc, AtlasError)
    assert isinstance(exc, RuntimeError)

    caught = False
    try:
        raise exc
    except AtlasError as err:
        caught = True
        assert "InvariantViolation[core:kv_alive]" in str(err)
    assert caught is True


# --- 6. Linguistics Protection of Technical Terms ---
def test_linguistics_protects_english_technical_terms():
    technical_words = ["node", "code", "cache", "base", "pipeline", "cookie", "zombie", "calorie"]
    for w in technical_words:
        assert _heuristic_stem_fallback(w) == w, f"_heuristic_stem_fallback mangled {w}"
        assert _bilingual_stem(w) == w, f"_bilingual_stem mangled {w}"

    assert spacy_lemmatize_word("cookie") == "cookie"
    assert spacy_lemmatize_word("zombie") == "zombie"
    assert spacy_lemmatize_word("calorie") == "calorie"


# --- 7. config.py Inline Comments in .env ---
def test_load_env_file_dict_strips_inline_comments(tmp_path: Path):
    env_content = (
        "# Comment header\n"
        "ATLAS_KUZU_BUFFER_MB=2048 # bufor pamieci\n"
        "ATLAS_DEBUG=1 # turn on debug\n"
        "QUOTED_VAL=\"hello # not a comment\"\n"
        "EMPTY_LINE=\n"
    )
    env_file = tmp_path / "test.env"
    env_file.write_text(env_content, encoding="utf-8")

    loaded = _load_env_file_dict(env_file)
    assert loaded["ATLAS_KUZU_BUFFER_MB"] == "2048"
    assert loaded["ATLAS_DEBUG"] == "1"
    assert loaded["QUOTED_VAL"] == "hello # not a comment"

    # From env loading integration
    cfg = AtlasConfig.from_env(env_file=env_file)
    assert cfg.kuzu_buffer_mb == 2048


# --- 8. AtlasConfig HERMES_HOME Dynamic Paths ---
def test_atlas_config_respects_hermes_home(tmp_path: Path):
    custom_home = tmp_path / "custom_hermes"
    custom_home.mkdir(parents=True, exist_ok=True)

    # 1. Direct constructor initialization
    cfg_direct = AtlasConfig(hermes_home=custom_home)
    assert cfg_direct.hermes_home == custom_home
    assert cfg_direct.socket_path == custom_home / "atlas.sock"
    assert cfg_direct.pid_path == custom_home / "atlas.pid"
    assert cfg_direct.vector_backend_config == custom_home / "atlas" / "vector_backend.json"
    assert cfg_direct.embed_model_dir == custom_home / "atlas" / "models" / "poldense68"
    assert cfg_direct.hermes_skills_dir == custom_home / "skills"

    # 2. Environment overlay initialization
    with patch.dict(os.environ, {"HERMES_HOME": str(custom_home)}, clear=False):
        cfg_env = AtlasConfig.from_env(env_dict={"HERMES_HOME": str(custom_home)})
        assert cfg_env.hermes_home == custom_home
        assert cfg_env.socket_path == custom_home / "atlas.sock"
        assert cfg_env.pid_path == custom_home / "atlas.pid"
        assert cfg_env.vector_backend_config == custom_home / "atlas" / "vector_backend.json"
        assert cfg_env.embed_model_dir == custom_home / "atlas" / "models" / "poldense68"
        assert cfg_env.hermes_skills_dir == custom_home / "skills"


# --- 9. Thread-Safe Concurrency in Metrics ---
def test_counter_metric_thread_safety():
    metric = _CounterMetric("concurrent_ops")
    num_threads = 10
    increments_per_thread = 1000

    def worker():
        for _ in range(increments_per_thread):
            metric.inc()

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert metric.value == num_threads * increments_per_thread


def test_metrics_registry_thread_safety():
    reg = _MetricsRegistry()
    num_threads = 8

    def worker(idx: int):
        for _ in range(200):
            reg.counter(f"metric_{idx % 2}").inc(1)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = reg.get("metric_0") + reg.get("metric_1")
    assert total == num_threads * 200


# --- 10. verify_kv_keys_batch Single-Key Priority ---
@pytest.mark.asyncio
async def test_verify_kv_keys_batch_prefers_single_get_over_all_sync():
    class MockKVStore:
        def __init__(self):
            self.get_sync_calls = 0

        def get_all_sync(self):
            raise AssertionError("get_all_sync should not be called!")

        def get_sync(self, k: str):
            self.get_sync_calls += 1
            return {"val": 1} if k in ("k1", "k2") else None

    mock_kv = MockKVStore()
    keys = ["k1", "k2", "missing_k"]
    verified = await verify_kv_keys_batch(mock_kv, keys)
    assert verified == {"k1", "k2"}
    assert mock_kv.get_sync_calls == 3

