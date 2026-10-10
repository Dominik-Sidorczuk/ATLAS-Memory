"""
Tests for V149 Cognitive Runtime Remediation (Directives 1-4).
- Directive 1: Kùzu GC DETACH DELETE resilience against connected edges.
- Directive 2: Heuristic & Lexical Grounding Check in WritePath blocking ungrounded tokens/typos.
- Directive 3: Sliding Dedup Buffer (TTL: 1500ms) in PrefetchEngine.
- Directive 4: USER.md canonical Crawl4AI CLI rule verification.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from atlas_memory.cognitive.retrieval.prefetch import PrefetchEngine
from atlas_memory.cognitive.write_path import (
    EpistemicWritePath,
    is_lexically_grounded_entity,
)
from atlas_memory.l2_semantic.kuzu.schema_driver import KuzuSchemaDriverMixin


def test_directive_2_lexical_grounding_filter():
    """Verify is_lexically_grounded_entity catches conversational noise, typos, and bare particles."""
    # 1. Bare conversational noise
    ok, reason = is_lexically_grounded_entity("cztli")
    assert not ok
    assert "conversational_noise" in reason

    ok, reason = is_lexically_grounded_entity("czyli")
    assert not ok
    assert "conversational_noise" in reason

    ok, reason = is_lexically_grounded_entity("poniewaz")
    assert not ok
    assert "conversational_noise" in reason

    # 2. Typos with edit distance <= 1 to noise tokens
    ok, reason = is_lexically_grounded_entity("czyly")
    assert not ok
    assert "conversational_noise" in reason

    # 3. Short tokens < 3 chars
    ok, reason = is_lexically_grounded_entity("co")
    assert not ok
    ok, reason = is_lexically_grounded_entity("ab")
    assert not ok

    # 4. Bare token with boolean-like affirmation/negation without predicate
    ok, reason = is_lexically_grounded_entity("status", val="nie")
    assert not ok
    assert "ungrounded_discourse" in reason

    # 5. Legitimate entities
    ok, reason = is_lexically_grounded_entity("hermes:user_md:Środowisko")
    assert ok
    assert reason == "namespaced_entity"

    ok, reason = is_lexically_grounded_entity("Crawl4AI")
    assert ok
    assert reason == "grounded"

    ok, reason = is_lexically_grounded_entity("PostgreSQL")
    assert ok
    assert reason == "grounded"


def test_directive_2_write_path_blocks_cztli():
    """Verify EpistemicWritePath.write rejects ungrounded typos like cztli."""
    write_path = EpistemicWritePath()

    # Attempt to write cztli with conf=0.95
    res = write_path.write(
        key="cztli",
        value="nie",
        confidence=0.95,
    )
    assert res["status"] == "rejected"
    assert res["action"] == "reject_ungrounded"
    assert res["warning"] == "lexical_grounding_failed"

    # Attempt to commit relation with ungrounded entity
    added = write_path.commit_relation(
        subject="cztli",
        predicate="relates_to",
        object="nie",
    )
    assert not added


@pytest.mark.asyncio
async def test_directive_3_sliding_dedup_buffer():
    """Verify Sliding Dedup Buffer intercepts identical queries within 1500ms."""
    prefetch = PrefetchEngine()

    mock_engine = MagicMock()
    mock_engine.kv = MagicMock()
    mock_engine.kv.get_all_sync.return_value = {}

    params = {
        "query": "Przypomnij mi ile mam RAM",
        "session_id": "test_session",
        "limit": 10,
        "token_budget": 500,
    }

    # First call - executes normally
    res1 = await prefetch.execute(params, engine=mock_engine)
    assert not res1.get("dedup_hit", False)

    # Second call within 100ms - must hit dedup buffer
    res2 = await prefetch.execute(params, engine=mock_engine)
    assert res2.get("dedup_hit") is True
    assert res2["query"] == res1["query"]

    # Third call with force=True - must bypass dedup buffer
    params_forced = dict(params, force=True)
    res3 = await prefetch.execute(params_forced, engine=mock_engine)
    assert not res3.get("dedup_hit", False)

    # Fourth call after TTL simulation (artificially backdate timestamp)
    dedup_key = f"{params['query'].strip()}||{params['session_id']}||{params['limit']}||{params['token_budget']}"
    old_time, old_res = prefetch._dedup_cache[dedup_key]
    prefetch._dedup_cache[dedup_key] = (old_time - 2.0, old_res)

    res4 = await prefetch.execute(params, engine=mock_engine)
    assert not res4.get("dedup_hit", False)


def test_directive_1_detach_delete_query_generation():
    """Verify schema driver uses DETACH DELETE in GC logic."""
    import inspect
    src = inspect.getsource(KuzuSchemaDriverMixin._collect_and_delete_orphans_on_disk)
    assert "DETACH DELETE e" in src
    assert "DELETE e" not in src.replace("DETACH DELETE e", "")


def test_directive_4_user_md_sanitization():
    """Verify ~/.hermes/memories/USER.md has canonical Crawl4AI CLI rule without obsolete port 11235."""
    user_md_path = Path.home() / ".hermes" / "memories" / "USER.md"
    if user_md_path.exists():
        content = user_md_path.read_text(encoding="utf-8")
        assert "c4ai_agent_cli.py" in content
        assert "REST API BAN" in content
        assert "localhost:11235" not in content
