"""Unit tests verifying F-035 retrieval gate skip logging cosmetics."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pytest

from atlas_memory.cognitive.retrieval.prefetch import PrefetchEngine


def _install_prefetch_mocks(
    monkeypatch: pytest.MonkeyPatch,
    scored_pool: List[Dict[str, Any]],
) -> None:
    async def fake_gather(**kwargs: Any) -> List[Dict[str, Any]]:
        return [{"subject": "s", "predicate": "p", "object": "o", "source": "kv"}] if scored_pool else []

    def fake_score(query: str, candidates: List[Dict[str, Any]], fusion_strategy: Any = None, **kwargs: Any):
        return [dict(c) for c in scored_pool]

    monkeypatch.setattr(
        "atlas_memory.cognitive.retrieval.prefetch.gather_candidates", fake_gather
    )
    monkeypatch.setattr(
        "atlas_memory.cognitive.retrieval.prefetch.score_candidates", fake_score
    )


@pytest.mark.asyncio
async def test_prefetch_skip_log_no_impossible_inequality_on_continuation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """F-035: continuation_prompt short-circuit must not log impossible 'max_score < thresh=0.000' inequality."""
    _install_prefetch_mocks(
        monkeypatch,
        scored_pool=[{
            "subject": "kv:fact",
            "predicate": "stated",
            "object": "some fact",
            "source": "kv",
            "score": 0.101,
            "dense_score": 0.05,
            "sparse_score": 0.05,
            "graph_score": 0.0,
        }],
    )
    engine = PrefetchEngine()

    with caplog.at_level(logging.INFO, logger="atlas_memory.cognitive.retrieval.prefetch"):
        res = await engine.execute({"query": "dalej", "session_id": "s1"})

    assert res["skipped"] is True
    assert res["reason"] == "continuation_prompt"

    skip_logs = [record.message for record in caplog.records if "[RETRIEVAL:SKIP]" in record.message]
    assert len(skip_logs) >= 1
    continuation_log = next(log for log in skip_logs if "reason='continuation_prompt'" in log)

    # Must NOT contain '< thresh=' when short-circuiting on continuation_prompt
    assert "< thresh=" not in continuation_log
    assert "(short_circuit)" in continuation_log


@pytest.mark.asyncio
async def test_prefetch_skip_log_no_impossible_inequality_on_empty_pool(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """F-035: empty_pool short-circuit must not log '< thresh=' inequality."""
    _install_prefetch_mocks(monkeypatch, scored_pool=[])
    engine = PrefetchEngine()

    with caplog.at_level(logging.INFO, logger="atlas_memory.cognitive.retrieval.prefetch"):
        res = await engine.execute({"query": "nieznana kwerenda z pusta pula", "session_id": "s1"})

    assert res["skipped"] is True
    assert res["reason"] == "empty_pool"

    skip_logs = [record.message for record in caplog.records if "[RETRIEVAL:SKIP]" in record.message]
    assert len(skip_logs) >= 1
    empty_pool_log = next(log for log in skip_logs if "reason='empty_pool'" in log)

    # Must NOT contain '< thresh=' when short-circuiting on empty_pool
    assert "< thresh=" not in empty_pool_log
    assert "(short_circuit)" in empty_pool_log


@pytest.mark.asyncio
async def test_prefetch_skip_log_keeps_inequality_for_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """F-035: below_threshold retains max_score < thresh=... clause."""
    _install_prefetch_mocks(
        monkeypatch,
        scored_pool=[{
            "subject": "kv:fact",
            "predicate": "stated",
            "object": "some fact",
            "source": "kv",
            "score": 0.101,
            "dense_score": 0.05,
            "sparse_score": 0.05,
            "graph_score": 0.0,
        }],
    )
    engine = PrefetchEngine()

    with caplog.at_level(logging.INFO, logger="atlas_memory.cognitive.retrieval.prefetch"):
        res = await engine.execute({"query": "standardowa kwerenda ponizej progu", "session_id": "s1"})

    assert res["skipped"] is True
    assert res["reason"] == "below_threshold"

    skip_logs = [record.message for record in caplog.records if "[RETRIEVAL:SKIP]" in record.message]
    assert len(skip_logs) >= 1
    below_thresh_log = next(log for log in skip_logs if "reason='below_threshold'" in log)

    # Must retain '< thresh=' clause for real below_threshold
    assert "< thresh=" in below_thresh_log
