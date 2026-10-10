"""V121 (T2) — Unit tests for PrefetchEngine.execute decomposition helpers.

Pokrywa: skip-reason formater (_format_top_rejected), empty_pool,
continuation_prompt, below_threshold ścieżki, (short_circuit) marker (F-035),
_assemble_selection i _trim_by_token_budget.
Kontrakt Golden Master (30/30) już pokryty w tests/test_v117_golden_master.py.
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from atlas_memory.cognitive.retrieval.prefetch import PrefetchEngine


@pytest.fixture(autouse=True)
def _silence_logging():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


# ─── _format_top_rejected ─────────────────────────────────────────────────────


def test_format_top_rejected_empty_scored():
    assert PrefetchEngine._format_top_rejected([]) == "none"


def test_format_top_rejected_with_candidate():
    scored = [{"subject": "fact:kalibracja:progi", "score": 0.123,
               "dense_score": 0.45, "sparse_score": 0.32, "graph_score": 0.0}]
    out = PrefetchEngine._format_top_rejected(scored)
    assert "fact:kalibracja:progi" in out
    assert "s=0.123" in out
    assert "d=0.45" in out
    assert "l=0.32" in out
    assert "g=0.00" in out


def test_format_top_rejected_truncates_subject_at_36():
    long_subj = "x" * 80
    scored = [{"subject": long_subj, "score": 0.0,
               "dense_score": 0.0, "sparse_score": 0.0, "graph_score": 0.0}]
    out = PrefetchEngine._format_top_rejected(scored)
    # 36 znaków subjectu, potem '(s=...)'
    assert out.startswith("x" * 36 + "(s=")


# ─── _resolve_context ────────────────────────────────────────────────────────


def test_resolve_context_parses_all_params():
    pe = PrefetchEngine(engine=None, calibration_engine=None, default_limit=33, default_token_budget=444)
    params = {"query": "  alfa beta  ", "session_id": "s1", "limit": 7, "token_budget": 99, "force": True}
    q, sid, lim, bud, force, eff_eng, eff_cal, eff_orch, en_cal = pe._resolve_context(
        params, engine=None, calibration_engine=None, orchestrator=None,
    )
    assert q == "alfa beta"  # stripped
    assert sid == "s1"
    assert lim == 7
    assert bud == 99
    assert force is True
    assert en_cal is True  # default fallback


def test_resolve_context_budget_aliases():
    pe = PrefetchEngine(engine=None, calibration_engine=None, default_token_budget=1000)
    # budget alias
    q, sid, lim, bud, *_ = pe._resolve_context({"query": "x", "budget": 222}, None, None, None)
    assert bud == 222
    # max_tokens alias
    q, sid, lim, bud, *_ = pe._resolve_context({"query": "x", "max_tokens": 333}, None, None, None)
    assert bud == 333


def test_resolve_context_eff_cal_falls_back_to_engine():
    fake_cal = MagicMock(name="cal")
    fake_eng = MagicMock(name="eng")
    fake_eng.calibration_engine = fake_cal
    pe = PrefetchEngine(engine=fake_eng, calibration_engine=None)
    q, sid, lim, bud, force, eff_eng, eff_cal, eff_orch, en_cal = pe._resolve_context(
        {"query": "x"}, engine=fake_eng, calibration_engine=None, orchestrator=None,
    )
    assert eff_cal is fake_cal


def test_resolve_context_explicit_calibration_engine_wins():
    fake_cal1 = MagicMock(name="c1")
    fake_cal2 = MagicMock(name="c2")
    pe = PrefetchEngine(engine=None, calibration_engine=fake_cal1)
    q, sid, lim, bud, force, eff_eng, eff_cal, eff_orch, en_cal = pe._resolve_context(
        {"query": "x"}, engine=None, calibration_engine=fake_cal2, orchestrator=None,
    )
    assert eff_cal is fake_cal2


# ─── _orchestrator_policy_skip ───────────────────────────────────────────────


def test_policy_skip_no_orchestrator_returns_none():
    out = PrefetchEngine._orchestrator_policy_skip(None, "anything", force=False)
    assert out is None


def test_policy_skip_should_retrieve_returns_none_when_allowed():
    orch = MagicMock()
    orch.should_retrieve = MagicMock(return_value=(True, "ok"))
    assert PrefetchEngine._orchestrator_policy_skip(orch, "q", force=False) is None


def test_policy_skip_3tuple():
    orch = MagicMock()
    orch.should_retrieve = MagicMock(return_value=(False, "extra", "policy_block"))
    skip = PrefetchEngine._orchestrator_policy_skip(orch, "q", force=False)
    assert skip is not None
    assert skip["skipped"] is True
    assert skip["reason"] == "policy_block"
    assert skip["threshold_selection_reason"] == "orchestrator_policy_gate"


def test_policy_skip_2tuple():
    orch = MagicMock()
    orch.should_retrieve = MagicMock(return_value=(False, "no_context"))
    skip = PrefetchEngine._orchestrator_policy_skip(orch, "q", force=False)
    assert skip["reason"] == "no_context"


def test_policy_skip_bool_true():
    orch = MagicMock()
    orch.should_retrieve = MagicMock(return_value=True)
    assert PrefetchEngine._orchestrator_policy_skip(orch, "q", force=False) is None


def test_policy_skip_bool_false():
    orch = MagicMock()
    orch.should_retrieve = MagicMock(return_value=False)
    skip = PrefetchEngine._orchestrator_policy_skip(orch, "q", force=False)
    assert skip is not None
    assert skip["reason"] == "orchestrator_policy"


def test_policy_skip_force_bypasses_gate():
    orch = MagicMock()
    orch.should_retrieve = MagicMock(return_value=(False, "x", "y"))
    assert PrefetchEngine._orchestrator_policy_skip(orch, "q", force=True) is None


def test_policy_skip_should_retrieve_raises_returns_none():
    orch = MagicMock()
    orch.should_retrieve = MagicMock(side_effect=RuntimeError("boom"))
    assert PrefetchEngine._orchestrator_policy_skip(orch, "q", force=False) is None


# ─── _fast_policy_skip ───────────────────────────────────────────────────────


def test_fast_policy_force_skips_check():
    assert PrefetchEngine._fast_policy_skip("system: anything", force=True) is None


def test_fast_policy_non_matching_returns_none():
    assert PrefetchEngine._fast_policy_skip("zwykłe pytanie", force=False) is None


def test_fast_policy_system_notification_returns_skip():
    skip = PrefetchEngine._fast_policy_skip("[system: notification gotowe", force=False)
    assert skip is not None
    assert skip["reason"] == "system_notification"
    assert skip["threshold_selection_reason"] == "policy_gate_fast_exit"


# ─── _check_sync ─────────────────────────────────────────────────────────────


def test_check_sync_none_noop():
    PrefetchEngine._check_sync(None)  # should not raise


def test_check_sync_calls_callback():
    called = []
    PrefetchEngine._check_sync(lambda: called.append(True))
    assert called == [True]


def test_check_sync_swallows_exception():
    def boom():
        raise RuntimeError("nope")
    PrefetchEngine._check_sync(boom)  # debug log only, no raise


# ─── _orchestrator_fallback_recall ───────────────────────────────────────────


async def test_fallback_recall_no_storage_no_orchestrator_does_nothing():
    eng = MagicMock()
    eng.kv = None
    eng.kv_store = None
    eng.vector_store = None
    eng.vector = None
    candidates: list = []
    await PrefetchEngine._orchestrator_fallback_recall(eng, None, "q", None, candidates)
    assert candidates == []


async def test_fallback_recall_with_real_storage_skips():
    eng = MagicMock()
    eng.kv = MagicMock()
    eng.vector_store = None
    candidates = [{"subject": "existing", "object": "x"}]
    await PrefetchEngine._orchestrator_fallback_recall(eng, None, "q", None, candidates)
    assert len(candidates) == 1  # unchanged


async def test_fallback_recall_appends_records():
    eng = MagicMock()
    eng.kv = None
    eng.kv_store = None
    eng.vector_store = None
    eng.vector = None
    orch = MagicMock()
    orch.orchestrated_recall = AsyncMock(
        return_value={"records": [{"subject": "fact:a", "object": "alpha", "content": "alpha"}]},
    )
    candidates: list = []
    await PrefetchEngine._orchestrator_fallback_recall(eng, orch, "q", "s1", candidates)
    assert len(candidates) == 1
    assert candidates[0]["subject"] == "fact:a"
    assert candidates[0]["vector_score"] == 0.0  # V91 BUG-S5: brak pomiaru → 0


async def test_fallback_recall_orchestrator_raises_swallowed():
    eng = MagicMock()
    eng.kv = None
    eng.kv_store = None
    eng.vector_store = None
    eng.vector = None
    orch = MagicMock()
    orch.orchestrated_recall = AsyncMock(side_effect=RuntimeError("boom"))
    candidates: list = []
    await PrefetchEngine._orchestrator_fallback_recall(eng, orch, "q", None, candidates)
    assert candidates == []


# ─── _gate_skip_log / F-035 short_circuit marker ─────────────────────────────


def _mk_gate(reason, threshold, should_retrieve, tsr="pava_default"):
    g = MagicMock()
    g.reason = reason
    g.threshold = threshold
    g.should_retrieve = should_retrieve
    g.threshold_selection_reason = tsr
    return g


def test_gate_skip_log_short_circuit_marker_present():
    """F-035: short_circuit clause emitted for empty_pool / continuation_prompt / system_notification / empty_query / casual_greeting."""
    gate = _mk_gate("empty_pool", 0.5, False)
    ch_counts = {"kv": 0, "vec": 0, "graph": 0, "hot": 0}
    scored: list = []
    # log emit → check internal: we capture via caplog
    import io
    import logging as _lg
    _lg.disable(_lg.NOTSET)
    logger = _lg.getLogger("atlas_memory.cognitive.retrieval.prefetch")
    buf = io.StringIO()
    h = _lg.StreamHandler(buf)
    h.setLevel(_lg.INFO)
    logger.addHandler(h)
    logger.setLevel(_lg.INFO)
    try:
        skip = PrefetchEngine._gate_skip_log("q", gate, False, ch_counts, scored, "PROSE", ["x"], [])
    finally:
        logger.removeHandler(h)
    log = buf.getvalue()
    assert "(short_circuit)" in log, f"F-035 marker missing in log: {log!r}"
    assert skip is not None
    assert skip["reason"] == "empty_pool"


def test_gate_skip_log_below_threshold_uses_score_clause():
    gate = _mk_gate("below_threshold", 0.55, False)
    ch_counts = {"kv": 0, "vec": 0, "graph": 0, "hot": 0}
    scored = [{"subject": "x", "score": 0.30, "dense_score": 0.4, "sparse_score": 0.2, "graph_score": 0.0}]
    import io
    import logging as _lg
    _lg.disable(_lg.NOTSET)
    logger = _lg.getLogger("atlas_memory.cognitive.retrieval.prefetch")
    buf = io.StringIO()
    h = _lg.StreamHandler(buf)
    h.setLevel(_lg.INFO)
    logger.addHandler(h)
    logger.setLevel(_lg.INFO)
    try:
        PrefetchEngine._gate_skip_log("q", gate, False, ch_counts, scored, "PROSE", ["x"], scored)
    finally:
        logger.removeHandler(h)
    log = buf.getvalue()
    assert "max_score=0.300 < thresh=0.550" in log
    assert "(short_circuit)" not in log


def test_gate_skip_log_force_returns_none():
    gate = _mk_gate("empty_pool", 0.5, False)
    skip = PrefetchEngine._gate_skip_log("q", gate, force=True, ch_counts={}, scored=[], mode_diag="P", q_stems_diag=[], candidates=[])
    assert skip is None


# ─── _assemble_selection ────────────────────────────────────────────────────


def test_assemble_selection_continuation_empty_skip():
    pe = PrefetchEngine()
    gate = _mk_gate("continuation_prompt", 0.0, True)
    selected, skip = pe._assemble_selection("q", [], gate, force=False, limit=5, eff_cal=None)
    assert selected == []
    assert skip is not None
    assert skip["reason"] == "continuation_prompt"
    assert skip["threshold_selection_reason"] == "no_active_session_working_memory"


def test_assemble_selection_continuation_with_hot_records():
    pe = PrefetchEngine()
    gate = _mk_gate("continuation_prompt", 0.0, True)
    scored = [
        {"subject": "hot:working:a", "source": "hot_buffer", "score": 0.5},
        {"subject": "hermes:fact:b", "source": "hot_buffer", "score": 0.4},  # excluded (PERSISTENT prefix)
        {"subject": "hot:working:c", "source": "hot_buffer", "score": 0.3},
    ]
    selected, skip = pe._assemble_selection("q", scored, gate, force=False, limit=5, eff_cal=None)
    assert skip is None
    assert len(selected) == 2
    assert all(not s["subject"].startswith("hermes:") for s in selected)


def test_assemble_selection_force_returns_full_pool():
    pe = PrefetchEngine()
    gate = _mk_gate("below_threshold", 0.9, False)
    scored = [{"score": 0.1}, {"score": 0.2}, {"score": 0.3}]
    selected, skip = pe._assemble_selection("q", scored, gate, force=True, limit=2, eff_cal=None)
    assert skip is None
    assert len(selected) == 2  # first 2 in order


def test_assemble_selection_entropy_filter_caps_by_limit():
    pe = PrefetchEngine()
    gate = _mk_gate("accepted", 0.0, True)
    # All scores identical (max-entropy) → entropy filter may drop them above threshold;
    # sprawdzamy, że selected ma ≤ limit i jest typu list
    scored = [{"subject": f"s{i}", "score": 0.5, "dense_score": 0.5,
               "sparse_score": 0.5, "graph_score": 0.0, "graph_hit": False,
               "source": "kv"} for i in range(20)]
    selected, skip = pe._assemble_selection("q", scored, gate, force=False, limit=5, eff_cal=None)
    assert skip is None
    assert isinstance(selected, list)
    assert len(selected) <= 5


# ─── _trim_by_token_budget ──────────────────────────────────────────────────


def test_trim_within_budget():
    selected = [{"subject": f"s{i}"} for i in range(5)]
    out, tot = PrefetchEngine._trim_by_token_budget(selected, budget=10000)
    assert len(out) == 5


def test_trim_breaks_when_full():
    # estimate_tokens ~117 per record; budget=200 mieści 1 rekord, drugi nie przejdzie
    selected = [{"subject": "x" * 200, "object": "y" * 200} for _ in range(5)]
    out, tot = PrefetchEngine._trim_by_token_budget(selected, budget=200)
    assert len(out) == 1
    assert tot <= 200


def test_trim_empty_selection_returns_empty():
    out, tot = PrefetchEngine._trim_by_token_budget([], budget=100)
    assert out == []
    assert tot == 0


# ─── end-to-end execute (skip path coverage) ────────────────────────────────


async def test_execute_empty_query_returns_skip():
    pe = PrefetchEngine(engine=None, calibration_engine=None)
    out = await pe.execute({"query": ""})
    assert out["skipped"] is True
    assert out["count"] == 0


async def test_execute_system_notification_short_circuits():
    pe = PrefetchEngine(engine=None, calibration_engine=None)
    out = await pe.execute({"query": "[system: notification gotowe"})
    assert out["skipped"] is True
    assert out["reason"] == "system_notification"


async def test_execute_with_force_bypasses_gate_when_pool_empty():
    pe = PrefetchEngine(engine=None, calibration_engine=None)
    out = await pe.execute({"query": "", "force": True})
    # empty pool → reason='empty_pool' still, ale skipped=False (V111 telemetry)
    # albo skipped=True depending on gate.should_retrieve; tu gate ocenia empty_pool → skipped=True
    # ale force=True → policy/fast/gate skip suppressed; gather_candidates([]) → still no records
    assert out["count"] == 0
