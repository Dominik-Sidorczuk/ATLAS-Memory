"""V117 Golden Master Equivalence Harness.

Pokrywa trzy monolity z V115 CC hotspots:
  - ConflictArbiter.generate_candidates  (CC=126)
  - score.score_candidates              (CC=111)
  - PrefetchEngine.execute              (CC=86)

Pancerz: asercja bitowa deterministycznego wyjścia na NIEZREFKTOROWANYM kodzie V116 (HEAD d405668).
Jakakolwiek zmiana bitu w wyniku → harness FAIL → refactor odrzucony.

Kontrakt: KNOWN_GOOD_HASH zamrożony; wektory + sniffowanie side-effectów (instance state)
zassane w pełni; record_swallow z V112 zassane w osobnym teście (freeze: 0 events
emitowanych przez te 3 funkcje — jeśli refaktor wprowadzi swallow, test padnie).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import pytest

# ─────────────────────────────────────────────────────────────────────────────────
# KNOWN_GOOD_HASH  (zamrożony na V116 d405668; SHA-256[:16]; 64-bit trunc OK)
# ─────────────────────────────────────────────────────────────────────────────────

KNOWN_GOOD_HASH: dict[str, tuple[str, str]] = {
    # ConflictArbiter.generate_candidates: (out_hash, instance_state_hash)
    "A1_empty_pool":              ("4f53cda18c2baa0c", "31c41ea2972e7b4e"),
    "A2_explicit_marker":         ("32b70c75c388721d", "31c41ea2972e7b4e"),
    "A3_inversion_same_value":    ("08a494944ed41838", "31c41ea2972e7b4e"),
    "A4_supersedes_case0":        ("8b50cc1fbbce5384", "31c41ea2972e7b4e"),
    "A5_superseded_skip":         ("4f53cda18c2baa0c", "31c41ea2972e7b4e"),
    "A6_user_revision":           ("4f53cda18c2baa0c", "31c41ea2972e7b4e"),
    "A7_inversion_diff_value":    ("b54a9716b0a74c21", "31c41ea2972e7b4e"),
    "A8_sibling_collision":       ("4f53cda18c2baa0c", "31c41ea2972e7b4e"),
    "A9_env_guard":               ("4f53cda18c2baa0c", "31c41ea2972e7b4e"),
    "A10_multislot_skip":         ("4f53cda18c2baa0c", "31c41ea2972e7b4e"),
    "A11_symbolic":               ("4f53cda18c2baa0c", "31c41ea2972e7b4e"),
    "A12_user_revision_true":     ("37e40c83dca079d9", "31c41ea2972e7b4e"),
    "A13_exclusivity_siblings":   ("4f53cda18c2baa0c", "31c41ea2972e7b4e"),
    # V117 sort-killers: 2 candidates each via _scan_existing_item (Case0 score 100 + inversion score 90)
    "A14_case0_plus_inversion":   ("1ac3b14f7ee4d7a8", "31c41ea2972e7b4e"),
    "A15_inversion_plus_case0":   ("68c0638ad4c4974c", "31c41ea2972e7b4e"),
    # score.score_candidates: (out_hash, "n/a")
    "S1_empty":                   ("4f53cda18c2baa0c", "n/a"),
    "S2_single":                  ("7a4f691fa57d341b", "n/a"),
    "S3_collision_fingerprints":  ("18396978e674f39d", "n/a"),
    "S4_rrf_strategy":            ("10ec8199a488c3f2", "n/a"),
    "S5_missing_object":          ("abf44d0f49f08be7", "n/a"),
    "S6_dict_object_superseded":  ("62d203535998c767", "n/a"),
    "S7_no_key":                  ("c1f4ab8cfe93c2c8", "n/a"),
    "S8_empty_scores":            ("d6af845d06886dc3", "n/a"),
    # PrefetchEngine.execute: (out_hash, instance_state_hash)
    "P1_empty_query":             ("4711509d83970692", "80660710d7c0133d"),
    "P2_nonsense_query":          ("bc18119144a87524", "80660710d7c0133d"),
    "P3_full_params_no_engine":   ("fbaaa30cc0ef9090", "80660710d7c0133d"),
    "P4_symbolic_query":          ("42b204a153e43204", "80660710d7c0133d"),
    "P5_nonstring_query":         ("46fdf2f9224dc3e9", "80660710d7c0133d"),
    "P6_budget_zero":             ("28821f34982658ef", "80660710d7c0133d"),
}


# ─────────────────────────────────────────────────────────────────────────────────
# Canonicalization helpers (muszą być identyczne z probe)
# ─────────────────────────────────────────────────────────────────────────────────

_TIMING_KEYS = frozenset({"ts", "duration_ms", "latency_ms", "elapsed_ms", "took_ms", "now"})


def _strip(o: Any) -> Any:
    if isinstance(o, dict):
        return {
            k: _strip(v)
            for k, v in sorted(o.items())
            if k not in _TIMING_KEYS
        }
    if isinstance(o, (list, tuple)):
        return [_strip(x) for x in o]
    if isinstance(o, float):
        return round(o, 12)
    if isinstance(o, set):
        return sorted(str(x) for x in o)
    return o


def canonical(obj: Any) -> str:
    return json.dumps(_strip(obj), ensure_ascii=False, sort_keys=True, default=str)


def sha16(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()[:16]


def _arbiter_payload(out: list) -> list:
    return [
        {k: getattr(c, k) for k in sorted(vars(c))} if hasattr(c, "__dict__") else str(c)
        for c in out
    ]


def _ex(v: Any, c: float = 1.0, m: dict | None = None) -> dict:
    return {"value": v, "confidence": c, "metadata": m or {}}


# ─────────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def arbiter():
    from atlas_memory.cognitive.conflict_arbiter import ConflictArbiter
    return ConflictArbiter(config=None)


# ─────────────────────────────────────────────────────────────────────────────────
# ConflictArbiter.generate_candidates  (CC=126)
# ─────────────────────────────────────────────────────────────────────────────────

ARB_VECTORS = {
    "A1_empty_pool": dict(key="k:x", val="v", conf=0.5, meta={}, ex={}),
    "A2_explicit_marker": dict(key="hermes:conflict:status", val="wartosc sporowa",
                                conf=0.8, meta={}, ex={}),
    "A3_inversion_same_value": dict(key="hermes:fakt:gaz", val="identical", conf=0.5,
                                    meta={}, ex={"hermes:fakt:gaz": _ex("IDENTICAL", 0.95, {"source": "user_explicit"})}),
    "A4_supersedes_case0": dict(key="hermes:fakt:nowy", val="nowa wiara", conf=0.7,
                                 meta={"supersedes": "hermes:fakt:stary"},
                                 ex={"hermes:fakt:stary": _ex("stara wiara", 0.9, {})}),
    "A5_superseded_skip": dict(key="hermes:fakt:aktywny", val="nowe", conf=0.7, meta={},
                                ex={"hermes:fakt:aktywny": _ex("stare", 0.9, {"is_superseded": True})}),
    "A6_user_revision": dict(key="hermes:fakt:gaz", val="nowa wartosc", conf=0.9,
                              meta={"source_type": "user_explicit"},
                              ex={"hermes:fakt:gaz": _ex("stara wartosc", 0.5, {})}),
    "A7_inversion_diff_value": dict(key="hermes:fakt:gaz", val="slaba proba", conf=0.3,
                                     meta={"source": "agent_inference"},
                                     ex={"hermes:fakt:gaz": _ex("ustalona prawda", 0.92, {})}),
    "A8_sibling_collision": dict(key="atlas:kalibracja:progi",
                                  val="progi wag kalibracji zamrożone", conf=0.8, meta={},
                                  ex={"atlas:kalibracja:metoda": _ex("progi wag kalibracji aktywne", 0.7, {})}),
    "A9_env_guard": dict(key="staging:db:pool", val="rozmiar 10", conf=0.9,
                          meta={"source_type": "user_explicit"},
                          ex={"production:db:pool": _ex("rozmiar 50", 0.9, {})}),
    "A10_multislot_skip": dict(key="hermes:preference:tags", val="nowy tag", conf=0.9,
                                meta={"source_type": "user_explicit"},
                                ex={"hermes:preference:tags~sib": _ex("inny tag", 0.9, {})}),
    "A11_symbolic": dict(key="42.0.0.1:8080/health", val="ok", conf=0.75,
                          meta={"kind": "identifier"}, ex={"cache:ttl": _ex("3600", 0.9, {})}),
    "A12_user_revision_true": dict(key="hermes:fakt:gaz", val="nowa wartosc", conf=0.97,
                                    meta={"source_type": "user_explicit"},
                                    ex={"hermes:fakt:gaz": _ex("stara wartosc", 0.5, {})}),
    "A13_exclusivity_siblings": dict(key="atlas:kalibracja:progi",
                                      val="tylko progi wag kalibracji zamrozenie", conf=0.9,
                                      meta={"source_type": "user_explicit"},
                                      ex={"atlas:kalibracja:metoda": _ex("rowniez progi wag kalibracji aktywne", 0.7, {})}),
    # V117 sort-killers: 2 candidates via _scan_existing_item — DIFFERENT scores
    "A14_case0_plus_inversion": dict(key="hermes:fakt:alpha", val="nowy alpha", conf=0.5,
                                      meta={"supersedes": "hermes:fakt:beta"},
                                      ex={"hermes:fakt:beta": _ex("stary beta", 0.6, {}),
                                          "hermes:fakt:alpha": _ex("ustalony alpha", 0.92, {})}),
    "A15_inversion_plus_case0": dict(key="hermes:fakt:gamma", val="nowa gamma", conf=0.4,
                                      meta={"supersedes": "hermes:fakt:delta"},
                                      ex={"hermes:fakt:gamma": _ex("ustalona gamma", 0.99, {}),
                                          "hermes:fakt:delta": _ex("stara delta", 0.85, {})}),
}


@pytest.mark.parametrize("vector_name", list(ARB_VECTORS.keys()))
def test_arbiter_generate_candidates_bit_for_bit(arbiter, vector_name: str) -> None:
    v = ARB_VECTORS[vector_name]
    out = arbiter.generate_candidates(
        key=v["key"], val=v["val"], confidence=v["conf"],
        metadata=dict(v["meta"]), existing_items=dict(v["ex"]),
    )
    payload = _arbiter_payload(out)
    expected_out, expected_state = KNOWN_GOOD_HASH[vector_name]
    actual_out = sha16(payload)
    actual_state = sha16(arbiter.__dict__)
    assert actual_out == expected_out, (
        f"{vector_name}: output drift {actual_out} != {expected_out}; n={len(out)}"
    )
    assert actual_state == expected_state, (
        f"{vector_name}: instance state drift {actual_state} != {expected_state}"
    )


# ─────────────────────────────────────────────────────────────────────────────────
# score.score_candidates  (CC=111)
# ─────────────────────────────────────────────────────────────────────────────────

SCORE_VECTORS = {
    "S1_empty": ("", []),
    "S2_single": ("raport kalibracja", [
        {"key": "k1", "object": "raport kalibracji v93 gotowy", "source": "kv",
         "scores": {"lex": 0.6, "dense": 0.2, "graph": 0.0, "sim": 0.5}},
    ]),
    "S3_collision_fingerprints": ("raport kalibracja", [
        {"key": "k1", "object": "raport kalibracji v93 gotowy", "source": "kv", "scores": {"lex": 0.6, "dense": 0.2}},
        {"key": "k2", "object": "raport kalibracji v93 gotowy", "source": "vector_dense", "scores": {"lex": 0.6, "dense": 0.9}},
        {"key": "k3", "object": "zupelnie inny tekst o wazach", "source": "graph", "scores": {"lex": 0.1, "graph": 0.9}},
    ]),
    "S4_rrf_strategy": ("xyz", [
        {"key": "a", "object": "alpha", "source": "kv", "scores": {"lex": 1.0}},
        {"key": "b", "object": "beta", "source": "vec", "scores": {"dense": 0.5}},
    ]),
    "S5_missing_object": ("test", [{"key": "no-obj", "source": "kv"}]),
    "S6_dict_object_superseded": ("rewizja", [
        {"key": "s1", "object": {"structured": [1, 2]}, "source": "kv", "metadata": {"superseded_by": "s2"}},
        {"key": "s2", "object": {"structured": [1, 2, 3]}, "source": "hot", "metadata": {}},
    ]),
    "S7_no_key": ("pytanie", [{"object": "bez klucza fakt", "source": "kv", "scores": {"lex": 0.4}}]),
    "S8_empty_scores": ("pusto", [{"key": "z1", "object": "tekst", "source": "kv", "scores": {}}]),
}


@pytest.mark.parametrize("vector_name", list(SCORE_VECTORS.keys()))
def test_score_candidates_bit_for_bit(vector_name: str) -> None:
    from atlas_memory.cognitive.retrieval.score import score_candidates
    q, cands = SCORE_VECTORS[vector_name]
    strategy = "rrf" if "rrf" in vector_name else None
    out = score_candidates(q, [dict(c) for c in cands], strategy)
    expected, _ = KNOWN_GOOD_HASH[vector_name]
    assert sha16(out) == expected, f"{vector_name}: hash drift"


# ─────────────────────────────────────────────────────────────────────────────────
# PrefetchEngine.execute  (CC=86)  — async
# ─────────────────────────────────────────────────────────────────────────────────

PREFETCH_VECTORS = {
    "P1_empty_query": {"query": ""},
    "P2_nonsense_query": {"query": "xyzzy plugh kwadraturownia żabolit", "top_k": 3},
    "P3_full_params_no_engine": {"query": "raport kalibracja atlas", "top_k": 5,
                                  "mode": "PROSE", "token_budget": 800},
    "P4_symbolic_query": {"query": "42.0.0.1:8080/health"},
    "P5_nonstring_query": {"query": 12345},
    "P6_budget_zero": {"query": "progi wag kalibracji", "token_budget": 0},
}


@pytest.mark.parametrize("vector_name", list(PREFETCH_VECTORS.keys()))
def test_prefetch_engine_execute_bit_for_bit(vector_name: str) -> None:
    from atlas_memory.cognitive.retrieval.prefetch import PrefetchEngine

    async def _run():
        eng = PrefetchEngine(engine=None, calibration_engine=None)
        out = await eng.execute(dict(PREFETCH_VECTORS[vector_name]))
        return out, sha16(eng.__dict__)

    out, state = asyncio.run(_run())
    expected_out, expected_state = KNOWN_GOOD_HASH[vector_name]
    assert sha16(out) == expected_out, f"{vector_name}: output drift"
    assert state == expected_state, f"{vector_name}: instance state drift"


# ─────────────────────────────────────────────────────────────────────────────────
# V112 record_swallow freeze
# Kontrakt: te 3 funkcje NIE emitują swallow events na obecnym kodzie.
# Refaktoryzacja, która wprowadzi swallow (np. try/except w nowej ścieżce), złamie ten test.
# ─────────────────────────────────────────────────────────────────────────────────

def test_v112_swallow_freeze_three_monoliths() -> None:
    from atlas_memory.cognitive.conflict_arbiter import ConflictArbiter
    from atlas_memory.cognitive.retrieval.prefetch import PrefetchEngine
    from atlas_memory.cognitive.retrieval.score import score_candidates
    from atlas_memory.l2_semantic import audit_ledger as al

    events: list = []
    al.reset_swallow_state()
    al.set_swallow_sink(lambda e: events.append(e))
    try:
        arb = ConflictArbiter(config=None)
        arb.generate_candidates("k", "v", 0.5, {}, {})
        score_candidates("q", [{"key": "k", "object": "o", "source": "s"}])
        asyncio.run(PrefetchEngine().execute({"query": "k"}))
    finally:
        al.set_swallow_sink(None)
        al.reset_swallow_state()

    assert events == [], (
        f"V112 freeze violated: 3 monolity nie powinny emitować swallow, "
        f"emitowane={len(events)}: {events[:2]}"
    )
