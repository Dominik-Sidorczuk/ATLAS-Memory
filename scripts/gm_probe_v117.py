"""Probe V117: generuje Golden Master hashe na NIEZREFKTOROWANYM kodzie.

Uruchom DWUKROTNIE — hashe MUSZA byc identyczne (determinism gate przed zamrozeniem).
Wektory A1-A11 projektowane pod czytaniu generate_candidates (Case 0/A/B + guardy).
"""
from __future__ import annotations

import asyncio
import hashlib
import json

from atlas_memory.cognitive.conflict_arbiter import ConflictArbiter
from atlas_memory.cognitive.retrieval.prefetch import PrefetchEngine
from atlas_memory.cognitive.retrieval.score import score_candidates


def canonical(obj):
    def _strip(o):
        if isinstance(o, dict):
            return {
                k: _strip(v)
                for k, v in sorted(o.items())
                if k not in ("ts", "duration_ms", "latency_ms", "elapsed_ms", "took_ms", "now")
            }
        if isinstance(o, (list, tuple)):
            return [_strip(x) for x in o]
        if isinstance(o, float):
            return round(o, 12)
        if isinstance(o, set):
            return sorted(str(x) for x in o)
        return o
    return json.dumps(_strip(obj), ensure_ascii=False, sort_keys=True, default=str)


def sha(obj) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()[:16]


def cand_payload(out):
    return [
        {k: getattr(c, k) for k in sorted(vars(c))} if hasattr(c, "__dict__") else str(c)
        for c in out
    ]


EX = lambda v, c=1.0, m=None: {"value": v, "confidence": c, "metadata": m or {}}  # noqa: E731

arb_vectors = [
    # A1: pusta pula
    ("A1_empty_pool", "k:x", "v", 0.5, {}, {}),
    # A2: explicit conflict marker w kluczu
    ("A2_explicit_marker", "hermes:conflict:status", "wartosc sporowa", 0.8, {}, {}),
    # A3: same-key, same-value + confidence inversion (established 0.95 user, incoming 0.5)
    ("A3_inversion_same_value", "hermes:fakt:gaz", "identical", 0.5, {},
     {"hermes:fakt:gaz": EX("IDENTICAL", 0.95, {"source": "user_explicit"})}),
    # A4: belief revision przez metadata.supersedes (Case 0)
    ("A4_supersedes_case0", "hermes:fakt:nowy", "nowa wiara", 0.7, {"supersedes": "hermes:fakt:stary"},
     {"hermes:fakt:stary": EX("stara wiara", 0.9, {})}),
    # A5: superseded skip (is_superseded=True — pominiecie, CC guard)
    ("A5_superseded_skip", "hermes:fakt:aktywny", "nowe", 0.7, {},
     {"hermes:fakt:aktywny": EX("stare", 0.9, {"is_superseded": True})}),
    # A6: same-key belief revision (user_explicit, conf >= threshold, ex_conf <= conf)
    ("A6_user_revision", "hermes:fakt:gaz", "nowa wartosc", 0.9, {"source_type": "user_explicit"},
     {"hermes:fakt:gaz": EX("stara wartosc", 0.5, {})}),
    # A7: same-key inversion przy roznej wartosci (bez rewizji: conf < ex_conf, established 0.92)
    ("A7_inversion_diff_value", "hermes:fakt:gaz", "slaba proba", 0.3, {"source": "agent_inference"},
     {"hermes:fakt:gaz": EX("ustalona prawda", 0.92, {})}),
    # A8: cross-key overlap w tym samym namespace (Case B kolizja sióstr)
    ("A8_sibling_collision", "atlas:kalibracja:progi", "progi wag kalibracji zamrożone", 0.8, {},
     {"atlas:kalibracja:metoda": EX("progi wag kalibracji aktywne", 0.7, {})}),
    # A9: orthogonal environments guard (staging vs production → [] oczekiwane)
    ("A9_env_guard", "staging:db:pool", "rozmiar 10", 0.9, {"source_type": "user_explicit"},
     {"production:db:pool": EX("rozmiar 50", 0.9, {})}),
    # A10: multi-valued slot skip (tags na distinct keys)
    ("A10_multislot_skip", "hermes:preference:tags", "nowy tag", 0.9, {"source_type": "user_explicit"},
     {"hermes:preference:tags~sib": EX("inny tag", 0.9, {})}),
    # A12: same-key user belief revision POWYZEJ progu 0.95
    ("A12_user_revision_true", "hermes:fakt:gaz", "nowa wartosc", 0.97, {"source_type": "user_explicit"},
     {"hermes:fakt:gaz": EX("stara wartosc", 0.5, {})}),
    # A13: Case B exclusivity collision (siostry + "tylko")
    ("A13_exclusivity_siblings", "atlas:kalibracja:progi", "tylko progi wag kalibracji zamrozenie", 0.9,
     {"source_type": "user_explicit"},
     {"atlas:kalibracja:metoda": EX("rowniez progi wag kalibracji aktywne", 0.7, {})}),
    # A11: symbolic/identifier key (brak token overlap → [])
    ("A11_symbolic", "42.0.0.1:8080/health", "ok", 0.75, {"kind": "identifier"},
     {"cache:ttl": EX("3600", 0.9, {})}),
    # A14 (V117 sort-killer): 2 kandydaty o RÓŻNYCH skorach przez _scan_existing_item:
    # beta via Case 0 (score 100) + alpha via same-key confidence_inversion (score 90)
    ("A14_case0_plus_inversion", "hermes:fakt:alpha", "nowy alpha", 0.5,
     {"supersedes": "hermes:fakt:beta"},
     {"hermes:fakt:beta": EX("stary beta", 0.6, {}),
      "hermes:fakt:alpha": EX("ustalony alpha", 0.92, {})}),
    # A15 (V117 sort-killer): odwrócona kolejność wstawień vs A14 (inversion 90 przed Case0 100)
    ("A15_inversion_plus_case0", "hermes:fakt:gamma", "nowa gamma", 0.4,
     {"supersedes": "hermes:fakt:delta"},
     {"hermes:fakt:gamma": EX("ustalona gamma", 0.99, {}),
      "hermes:fakt:delta": EX("stara delta", 0.85, {})}),
]

arb = ConflictArbiter(config=None)
results = {}
print("=== generate_candidates ===")
for name, k, v, conf, meta, ex in arb_vectors:
    out = arb.generate_candidates(key=k, val=v, confidence=conf, metadata=dict(meta), existing_items=dict(ex))
    oh, sh = sha(cand_payload(out)), sha(arb.__dict__)
    results[name] = (oh, sh, len(out))
    print(f"{name}: out={oh} state={sh} n={len(out)}")

s_vectors = [
    ("S1_empty", "", []),
    ("S2_single", "raport kalibracja", [
        {"key": "k1", "object": "raport kalibracji v93 gotowy", "source": "kv", "scores": {"lex": 0.6, "dense": 0.2, "graph": 0.0, "sim": 0.5}},
    ]),
    ("S3_collision_fingerprints", "raport kalibracja", [
        {"key": "k1", "object": "raport kalibracji v93 gotowy", "source": "kv", "scores": {"lex": 0.6, "dense": 0.2}},
        {"key": "k2", "object": "raport kalibracji v93 gotowy", "source": "vector_dense", "scores": {"lex": 0.6, "dense": 0.9}},
        {"key": "k3", "object": "zupelnie inny tekst o wazach", "source": "graph", "scores": {"lex": 0.1, "graph": 0.9}},
    ]),
    ("S4_rrf_strategy", "xyz", [
        {"key": "a", "object": "alpha", "source": "kv", "scores": {"lex": 1.0}},
        {"key": "b", "object": "beta", "source": "vec", "scores": {"dense": 0.5}},
    ]),
    ("S5_missing_object", "test", [{"key": "no-obj", "source": "kv"}]),
    ("S6_dict_object_superseded", "rewizja", [
        {"key": "s1", "object": {"structured": [1, 2]}, "source": "kv", "metadata": {"superseded_by": "s2"}},
        {"key": "s2", "object": {"structured": [1, 2, 3]}, "source": "hot", "metadata": {}},
    ]),
    ("S7_no_key", "pytanie", [{"object": "bez klucza fakt", "source": "kv", "scores": {"lex": 0.4}}]),
    ("S8_empty_scores", "pusto", [{"key": "z1", "object": "tekst", "source": "kv", "scores": {}}]),
]
print("=== score_candidates ===")
for name, q, cands in s_vectors:
    try:
        out = score_candidates(q, [dict(c) for c in cands], "rrf" if "rrf" in name else None)
        results[name] = (sha(out), "n/a", len(out))
        print(f"{name}: {sha(out)} n={len(out)}")
    except Exception as exc:
        results[name] = (f"EXC:{type(exc).__name__}", "n/a", str(exc)[:60])
        print(f"{name}: EXC {type(exc).__name__} msg={sha(str(exc))}")

async def run_prefetch(params: dict):
    eng = PrefetchEngine(engine=None, calibration_engine=None)
    try:
        out = await eng.execute(dict(params))
        return sha(out), sha(eng.__dict__)
    except Exception as exc:
        return f"EXC:{type(exc).__name__}", sha(str(exc))

p_vectors = [
    ("P1_empty_query", {"query": ""}),
    ("P2_nonsense_query", {"query": "xyzzy plugh kwadraturownia żabolit", "top_k": 3}),
    ("P3_full_params_no_engine", {"query": "raport kalibracja atlas", "top_k": 5, "mode": "PROSE", "token_budget": 800}),
    ("P4_symbolic_query", {"query": "42.0.0.1:8080/health"}),
    ("P5_nonstring_query", {"query": 12345}),
    ("P6_budget_zero", {"query": "progi wag kalibracji", "token_budget": 0}),
]
print("=== PrefetchEngine.execute ===")
for name, params in p_vectors:
    oh, sh = asyncio.run(run_prefetch(params))
    results[name] = (oh, sh, "-")
    print(f"{name}: out={oh} state={sh}")

from atlas_memory.l2_semantic import audit_ledger as al

events = []
al.set_swallow_sink(lambda e: events.append(e))
arb.generate_candidates("k", "v", 0.5, {}, {})
score_candidates("q", [{"key": "k", "object": "o", "source": "s"}])
asyncio.run(PrefetchEngine().execute({"query": "k"}))
print(f"V112_swallow_freeze: events={len(events)}")
al.set_swallow_sink(None)
al.reset_swallow_state()

print("SERIALIZE:", json.dumps({k: v[:2] for k, v in results.items()}, sort_keys=True))
