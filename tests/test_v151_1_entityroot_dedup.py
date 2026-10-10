"""V151.1 F-032 — ekwiwalencja EntityRoot × Slot w knapsack_semantic_dedup.

Diagnoza (live): blok recall zwracał 4 warianty tego samego faktu
('Public ATLAS-Memory v2.0.0 (08.09, main=0873d87)' vs '... (main=0873d87)' vs
prefiks 'hermes:memory_md:'), bo WEKTOR II łączył tylko SUROWO zgodne/prefiksowe
podmioty z kosinusem >= progu. Warianty dat/wersji/SHA/transportowych prefiksów
nie łapały się na tę relację.

Kontrakt V151.1 F-032:
  (a) entity_root(subject): zdj. prefiks transportowy ('hermes:memory_md:', 'kv:',
      'fact:'...), regexem usuwaj daty (ISO, dd.mm), wersje v\\d+(\\.\\d+)*,
      hex-hash [0-9a-f]{7,40} i słowa szumu (final|release|tag|starego|nowy).
  (b) slot(record) = (predicate, source_type) kanonicznie — NIE łączyć przez slot.
  (c) union-find po (EntityRoot, Slot); kanon = max(confidence × exp(-λ·age)),
      λ = salience decay (0.035/dzień, konfigurowalne).
  (d) wiring: atlas_recall → prefetch → pack_knapsack_01 → knapsack_semantic_dedup
      (sprawdzone: prefetch.py:357 + pipeline.py:86) — test idzie przez
      pack_knapsack_01 jako proxy ścieżki recall.
"""
from __future__ import annotations

import time

from atlas_memory.cognitive.retrieval.render import (
    entity_root,
    knapsack_semantic_dedup,
    pack_knapsack_01,
)

DAY = 86400.0
NOW = time.time()


def _rec(subject: str, predicate: str, obj: str, conf: float, age_days: float, source: str = "user_explicit") -> dict:
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "confidence": conf,
        "timestamp": NOW - age_days * DAY,
        "source_type": source,
    }


VARIANT_OLD = "Public ATLAS-Memory v2.0.0 (08.09, main=0873d87)"
VARIANT_NEW = "Public ATLAS-Memory v2.0.0 (main=0873d87)"
VARIANT_MD = "hermes:memory_md:Public ATLAS-Memory (main=0873d87) release"


# ── (a) normalizacja entity_root ──────────────────────────────────────────────

def test_entity_root_strips_dates_versions_hashes_transport():
    r1 = entity_root(VARIANT_OLD)
    r2 = entity_root(VARIANT_NEW)
    r3 = entity_root(VARIANT_MD)
    assert r1 == r2 == r3, (r1, r2, r3)
    assert "0873d87" not in r1 and "08.09" not in r1 and "v2.0.0" not in r1
    assert "hermes" not in r1 and "memory_md" not in r1
    # rdzeń sensowny: nazwa encji zostaje
    assert "public atlas-memory" in r1


def test_entity_root_keeps_distinct_entities():
    assert entity_root("Prywatny Magazyn X") != entity_root("Public ATLAS-Memory v2.0.0")


# ── (c)+(b) union-find po (EntityRoot, Slot); różne sloty NIE łączone ────────

def test_dedup_collapses_date_variants():
    """Para wariantów 'Public ATLAS-Memory v2.0.0 (08.09, main=0873d87)' vs
    '... (main=0873d87)' → w packed WYCHODZI 1 kanonik; kanon = nowszy+confidentniejszy."""
    a = _rec(VARIANT_OLD, "status", "wdrozenie kanaliku ALFA", 0.7, 3.0)
    b = _rec(VARIANT_NEW, "status", "wdrozono kanalik BETA nowszy", 0.9, 1.0)
    out = knapsack_semantic_dedup([a, b])
    assert len(out) == 1
    assert out[0] is b, "kanonikiem ma być rekord nowszy i pewniejszy"


def test_dedup_different_slots_not_merged():
    """Ten sam entity_root, INNY slot (predicate) → dwa kanony (nie łączymy przez slot)."""
    a = _rec(VARIANT_NEW, "status", "wdrozono kanalik BETA", 0.9, 1.0)
    b = _rec(VARIANT_NEW, "release_date", "rok publikacji dokumentacji", 0.8, 1.0)
    out = knapsack_semantic_dedup([a, b])
    assert len(out) == 2, (a, b, out)


def test_dedup_canon_maximizes_decayed_confidence():
    """Stary bardzo pewny przegrywa z świeżym umiarkowanym: max(conf × e^(−λ·age))."""
    old = _rec(VARIANT_OLD, "status", "wdrozenie wariantu ALFA", 0.95, 30.0)
    new = _rec(VARIANT_NEW, "status", "wdrozono wariant BETA", 0.70, 0.0)
    out = knapsack_semantic_dedup([old, new])
    assert len(out) == 1
    assert out[0] is new


def test_dedup_transport_prefix_variant_collapse():
    """Prefiks transportowy 'hermes:memory_md:' nie tworzy drugiego kanonu."""
    a = _rec(VARIANT_NEW, "status", "wdrozono kanalik BETA", 0.9, 1.0)
    b = _rec(VARIANT_MD, "status", "wdrozenie kanaliku ALFA", 0.6, 2.0)
    out = knapsack_semantic_dedup([a, b])
    assert len(out) == 1
    assert out[0] is a


# ── (d) wiring recall: pack_knapsack_01 (ścieżka atlas_recall → prefetch) ─────

def test_pack_knapsack_recall_path_yields_single_canon():
    """Live-sygnał: 'Public ATLAS-Memory' w 4 wariantach → w spakowanym bloku ≤1 kanonik."""
    recs = [
        _rec(VARIANT_OLD, "status", "wdrozenie wariantu ALFA", 0.7, 3.0),
        _rec(VARIANT_NEW, "status", "wdrozono wariant BETA", 0.9, 1.0),
        _rec(VARIANT_MD, "status", "wdrozenie wariantu GAMMA", 0.6, 2.0),
        _rec("Prywatny Magazyn X", "status", "niepowiazany fakt delta", 0.8, 0.5),
    ]
    packed = pack_knapsack_01(recs, budget=4000)
    roots = [entity_root(str(r.get("subject", ""))) for r in packed]
    atlas_variants = [r for r in roots if "public atlas-memory" in r]
    assert len(atlas_variants) <= 1, f"recall zwraca {len(atlas_variants)} kanoniki dla jednej encji"
    assert any("prywatny magazyn" in r for r in roots), "legit encja nie może zniknąć"


def test_recall_transport_prefix_live_signal():
    """Sygnał live: warianty tylko-prefiks-transportowy ('hermes:memory_md:X' vs
    'X') też składają się do ≤1 kanonu. Bez F-032 oba leciały w bloku."""
    a = _rec("hermes:memory_md:Public ATLAS-Memory", "has_canonical", "zrodlo A", 0.6, 2.0)
    b = _rec("Public ATLAS-Memory", "has_canonical", "zrodlo B", 0.9, 1.0)
    packed = pack_knapsack_01([a, b], budget=3000)
    roots = [entity_root(str(r.get("subject", ""))) for r in packed]
    n_atlas = sum(1 for r in roots if "public atlas-memory" in r)
    assert n_atlas <= 1, f"transport-prefix warianty nie zostały sklejone: n={n_atlas}"
    assert len(packed) == 1 and packed[0]["object"] == "zrodlo B", "kanon = nowszy/pewniejszy (decayed conf)"
