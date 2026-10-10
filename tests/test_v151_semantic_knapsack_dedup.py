"""V151.6 — Semantyczna deduplikacja knapsacka (WEKTOR II).

Kontrakt:
  1. `knapsack_semantic_dedup(records, dedup_sim=None)` (NOWE API): grupuje
     rekordy o tym samym podmiocie (po normalizacji / prefiksie) i obiektowej
     similarności tokenów >= dedup_sim, zwraca listę z jednym kanonicznym
     wariantem z każdej grupy (zachowuje najświeższy: highest rank = confidence
     × exp(-λ·age)).
  2. `pack_knapsack_01` AKTYWNIE używa dedup (config-driven: AtlasConfig.knapsack_dedup_sim,
     def 0.88) — 5 zduplikowanych wpisów w tej samej podmiotowej grupie
     redukuje się do 1 w packed.
  3. Konfigurowalny próg (knapsack_dedup_sim=0.99) wyłącza dedup (zachowuje 5).
"""
from __future__ import annotations

import pytest

from atlas_memory.cognitive.retrieval.render import (
    knapsack_semantic_dedup,
    pack_knapsack_01,
)
from atlas_memory.core.config import AtlasConfig, reset_config


@pytest.fixture(autouse=True)
def _config_isolation():
    reset_config(AtlasConfig(knapsack_dedup_sim=0.88))
    yield
    reset_config(None)


def _make_records(theme: str, count: int, base_ts: float = 1000.0) -> list[dict]:
    """Tworzy `count` rekordów o wspólnym podmiocie i współdzielonej treści
    obiektu. Wariantacja: każdy rekord dostaje unikalny `fragment` (nowe żetony)
    tak, że:
      * cos(i, i+1) ∈ (0.88, 0.99) — `knapsack_dedup_sim=0.88` łączy (kanoniczny),
      * cos(i, i+1) < 0.99 przy progu 0.99 (knapsack_dedup_sim=0.99) — nie łączy.
    """
    base = (
        "Kanoniczny wynik ATLAS Memory: hermes monitoruje publiczny atlas "
        "memory md jako kanoniczne zrodlo prawdy o strukturze projektu "
        "wektor ii 2026 oraz o wersjach runtime pixi lock v7"
    )
    noise = [
        "delt en alfa beta gamma",
        "epsilon zeta eta theta iota kappa",
        "lambda mi ni ksi omikron pi ro sigma",
        "tau ypsilon fi chi psi omega alfa2 beta2",
        "g1 g2 g3 g4 g5 g6 g7 g8 g9 g10",
    ]
    out = []
    for i in range(count):
        diff = noise[i % len(noise)]
        obj = f"{base} {diff}"
        out.append({
            "subject": "hermes:memory_md:Public ATLAS-Memory",
            "predicate": "has_canonical",
            "object": obj,
            "confidence": 0.9,
            "timestamp": base_ts + float(i) * 1000.0,
            "source": "kv",
            "score": 0.5 + i * 0.01,
        })
    return out


def test_v151_6a_near_dup_group_collapses_to_one_in_packed() -> None:
    """5 niemal-duplikatów w packed → 1 kanoniczny (najświeższy)."""
    records = _make_records("atlas_md", 5, base_ts=1_000_000.0)
    packed = pack_knapsack_01(records, budget=10_000)
    subjects = {r.get("subject") for r in packed}
    assert subjects == {"hermes:memory_md:Public ATLAS-Memory"}, subjects
    matching = [r for r in packed if r.get("subject") == "hermes:memory_md:Public ATLAS-Memory"]
    assert len(matching) == 1, f"kanoniczna wersja powinna być 1, mamy {len(matching)}"
    freshest = max(records, key=lambda r: r["timestamp"])
    chosen = matching[0]
    assert chosen["timestamp"] == freshest["timestamp"], (
        f"powinien zostać najświeższy ts={freshest['timestamp']}, mamy {chosen['timestamp']}"
    )


def test_v151_6b_dedup_helper_keeps_canonical_only() -> None:
    """knapsack_semantic_dedup zwraca po jednym kanonicznym rekordzie z grupy."""
    records = _make_records("atlas_md", 5, base_ts=2_000_000.0)
    deduped = knapsack_semantic_dedup(records, dedup_sim=0.88)
    matching = [r for r in deduped if r.get("subject") == "hermes:memory_md:Public ATLAS-Memory"]
    assert len(matching) == 1, matching
    assert matching[0]["timestamp"] == max(r["timestamp"] for r in records)


def test_v151_6c_threshold_from_config_can_disable_dedup() -> None:
    """knapsack_dedup_sim=0.99 → 5 zduplikowanych w packed (próg zbyt wysoki)."""
    reset_config(AtlasConfig(knapsack_dedup_sim=0.99))
    records = _make_records("atlas_md", 5, base_ts=3_000_000.0)
    packed = pack_knapsack_01(records, budget=10_000)
    matching = [r for r in packed if r.get("subject") == "hermes:memory_md:Public ATLAS-Memory"]
    assert len(matching) == 5, f"wysoki próg → brak dedup, mamy {len(matching)}"


def test_v151_6d_distinct_subjects_never_deduped_together() -> None:
    """Rekordy o różnych podmiotach (po normalizacji) NIE są łączone w grupę."""
    records = _make_records("atlas_md", 3, base_ts=4_000_000.0)
    records.append({
        "subject": "hermes:memory_md:Second Canonical Source",
        "predicate": "has_canonical",
        "object": records[0]["object"],
        "confidence": 0.9,
        "timestamp": 5_000_000.0,
        "source": "kv",
        "score": 0.5,
    })
    deduped = knapsack_semantic_dedup(records, dedup_sim=0.88)
    subjects = {r.get("subject") for r in deduped}
    assert subjects == {
        "hermes:memory_md:Public ATLAS-Memory",
        "hermes:memory_md:Second Canonical Source",
    }, subjects


def test_v151_6e_packed_uses_current_config_via_reset() -> None:
    """pack_knapsack_01 re-odczytuje config (nie używa zamrożonej wartości)."""
    records = _make_records("atlas_md", 5, base_ts=6_000_000.0)
    reset_config(AtlasConfig(knapsack_dedup_sim=0.99))  # disable
    loose = pack_knapsack_01(records, budget=10_000)
    n_loose = sum(1 for r in loose if r.get("subject") == "hermes:memory_md:Public ATLAS-Memory")
    assert n_loose == 5, n_loose
    reset_config(AtlasConfig(knapsack_dedup_sim=0.80))  # enforce (more aggressive)
    tight = pack_knapsack_01(records, budget=10_000)
    n_tight = sum(1 for r in tight if r.get("subject") == "hermes:memory_md:Public ATLAS-Memory")
    assert n_tight == 1, n_tight
