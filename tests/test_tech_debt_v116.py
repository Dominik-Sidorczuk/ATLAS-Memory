"""V116 tech-debt eradication — kontrakty regresji.

T1: sparse_terms_preview NIE istnieje (excision zombie — zero konsumentów).
T2: record_channel_query cap: >32 dynamicznych kanałów nie rozrasta słownika.
T3: kanały znane (kv/vector_dense/...) nadal się logują + dedupe adjacent nietknięty.
T4: type:ignore mają weterynaryjne uzasadnienie (optional-dep fallback provenance).
"""
from __future__ import annotations

import re
from pathlib import Path

import atlas_memory.l2_semantic.sparse_encoder as se_mod
from atlas_memory.cognitive.retrieval import ppr as ppr_mod


def test_t1_sparse_terms_preview_gone() -> None:
    assert not hasattr(se_mod, "sparse_terms_preview"), (
        "zombie excised — symbol powinien zniknąć z modułu"
    )
    src = Path(se_mod.__file__).read_text(encoding="utf-8")
    assert "sparse_terms_preview" not in src


def test_t2_dynamic_channel_cap() -> None:
    ppr_mod._HYBRID_CHANNELS_LOG.clear()
    for i in range(64):
        ppr_mod.record_channel_query(f"dyn-{i}", f"q-{i}")
    limit = ppr_mod._HYBRID_CHANNELS_LOG_MAX
    assert len(ppr_mod._HYBRID_CHANNELS_LOG) <= limit, (
        f"cap nie działa; słownik ma {len(ppr_mod._HYBRID_CHANNELS_LOG)} kanałów (limit {limit})"
    )


def test_t3_known_channels_still_recorded() -> None:
    ppr_mod._HYBRID_CHANNELS_LOG.clear()
    ppr_mod.record_channel_query("kv", "q-a")
    ppr_mod.record_channel_query("vector_dense", "q-b")
    blob = ppr_mod._HYBRID_CHANNELS_LOG
    assert "kv" in blob and "vector_dense" in blob
    assert list(blob["kv"]) == ["q-a"]
    # dedupe adjacent istniał przed V116 — nietknięty
    ppr_mod.record_channel_query("kv", "q-a")
    assert list(blob["kv"]) == ["q-a"]


def test_t4_typeignore_have_provenance() -> None:
    targets = [
        Path("src/atlas_memory/hermes/atlas_provider.py"),
        Path("src/atlas_memory/l1_working/session_tracker.py"),
    ]
    root = Path(__file__).resolve().parents[1]
    bare_re = re.compile(r"#\s*type:\s*ignore\s*(—|$)")  # BARE (bez kodu [misc]); kody sa self-justifying
    for rel in targets:
        text = (root / rel).read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines() if bare_re.search(ln)]
        for ln in lines:
            assert "fallback" in ln, (
                f"{rel}: bare type:ignore bez uzasadnienia: {ln.strip()}"
            )
