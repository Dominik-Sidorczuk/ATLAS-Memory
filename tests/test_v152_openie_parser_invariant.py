"""V152 FILAR I-fix (BUG-600/601b): extract_openie wymaga parsera dla tok.subtree.

Segfault repro (2026-10-09): get_spacy_nlp('pl') z fast_lemma_only=True
(exclude=['ner','parser','senter']) => tok.subtree iteruje po uszkodzonych
head-refach => SIGSEGV w spaCy token.c/strings. Inwariant: gdy kod konsumuje
tok.subtree, pipeline MUSI zawierać parser; extract_openie_from_record wysyła
znaki żądania parsera.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from atlas_memory.core.linguistics import (
    _import_spacy_zero_torch,
    get_spacy_nlp,
)
from atlas_memory.l2_semantic.openie import extract_openie_from_record

ATLAS_DB = Path.home() / ".hermes" / "atlas" / "atlas.db"


def _load_real_memory_md_record():
    """Ładuje prawdziwy (nie skonstruowany) rekord memory_md z ATLAS KV."""
    conn = sqlite3.connect(ATLAS_DB)
    try:
        row = conn.execute(
            "SELECT key, value FROM state_variables "
            "WHERE key LIKE 'hermes:memory_md:%' AND length(value) > 100 "
            "ORDER BY length(value) DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        pytest.skip("brak rekordu memory_md w produkcyjnym ATLASie")
    return row[0], row[1]


@pytest.mark.skipif(
    _import_spacy_zero_torch() is None, reason="spaCy niedostępny w tym runtime"
)
class TestOpenIERequiresParser:
    """Inwariant OI-1: extract_openie_from_record nie może działać na pipeline bez parsera."""

    def test_openie_pipeline_must_contain_parser(self):
        nlp = get_spacy_nlp("pl", fast_lemma_only=False)
        assert nlp is not None
        assert "parser" in nlp.pipe_names, (
            "extract_openie_from_record konsumuje tok.subtree — pipeline "
            "MUSI mieć parser; inaczej Satellite Subtree iteruje po obcych "
            "head-refach i SIGSEGV w spaCy native."
        )

    def test_openie_extract_full_pipeline_no_segfault_real_record(self):
        """Live ewizja na produkcyjnym rekordzie memory_md (długim, PL)."""
        key, value = _load_real_memory_md_record()
        res = extract_openie_from_record(
            key, value, {"subject": key.split(":", 2)[-1]}
        )
        assert res is not None
        # Czytelnika interesują frazy, więc nie może być nienaruszone pustką
        assert len(res.entities) >= 1

    def test_openie_extract_phrases_from_subtree(self):
        """Frazy (subtree walk) dostępne tylko z parserem."""
        res = extract_openie_from_record(
            "test:k", "Zwykły pies goni czerwonego kota po podwórku.", {"subject": "test"}
        )
        assert any("pies" in e.lower() for e in res.entities), res.entities
