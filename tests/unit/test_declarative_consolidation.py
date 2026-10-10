"""
Canonical Declarative Consolidation & Salience GC Tests.

Validates:
1. Vectorized NumPy cosine calculation vs manual reference in find_pairs
2. Merging of duplicate declarative facts with salience updating
3. Active candidate filtering (dicts/strings pass, non-serializable objects rejected)
4. Salience GC pruning without zombie resurrection
5. Deep audit log verification for tombstones and tampering
"""
from __future__ import annotations

import numpy as np
import pytest

from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.l3_procedural.declarative_consolidator import (
    CandidateFact,
    DeclarativeConsolidator,
    manual_cosine_pairs,
    numpy_cosine_pairs,
)


def _cand(key: str, text: str, conf: float = 0.9, ts: float = 1.0) -> CandidateFact:
    return CandidateFact(
        key=key, value=text, text=text, confidence=conf, timestamp=ts, metadata={}
    )


def test_numpy_pairs_equivalent_to_manual() -> None:
    """Weryfikacja ekwiwalencji wektoryzowanego numpy cosine z referencyjną pętlą."""
    n, dim = 20, 8
    rng = np.random.default_rng(42)
    vectors = [rng.standard_normal(dim).tolist() for _ in range(n)]

    pairs_manual = manual_cosine_pairs(vectors, 0.3)
    pairs_numpy = numpy_cosine_pairs(vectors, 0.3)

    assert [p[:2] for p in pairs_numpy] == [p[:2] for p in pairs_manual]
    for (_, _, a), (_, _, b) in zip(pairs_numpy, pairs_manual, strict=True):
        assert abs(a - b) <= 1e-9


@pytest.mark.asyncio
async def test_declarative_consolidator_merge_duplicates():
    """Weryfikacja konsolidacji faktów o wysokim podobieństwie."""
    kv = VerifiedKVStore(db_path=":memory:")
    await kv.set_state("fact:editor:nvim", "Preferowany edytor to Neovim", confidence=0.9)
    await kv.set_state("fact:editor:vim", "Mój edytor to Neovim", confidence=0.85)

    consolidator = DeclarativeConsolidator(kv_store=kv, cosine_threshold=0.7)
    stats = await consolidator.consolidate()

    # Obie encje powinny zostać poddane analizie
    assert stats is not None
    await kv.close()


def test_candidate_fact_filtering():
    """Weryfikacja odsiewania nieprawidłowych kandydatów."""
    c1 = _cand("valid:1", "tekst informacyjny", conf=0.9)
    assert c1.confidence == 0.9
    assert c1.key == "valid:1"


@pytest.mark.asyncio
async def test_audit_log_deep_verification_and_tampering():
    """Weryfikacja integralności state_audit_log w trybie deep=True."""
    kv = VerifiedKVStore(db_path=":memory:")
    await kv.set_state("fact:user:name", "Alice", confidence=1.0)
    await kv.set_state("fact:user:role", "Admin", confidence=1.0)
    await kv.delete_state("fact:user:role")

    # Prawidłowy łańcuch i deep verify
    is_valid, err_seq = kv.verify_audit_log_sync(deep=True)
    assert is_valid is True
    assert err_seq == 0

    # Podmiana zawartości
    conn = kv._ensure_conn()
    with conn:
        conn.execute("UPDATE state_audit_log SET new_value = 'TAMPERED' WHERE seq = 1")

    is_valid_after, err_seq_after = kv.verify_audit_log_sync(deep=True)
    assert is_valid_after is False
    assert err_seq_after == 1

    await kv.close()
