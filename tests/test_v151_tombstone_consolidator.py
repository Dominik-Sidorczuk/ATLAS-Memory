"""V151 Wektor I / H3 — garbargowe tombstony w deklaratywnym konsolidatorze snu.

Falsyfikuje hipotezę H3: garbage records (śmieci dyskursywne typu 'jak_obecnie')
przeciekające do zweryfikowanego stanu (VerifiedKVStore) nie są porządkowane w
kaskadzie snu — zostają jako aktywne pseudosubjekty, zanieczyszczając
szczegóły konfliktu w online'owym ConflictArbiter (wzbudzają duchy-skupienia).

Kontrakt V151:
  1. DeclarativeConsolidator uruchomiony w cyklu snu (consolidate()) MUSI
     wykryć śmieci w active state (entity_root lub temat trafiony w
     barierę dyskursywną) i oznaczyć je tombstone: meta.is_superseded=True
     + revision_count inkrementowany + supersession_reason='discourse_garbage_tombstone'.
  2. Legitymne rekordy (np. ServerNAS) NIE są dotykane (false-positive zablokowany).
  3. Raport konsolidacji zawiera pole 'garbage_tombstoned' > 0.
"""
from __future__ import annotations

import pytest

from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.l3_procedural.declarative_consolidator import DeclarativeConsolidator


@pytest.mark.asyncio
async def test_v151_6_sleep_consolidation_tombstones_discourse_garbage():
    """H3: konsolidator podczas snu oznacza śmieci jako superseded (inkr. revision_count)."""
    kv = VerifiedKVStore(db_path=":memory:")
    # Aktywne śmieci (entity_root='jak_obecnie' — bariera V151 z head 'jak' = stopstem).
    await kv.set_state("fact:jak_obecnie:state", "nie wiem", confidence=0.7)
    await kv.set_state("fact:ktory_obecnie_nie:slot", "placeholder", confidence=0.7)
    # Legitymny rekord — NIE ruszany.
    await kv.set_state("fact:servernas:ip", "10.0.0.9", confidence=0.95)

    consolidator = DeclarativeConsolidator(kv_store=kv)
    stats = await consolidator.consolidate()

    # Garbage aktywne: 2 rekordy.
    assert stats.garbage_tombstoned == 2, f"oczekiwano 2 tombstone'ów, mamy {stats.garbage_tombstoned}"

    # Śmieci oznaczone jako superseded z inkrementowanym revision_count.
    for key in ("fact:jak_obecnie:state", "fact:ktory_obecnie_nie:slot"):
        rec = await kv.get_state(key)
        meta = rec.get("metadata", {}) if isinstance(rec, dict) else {}
        assert meta.get("is_superseded") is True, f"brak is_superseded na {key}"
        assert meta.get("revision_count", 0) >= 1, f"brak revision_count na {key}"
        assert meta.get("supersession_reason") == "discourse_garbage_tombstone"

    # Legitymne NIE ruszone.
    legit = await kv.get_state("fact:servernas:ip")
    legit_meta = legit.get("metadata", {}) if isinstance(legit, dict) else {}
    assert not legit_meta.get("is_superseded"), "fałszywy alarm: legitymne IP potraktowane jako śmieć"
