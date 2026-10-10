"""Contract test for truthfulness of atlas_dedup_kv.py (FINDING-075, V72.7).

Verifies:
1. Qdrant point count decreases by EXACTLY N after --apply.
2. Truncated KV keys (>60 chars) properly delete their full upsert_key in Qdrant via prefix deletion.
3. Canonical hermes:* keys are strictly protected from prefix deletion.
4. Vector store failures fail loudly (exit != 0 / RuntimeError) and never claim silent success.
"""
from pathlib import Path
from unittest.mock import patch

import pytest

from atlas_memory.core.models import MemoryRecord
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.l2_semantic.qdrant_store import QdrantVectorStore
from scripts.atlas_dedup_kv import (
    DedupResult,
    execute_deduplication,
    find_duplicates,
    plan_deduplication,
)


def test_dedup_truthfulness_qdrant_count_decreases_by_exact_n(tmp_path: Path):
    """Proves Qdrant point count decreases by exactly N and handles truncated KV keys."""
    db_file = tmp_path / "atlas_test.db"
    qdrant_dir = tmp_path / "qdrant_test"

    # Inicjalizacja silnika z dyskowym Qdrant
    engine = HybridMemoryEngine.create_default(
        db_path=str(db_file),
        qdrant_location=str(qdrant_dir),
    )

    # 1. Para 1: canonical hermes:* oraz duplikat z kluczem uciętym w KV do 60 znaków,
    # podczas gdy w Qdrant zapisano pełny długi upsert_key
    text_1 = "Obsidian: edycja przez REST API na porcie 27124"
    long_key_full = "fact:obsidian:stated_memory:wspolpraca_z_modelem_ai_zaklada_bardzo_dlugi_tekst_ktory_zostal_uciety_w_kv_do_60_znakow"
    long_key_truncated = long_key_full[:60]

    engine.kv.set_sync("hermes:memory_md:Obsidian", text_1)
    engine.kv.set_sync(long_key_truncated, text_1)

    rec_1 = MemoryRecord(
        id="rec-1",
        subject="Obsidian",
        predicate="stated_memory",
        object=text_1,
        confidence=1.0,
    )
    rec_2 = MemoryRecord(
        id="rec-2",
        subject="Obsidian",
        predicate="stated_memory",
        object=text_1,
        confidence=1.0,
    )
    # Zapis w Qdrant z pełnym upsert_key
    engine.vector_store.insert_sync(rec_1, raw_text=text_1, upsert_key="hermes:memory_md:Obsidian")
    engine.vector_store.insert_sync(rec_2, raw_text=text_1, upsert_key=f"kv:{long_key_full}")

    # 2. Para 2: standardowe bliźniaki
    text_2 = "Python 3.11 jest standardem uruchomieniowym"
    engine.kv.set_sync("hermes:memory_md:Python", text_2)
    engine.kv.set_sync("fact:python:stated_memory", text_2)

    rec_3 = MemoryRecord(id="rec-3", subject="Python", predicate="stated_memory", object=text_2, confidence=1.0)
    rec_4 = MemoryRecord(id="rec-4", subject="Python", predicate="stated_memory", object=text_2, confidence=1.0)
    engine.vector_store.insert_sync(rec_3, raw_text=text_2, upsert_key="kv:hermes:memory_md:Python")
    engine.vector_store.insert_sync(rec_4, raw_text=text_2, upsert_key="kv:fact:python:stated_memory")

    # 3. Unikalny wpis chroniony (nie duplikat)
    text_3 = "Dominik preferuje czysty TypeScript i Python"
    engine.kv.set_sync("hermes:user_md:Dominik", text_3)
    rec_5 = MemoryRecord(id="rec-5", subject="Dominik", predicate="stated_memory", object=text_3, confidence=1.0)
    engine.vector_store.insert_sync(rec_5, raw_text=text_3, upsert_key="kv:hermes:user_md:Dominik")

    # Sprawdzenie stanu początkowego
    initial_qdrant_count = engine.vector_store.client.get_collection(engine.vector_store.collection_name).points_count
    assert initial_qdrant_count == 5, f"Oczekiwano 5 punktów, otrzymano {initial_qdrant_count}"

    # Planowanie deduplikacji
    dup_groups = find_duplicates(db_file)
    plan, total_deletes = plan_deduplication(dup_groups)
    assert total_deletes == 2, f"Oczekiwano 2 rekordów do usunięcia, znaleziono {total_deletes}"

    # Zamykamy aktywny silnik przed operacją offline deduplikacji
    engine.vector_store.close()

    # Wykonanie deduplikacji
    res = execute_deduplication(db_file, qdrant_dir, plan)

    assert isinstance(res, DedupResult)
    assert res.kv_deleted == 2
    assert res.vectors_deleted == 2
    assert res == 2  # kompatybilność z int

    # WERYFIKACJA KONTRAKTOWA: Liczba punktów w Qdrant spadła o DOKŁADNIE 2 (z 5 do 3)
    check_vstore = QdrantVectorStore(path=str(qdrant_dir), collection_name=engine.vector_store.collection_name)
    final_qdrant_count = check_vstore.client.get_collection(check_vstore.collection_name).points_count
    assert final_qdrant_count == 3, f"Qdrant point count musi spaść o dokładnie 2 (5 -> 3), a wynosi {final_qdrant_count}"
    check_vstore.close()

    # Weryfikacja bazy KV: pozostały dokładnie 3 rekordy
    check_kv = VerifiedKVStore(db_path=str(db_file))
    all_kv = check_kv.get_all_sync()
    assert len(all_kv) == 3
    assert "hermes:memory_md:Obsidian" in all_kv
    assert "hermes:memory_md:Python" in all_kv
    assert "hermes:user_md:Dominik" in all_kv
    assert long_key_truncated not in all_kv
    assert "fact:python:stated_memory" not in all_kv


def test_dedup_protected_prefix_rejection(tmp_path: Path):
    """Prefix deletion must explicitly reject hermes:* and kv:hermes:* prefixes."""
    qdrant_dir = tmp_path / "qdrant_prot"
    v_store = QdrantVectorStore(path=str(qdrant_dir))

    with pytest.raises(ValueError, match="protected"):
        v_store.delete_by_upsert_key_prefix_sync("hermes:memory_md:test")

    with pytest.raises(ValueError, match="protected"):
        v_store.delete_by_upsert_key_prefix_sync("kv:hermes:anything")

    with pytest.raises(ValueError, match="too short"):
        v_store.delete_by_upsert_key_prefix_sync("a")


def test_dedup_fails_loudly_on_vector_failure(tmp_path: Path):
    """If Qdrant fails during deletion, execute_deduplication MUST raise and not claim success."""
    db_file = tmp_path / "atlas_fail.db"
    qdrant_dir = tmp_path / "qdrant_fail"
    qdrant_dir.mkdir(parents=True)

    kv = VerifiedKVStore(db_path=str(db_file))
    shared_text = "Identyczna treść"
    kv.set_sync("hermes:key1", shared_text)
    kv.set_sync("fact:key2", shared_text)

    dup_groups = find_duplicates(db_file)
    plan, total_deletes = plan_deduplication(dup_groups)
    assert total_deletes == 1

    # Mock awarii kasowania wektorów w QdrantVectorStore
    with patch.object(QdrantVectorStore, "delete_by_upsert_key_sync", side_effect=IOError("Qdrant disk error")):
        with pytest.raises(RuntimeError, match="Vector deletion failed"):
            execute_deduplication(db_file, qdrant_dir, plan)


def test_dedup_fails_when_qdrant_is_locked(tmp_path: Path):
    """When Qdrant is locked by a live process, execute_deduplication MUST raise and fail loudly."""
    db_file = tmp_path / "atlas_locked.db"
    qdrant_dir = tmp_path / "qdrant_locked"

    # Open live vector store holding lock
    live_vstore = QdrantVectorStore(path=str(qdrant_dir))

    kv = VerifiedKVStore(db_path=str(db_file))
    shared_text = "Lock collision text"
    kv.set_sync("hermes:key1", shared_text)
    kv.set_sync("fact:key2", shared_text)

    dup_groups = find_duplicates(db_file)
    plan, total_deletes = plan_deduplication(dup_groups)
    assert total_deletes == 1

    try:
        # execute_deduplication must NOT swallow the VectorStoreLockedError into logger.warning!
        with pytest.raises(RuntimeError, match="Failed to initialize QdrantVectorStore"):
            execute_deduplication(db_file, qdrant_dir, plan)
    finally:
        live_vstore.close()

