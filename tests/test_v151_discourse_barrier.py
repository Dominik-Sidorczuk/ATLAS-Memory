"""V151 Wektor I / H2 — bariera dyskursywna (POS/gramatyka) w MnemosyneIngestEngine.

Falsyfikuje hipotezę H2: zaimek/wyraz pytający (jak, który, nie, czy, gdzie)
wcielony w SUBJECT trójki przechodził przez ingest do KV Store jako pseudo-encja
('jak_obecnie', 'ktory_obecnie_nie', ...) — śmieci dyskursywne zanieczyszczające
VerifiedKV, Qdrant i Kùzu.

Kontrakt V151:
  1. QUERY_META_STOPSTEMS rozszerzony o stemy ['nie', 'czy'] (+'cz'); head-token
     tematu trafiający w stopstem/stopword → temat jest śmieciem.
  2. `is_discourse_garbage_subject` = bariera POS/gramatyczna (moduł
     core.linguistics, bezpieczna do importu bez spaCy — fallback na stopstemy).
  3. Ingest z MEMORY.md INKREMENTUJE ZERO zapisów do KV dla śmieci
     (wynikowy dict pusty jeśli plik składa się wyłącznie z śmieci).
  4. Tematy legitymne (ServerNAS) przechodzą bez blokady — bez false-positive.
"""
from __future__ import annotations

from pathlib import Path

from atlas_memory.core.linguistics import QUERY_META_STOPSTEMS, is_discourse_garbage_subject
from atlas_memory.ingest.mnemosyne_ingest import MnemosyneIngestEngine

GARBAGE_SUBJECTS = [
    "jak_obecnie",
    "który_obecnie_nie",
    "nie_ma_porównania_jak",
    "czy_warto",
    "gdzie_ta_usługa",
]

LEGITIMATE_SUBJECTS = [
    "ServerNAS",
    "obsidian_vault",
    "Klow Stack",
]


class FakeWritePath:
    """Stub EpistemicWritePath: rejestruje wywołania .write() (zapis do KV)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def write(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return {"status": "persisted", "edges_added": 0}


class FakeIngestEngineTarget:
    """Silnik docelowy dla ingestu tylko z rejestrowanym write_path."""

    def __init__(self) -> None:
        self.kv = None
        self.vector_index = None
        self.canonicalizer = None
        self.write_path = FakeWritePath()

    async def close(self) -> None:
        pass


def _md_file(tmp_path: Path, subjects: list[str]) -> Path:
    sections = "\n§\n".join(f"{s}: notatka testowa" for s in subjects)
    p = tmp_path / "MEMORY.md"
    p.write_text(sections if len(subjects) > 1 else subjects[0] + ": notatka testowa", encoding="utf-8")
    return p


def test_v151_1_barrier_flags_all_trash_subjects():
    """Bariera klasyfikuje WSZYSTKIE 5 zanieczyszczonych tematów jako śmieć."""
    for subj in GARBAGE_SUBJECTS:
        assert is_discourse_garbage_subject(subj) is True, f"przeciekł temat: {subj!r}"


def test_v151_2_barrier_passes_legitimate_subjects():
    """Zero false-positive na normalnych encjach (słowa domenowe)."""
    for subj in LEGITIMATE_SUBJECTS:
        assert is_discourse_garbage_subject(subj) is False, f"fałszywy alarm: {subj!r}"


def test_v151_3_query_meta_stopstems_extended():
    """Nowe stemy zatrzymujące dyskurs MUSZĄ być w QUERY_META_STOPSTEMS."""
    for stem in ("nie", "czy", "cz"):
        assert stem in QUERY_META_STOPSTEMS, f"brak stemu w QUERY_META_STOPSTEMS: {stem!r}"
    # istniejące stemy zatrzymujące (regresja): 'jak', 'ktor', 'gdzi'.
    for stem in ("jak", "ktor", "gdzi"):
        assert stem in QUERY_META_STOPSTEMS


def test_v151_4_ingest_writes_zero_kv_for_trash(tmp_path):
    """H2: plik MEMORY.md z samymi śmieciami → pusty rezultat ingestu,
    ZERO zapisów do KV (write_path nie wołany ani razu)."""
    engine = MnemosyneIngestEngine(memory_md_path=_md_file(tmp_path, GARBAGE_SUBJECTS))
    target = FakeIngestEngineTarget()

    result = engine.sync_memory_md_into_engine(target)

    # 'pusty dict': zero zapisów do KV / wektorów / grafu.
    assert result["ingested"] == 0
    assert result["updated"] == 0
    assert target.write_path.calls == [], f"przeciekło do KV: {target.write_path.calls}"
    assert result.get("status") == "ok"


def test_v151_5_ingest_writes_only_legitimate(tmp_path):
    """H2 (kontrapunkt): temat legitymny przechodzi, śmieć jest odcięty.
    Ingest mieszany: 5 śmieci + ServerNAS → dokładnie 1 zapis do KV."""
    subjects = GARBAGE_SUBJECTS + ["ServerNAS"]
    engine = MnemosyneIngestEngine(memory_md_path=_md_file(tmp_path, subjects))
    target = FakeIngestEngineTarget()

    result = engine.sync_memory_md_into_engine(target)

    assert result["ingested"] == 1
    assert len(target.write_path.calls) == 1
    claimed = target.write_path.calls[0]
    assert "ServerNAS".lower() in str(claimed.get("key", "")).lower()
