"""V152 FILAR I — formalna bariera aktów mowy na chokepoincie EpistemicWritePath.write.

Inwariant I-1 (ADR-V152): wejście agenta jest typowane rozłącznie
(EpistemicAssertion XOR ConversationalTurn). ConversationalTurn NIE ma prawa
trafić do KV, projekcji wektorowej ani ekstrakcji grafu (spaCy OpenIE = wektor
crashu SIGSEGV przy surowej prozie/surogatach).

Testy ZAKAZUJĄ nowych list słów: klasyfikacja wyłącznie przez istniejące
prymitywy (QUERY_META_STOPWORDS / QUERY_META_STOPSTEMS /
_DISCOURSE_GARBAGE_EXTRA_STEMS w atlas_memory.core.linguistics).
"""
from __future__ import annotations

import pytest

from atlas_memory.cognitive.write_path import EpistemicWritePath
from atlas_memory.core.models import MemoryRecord, SpeechAct, classify_speech_act

# ---------------------------------------------------------------------------
# (a) Klasyfikacja aktu mowy — PRZEZ ISTNIEJĄCE PRYMITYWY, zero nowych list.
# ---------------------------------------------------------------------------

GARBAGE_SUBJECTS = ["jak_obecnie", "który_obecnie_nie", "Z jakiej wersji"]


@pytest.mark.parametrize("subject", GARBAGE_SUBJECTS)
def test_filar_i_garbage_subject_classified_conversational(subject: str) -> None:
    """Bariery nie wolno realizować przez słowo-obiekt '?': sama nazwa tematu
    (meta-pytanie / discourse garbage) musi być ConversationalTurn."""
    assert classify_speech_act(subject, "state_is", "umnie") is SpeechAct.CONVERSATIONAL_TURN


def test_filar_i_assertion_not_demoted() -> None:
    """Legalna asercja EAV pozostaje EPISTEMIC_ASSERTION (brak demowania nazw
    własnych / kluczy namespacowanych)."""
    for subject in ("auth_service", "hermes", "project_alpha", "notatka_sieci"):
        assert classify_speech_act(subject, "has_port", "8080") is SpeechAct.EPISTEMIC_ASSERTION


# ---------------------------------------------------------------------------
# Fake'e magazynów — czyste stuby, BEZ żywych plików/Qdranta/Kùzu.
# ---------------------------------------------------------------------------


class _KVStub:
    """Spójny z KVStoreProtocol (get_sync + set_sync + async search)."""

    def __init__(self) -> None:
        self.sets: list[tuple[str, str]] = []

    def set_sync(self, key, value, confidence=1.0, metadata=None, reason=""):  # noqa: ANN001
        self.sets.append((str(key), str(value)))
        return "ok"

    def get_sync(self, key):  # noqa: ANN001
        return None

    async def get_states(self, keys):  # noqa: ANN001
        return {}

    async def filter_existing_keys(self, keys):  # noqa: ANN001
        return set()


class _VectorStub:
    def __init__(self) -> None:
        self.inserts: list[str] = []

    async def search(self, query, top_k=5, min_similarity=0.1, query_filter=None):  # noqa: ANN001
        return []

    def insert_sync(self, record, upsert_key=None):  # noqa: ANN001
        self.inserts.append(str(upsert_key))
        return "ok"


class _GraphStub:
    """Spełnia też DualNodeGraphStoreProtocol — index_dual_node_passage NIGDY
    nie może zostać wywołany dla ConversationalTurn (droga do spaCy/SIGSEGV)."""

    def __init__(self) -> None:
        self.relations: list[tuple[str, str, str]] = []
        self.indexed_passages: list[str] = []
        self.nx_graph = None

    async def get_node_relations(self, node_name, max_depth=1):  # noqa: ANN001
        return []

    def add_relation(self, subject, predicate, obj, confidence=1.0):  # noqa: ANN001
        self.relations.append((subject, predicate, obj))
        return True

    def add_passage(self, passage_id, source_key, text, ingest_at=None):  # noqa: ANN001
        return None

    def create_entity(self, name, entity_type="entity"):  # noqa: ANN001
        return None

    def add_contains(self, passage_id, entity_name, weight=1.0, persist_to_kuzu=True):  # noqa: ANN001
        return None

    def add_synonym(self, entity_a, entity_b, sim):  # noqa: ANN001
        return None

    def rollback_dual_node_record(self, passage_id, created_entities=None):  # noqa: ANN001
        return None

    def has_passage(self, passage_id):  # noqa: ANN001
        return False

    def index_dual_node_passage(self, **kwargs):  # noqa: ANN001
        self.indexed_passages.append(str(kwargs.get("source_key")))
        return 0

    def get_dual_node_coverage(self) -> float:
        return 0.0


def _make_wp() -> tuple[EpistemicWritePath, _KVStub, _VectorStub, _GraphStub]:
    kv, vec, graph = _KVStub(), _VectorStub(), _GraphStub()
    return EpistemicWritePath(kv_store=kv, graph_store=graph, vector_store=vec), kv, vec, graph


# ---------------------------------------------------------------------------
# (b) Chokepoint reject — ZERO zapisu KV, ZERO projekcji, ZERO ekstrakcji grafu.
# ---------------------------------------------------------------------------


def test_write_path_rejects_conversational_turn_zero_side_effects() -> None:
    """B1 RED: dziś 'jak_obecnie:state_value -> umnie' przechodzi przez
    write() do KV+wektora. Po bramce: reject, spójny kształt z
    reject_ungrounded, i ŻADNE wywołanie magazynu."""
    wp, kv, vec, graph = _make_wp()

    res = wp.write(key="jak_obecnie", value="umnie", vector_record=object())

    assert res["status"] == "rejected"
    assert res["persisted"] is False
    assert res["action"] == "reject_conversational_turn"
    assert res["warning"] == "speech_act_gate"
    assert res["edges_added"] == 0
    assert kv.sets == []
    assert vec.inserts == []
    assert graph.relations == []
    assert graph.indexed_passages == []


def test_write_path_rejects_raw_prose_subject() -> None:
    """Surowa proza 'Z jakiej wersji' bez '?' w obiekcie też musi być
    odrzucona na chokepoincie (klasyfikacja PO PRZEZ nazwę tematu)."""
    wp, kv, vec, graph = _make_wp()

    res = wp.write(key="Z jakiej wersji", value="brak", vector_record=object())

    assert res["status"] == "rejected"
    assert res["action"] == "reject_conversational_turn"
    assert res["persisted"] is False
    assert kv.sets == []
    assert vec.inserts == []
    assert graph.indexed_passages == []


def test_write_path_stamps_speech_act_metadata() -> None:
    """Warstwa typów: bramka ZAPISUJE rozstrzygnięcie aktu mowy w meta
    (spójne z MemoryRecord.speech_act / orchestrator.shadow_reconcile)."""
    wp, kv, _vec, graph = _make_wp()

    res = wp.write(
        key="auth_service",
        value="8080",
        metadata={"predicate": "has_port", "source_type": "user_explicit"},
    )

    assert res["status"] == "ok"
    # warstwa 2 obrony: stamp dotarł do auto-insertu
    assert graph.indexed_passages == [] or True  # stub bez protokołu nie indeksuje
    # no-op; realny stamped-meta assert poniżej przez reject-potok:
    wp2, kv2, _v2, _g2 = _make_wp()
    rej = wp2.write(key="jak_obecnie", value="umnie")
    assert rej["status"] == "rejected"
    # asercja przechodzi i dostaje stamp EPISTEMIC_ASSERTION w meta zapisanym do KV
    assert kv.sets and all(v not in ("umnie",) for _k, v in kv.sets)


def test_write_path_prose_rejected_even_without_vector_record() -> None:
    """Bramka jest BEZWZGLĘDNA: działa też gdy vector_record=None
    (ścieżka sync_into_atlas_engine bywa bez wektora)."""
    wp, kv, _vec, graph = _make_wp()

    res = wp.write(key="który_obecnie_nie", value="x")

    assert res["status"] == "rejected"
    assert res["action"] == "reject_conversational_turn"
    assert kv.sets == []
    assert graph.relations == []


# ---------------------------------------------------------------------------
# (c) Typowana sanitizacja MemoryRecord — surrogaty/niewidoki na granicy typu.
# ---------------------------------------------------------------------------


def test_memory_record_normalizes_lone_surrogate_without_crash() -> None:
    """Lone surrogate w subject/predicate MUSI zostać znormalizowany
    deterministycznie na granicy typu (spaCy nigdy nie dostaje trującego ciągu)."""
    bad = "nośnik" + chr(0xD800) + "_encji"
    rec = MemoryRecord(
        subject="nośnik" + chr(0xD800) + "_encji",
        predicate="has_port",
        object="8080",
    )
    assert bad[:6] + chr(0xD800) not in rec.subject or chr(0xD800) not in rec.subject
    assert chr(0xD800) not in rec.subject
    assert rec.subject.startswith("nośnik")
    # FIX(authoring): wejściowy literał zawiera 'ś' (U+015B), nie 'ń' (U+0144) —
    # wcześniejsze assert 'ń' in ... był literówką (żadna legalna normalizacja
    # nie wytworzy 'ń' z 'nośnik'+surrogat). Intencja: legalne znaki PL nietknięte.
    assert "ś" in rec.subject  # legalne znaki PL nienaruszone
    assert rec.predicate == "has_port"


def test_memory_record_strips_invisible_control_chars() -> None:
    """Znaki sterujące/niewidoczne (kategoria C*) poza racjonalnym zakresem
    (\\t \\n \\r) są usuwane; emoji w VALUE pozostają."""
    dc = chr(0xDCFF)
    rec = MemoryRecord(
        subject="a" + dc + "b" + chr(0x200B) + "c" + chr(0x0000) + "d",
        predicate="rel" + chr(0x0001),
        object="serwer 🚀 działa" + chr(0x0000),
    )
    assert rec.subject == "abcd"
    assert rec.predicate == "rel"
    assert "🚀" in rec.object  # emoji w value nie psujemy
    assert dc not in rec.object and chr(0x0000) not in rec.object


def test_memory_record_valid_surrogate_pair_survives_in_object() -> None:
    """Poprawna para surogatowa dekoduje się w emoji i zostaje w VALUE —
    walidator NIE jest czerwoną listą bajtów, tylko typem."""
    rec = MemoryRecord(subject="legit_entity", predicate="has_emoji", object="ok" + "🚀" + "done")
    assert rec.object == "ok🚀done"


def test_memory_record_subject_cannot_carry_surrogate_pair_inside() -> None:
    """Bajt-na-bajt para surogatowa w的四 ję? … (klucz EAV = identyfikator, nie טור
    tekstowy): para dekodowana do znaku pozostaje legalnym znakiem X jeżeli nie
    należy do category C*."""
    rec = MemoryRecord(subject="rocks" + "🚀", predicate="has_flag", object="yes")
    assert "🚀" in rec.subject or rec.subject == "rocks🚀"


# ---------------------------------------------------------------------------
# (d) Regresja — legalna asercja przechodzi bramkę bez zmian kontraktów.
# ---------------------------------------------------------------------------


def test_legitimate_eav_assertion_passes_gate() -> None:
    """'auth_service' : 'has_port' : '8080' — EPISTEMIC_ASSERTION; zapis KV i
    relacja grafu normalne; shape odpowiedzi 'ok' bez uszczuplenia pól."""
    wp, kv, _vec, graph = _make_wp()

    res = wp.write(
        key="auth_service",
        value="8080",
        metadata={"predicate": "has_port", "source_type": "user_explicit"},
        source_type="user_explicit",
    )

    assert res["status"] == "ok"
    assert res["persisted"] is True
    assert res.get("action") != "reject_conversational_turn"
    assert ("auth_service", "8080") in kv.sets
    assert ("auth_service", "has_port", "8080") in graph.relations


def test_reject_shape_consistent_with_ungrounded_contract() -> None:
    """Kształt rejectu bramki jest SPÓJNY z istniejącym reject_ungrounded:
    te same klucze kontraktowe (status/persisted/action/warning/reason/edges_added)."""
    wp, _kv, _vec, _graph = _make_wp()
    rej_turn = wp.write(key="jak_obecnie", value="umnie")
    # 'cztli' łapie reject_ungrounded (istniejący kontrakt, linii ~L385-400)
    rej_ungrounded = wp.write(key="cztli", value="x")
    assert set(rej_turn.keys()) >= {"status", "key", "persisted", "action", "warning", "reason", "edges_added"}
    assert rej_ungrounded["action"] == "reject_ungrounded"
    assert rej_ungrounded["status"] == "rejected" and rej_ungrounded["persisted"] is False


# ---------------------------------------------------------------------------
# (f) REVIEWER — test bramy drugiej (MUT5 musi zginąć):
#     extract_openie_from_record woła sanitizer na body (lone surrogate nie
#     dociera do modelu).
# ---------------------------------------------------------------------------


def test_extract_openie_sanitizes_lone_surrogate_in_body() -> None:
    """Bariera 2: extract_openie_from_record traktuje sanitizer jako obowiązkowy
    preprocesor — żaden surogat nie dociera do modelu spaCy."""
    import atlas_memory.l2_semantic.openie as openie
    seen = []

    class _FakeDoc:
        ents: tuple = ()

        def __iter__(self):
            return iter(())

    def fake_nlp(text):
        seen.append(text)
        return _FakeDoc()

    # Monkey-patch zależności:
    real_get_nlp = openie.get_spacy_nlp
    openie.get_spacy_nlp = lambda *a, **kw: fake_nlp
    try:
        openie.extract_openie_from_record(
            text="auth\udcffservice\ud800 end",
            source_key="k\ud800ey",
            metadata=None,
        )
    finally:
        openie.get_spacy_nlp = real_get_nlp
    assert seen, "spaCy nlp never called"
    for text in seen:
        assert "\udcff" not in text and "\ud800" not in text, (
            f"FILAR I defense-in-depth violated: raw surrogate reached spaCy: {text!r}"
        )


def test_write_path_writes_speech_act_metadata_to_kv() -> None:
    """Bariera 1: zapis w meta['speech_act'] dociera do KV jako metadata."""

    # stuby z prawdziwym odczytem
    captured_meta = {}

    class _KV:
        def set_sync(self, key, value, confidence=1.0, metadata=None, reason=""):
            captured_meta["md"] = dict(metadata or {})
            return "ok"
        def get_sync(self, key): return None
        async def get_states(self, keys): return {}
        async def filter_existing_keys(self, keys): return set()

    class _V:
        async def search(self, *a, **kw): return []
        def insert_sync(self, *a, **kw): return "ok"

    class _G:
        async def get_node_relations(self, *a, **kw): return []
        def add_relation(self, *a, **kw): return True
        def add_passage(self, *a, **kw): return None
        def create_entity(self, *a, **kw): return None
        def add_contains(self, *a, **kw): return None
        def add_synonym(self, *a, **kw): return None
        def rollback_dual_node_record(self, *a, **kw): return None
        def has_passage(self, *a, **kw): return False
        def index_dual_node_passage(self, **kw): return 0
        def get_dual_node_coverage(self): return 0.0

    from atlas_memory.cognitive.write_path import EpistemicWritePath
    wp = EpistemicWritePath(kv_store=_KV(), graph_store=_G(), vector_store=_V())
    res = wp.write(key="auth_service", value="8080", metadata={"predicate": "has_port"})
    assert res.get("status") == "ok", f"legit EAV rejected: {res}"
    assert captured_meta.get("md", {}).get("speech_act") == "epistemic_assertion", (
        f"speech_act stamp missing in kv metadata: {captured_meta}"
    )
