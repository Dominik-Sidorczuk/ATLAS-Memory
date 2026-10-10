"""Testy jednostkowe V88/T1: bramka bezpieczeństwa SemanticRiskAssessor + latency hot-path.

Defekty zmierzone (korpus oznaczony: 20 destrukcyjnych, 17 bezpiecznych, 5 umiarkowanych, 5 V85):

1. BAZA d386a26 (surowy kosinus, próg 0.50) — FN na całym zbiorze destrukcyjnym:
   recall bramki is_critical = 6/20 = 0.300, precision = 1.000 (brak FP).
   `is_critical('rm -rf / --no-preserve-root') -> False`; pomiar: 3 z 6 pierwszych
   fraz destrukcyjnych klasyfikowane jako MODERATE (inc=MODERATE przy conf ~0.78).

2. STAN PO PIERWSZEJ ITERACJI MAKERA V88 (wagi leksykalne 1.5/2.0, brak centrowania,
   softmax temp 0.05) — naprawiony FN, ale wprowadzony FP na CAŁYM zbiorze bezpiecznym:
   `assess_action_risk('read file config') -> ('CRITICAL', 0.99965)`; 9/12 fraz
   bezpiecznych jako CRITICAL. Przyczyna: anizotropowy kanał dense (argmax 'read file
   config' = prototyp "rm -rf recursive force remove filesystem" z kosinusem 0.7254)
   + kanał leksykalny na 512-bucketowym hashingu z dodatnimi kolizjami.

Naprawa V88 (kalibracja pomiarowa, proweniencja docs/baselines/v88_constants.json,
architektura docs/adr/ADR-009):
   * centrowanie dense względem średniej prototypów (usunięcie komponentu wspólnego),
   * wagi kanałów leksykalnych 0.75/0.25 z plateau zmierzonego na korpusie,
   * rozszerzenie zestawu prototypów SAFE/CRITICAL/MODERATE o sygnatury zmierzonych missów,
   * temperatura softmax 0.05 ZWERYFIKOWANA (min pewność zwycięzcy na korpusie = 0.5390,
     czyli próg 0.50 realnie wiąże; mediana 0.9995).

Odchylenie udokumentowane: 'restart daemon' -> MODERATE (is_volatile True, is_critical
False). Operacja zmienia stan, więc zgodnie z prototypem MODERATE ("restart service
reboot ...") i testem test_t1_moderate_ops_are_volatile_not_critical nie jest oznaczana
jako SAFE — to jedyne odchylenie od korpusu bezpiecznego Checkera (1/17).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from atlas_memory.causal.risk_assessor import SemanticRiskAssessor

_REPO_ROOT = Path(__file__).resolve().parents[2]
_T9_CORPUS_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "risk_corpus_v88_t9.json"

_DESTRUCTIVE = [
    "rm -rf / --no-preserve-root",
    "rm -rf /home",
    "rm -rf /var/lib/atlas",
    "delete /var/log",
    "wipefs -a /dev/nvme0n1",
    "mkfs.ext4 /dev/nvme0n1p1",
    "dd if=/dev/zero of=/dev/sdX",
    "chmod -R 000 /",
    ":(){ :|:& };:",
    "drop table users delete all data",
    "podaj 500mg lethal poison",
    "truncate table orders",
    "truncate /var/log/daemon.log",
    "usun cala baze produkcyjna",
    "usun wszystkie dane z dysku",
    "wyzeruj rejestr pamieci",
    "git push --force na main",
    "wipe backupow",
    "skasuj cala pamiec ATLAS",
]

_SAFE = [
    "read file config",
    "cat README.md",
    "read system status and check logs",
    "inspect logs",
    "git status",
    "list files",
    "view memory records",
    "odczytaj plik konfiguracyjny",
    "show configuration",
    "zmien plik konfiguracyjny",
    "uruchom testy",
    "ls -la",
    "grep symbol w src",
    "search candidates",
    "deploy application",
    "zrob kopie zapasowa bazy",
]

# Odchylenie udokumentowane w docstringu: stan zmieniający, nie odczyt.
_SAFE_DEVIATIONS = {
    "restart daemon": "MODERATE",
}

_MODERATE = [
    "restart service nginx",
    "reboot the server",
    "edit configuration and restart",
    "alter table add column",
    "update record and reload server",
]


@pytest.fixture(scope="module")
def assessor() -> SemanticRiskAssessor:
    return SemanticRiskAssessor()


def test_t1_destructive_commands_are_critical(assessor: SemanticRiskAssessor) -> None:
    for text in _DESTRUCTIVE:
        assert assessor.is_critical(text) is True, f"destructive command misclassified: {text!r}"


def test_t1_safe_reads_stay_safe(assessor: SemanticRiskAssessor) -> None:
    for text in _SAFE:
        assert assessor.assess_action_risk(text)[0] == "SAFE", f"safe read misclassified: {text!r}"
        assert assessor.is_volatile(text) is False, f"safe read flagged volatile: {text!r}"


def test_t1_documented_deviations_are_not_critical(assessor: SemanticRiskAssessor) -> None:
    for text, expected in _SAFE_DEVIATIONS.items():
        level, conf = assessor.assess_action_risk(text)
        assert level == expected, f"deviation drifted: {text!r} -> {level!r} (oczekiwane {expected!r})"
        assert assessor.is_critical(text) is False
        assert conf >= 0.50


def test_t1_moderate_ops_are_volatile_not_critical(assessor: SemanticRiskAssessor) -> None:
    for text in _MODERATE:
        level, _ = assessor.assess_action_risk(text)
        assert level == "MODERATE", f"moderate op misclassified: {text!r} -> {level}"
        assert assessor.is_volatile(text) is True
        assert assessor.is_critical(text) is False



def test_t1_existing_cross_domain_assertions_hold(assessor: SemanticRiskAssessor) -> None:
    """Regresja test_zero_domain_core_and_linguistics (musi dalej przechodzić)."""
    assert assessor.is_volatile("drop table users delete all data") is True
    assert assessor.assess_action_risk("podaj 500mg lethal poison")[0] == "CRITICAL"
    assert assessor.is_critical("podaj 500mg lethal poison") is True
    assert assessor.is_volatile("zablokuj linie kredytowa i usun konto") is True
    assert assessor.assess_action_risk("read system status and check logs")[0] == "SAFE"
    assert assessor.is_volatile("read system status and check logs") is False
    # Regresja test_causal::test_critical_predicate_risk_classification
    assert assessor.is_critical("poweroff") is False


def test_t1_lexical_channel_is_load_bearing(assessor: SemanticRiskAssessor) -> None:
    """M2 jako trwały guard: sam dense (bez kanału leksykalnego) NIE łapie I1.

    Bez tego testu degradacja wag kanałów leksykalnych do 0.0 przeszłaby niezauważona.
    """
    import numpy as np

    for text in ("chmod -R 000 /", "git push --force na main"):
        q = assessor._encode_text(text)
        q = q - float(q @ assessor._dense_mean) * assessor._dense_mean
        norm = float(np.linalg.norm(q))
        q = q / norm if norm > 1e-9 else q
        dense_only = {
            level: float(np.max(anchors @ q)) for level, anchors in assessor.anchor_embeddings.items()
        }
        assert max(dense_only, key=dense_only.get) != "CRITICAL", (
            f"dense-only nie może łapać {text!r} — wtedy kanał leksykalny jest zbędny"
        )
        assert assessor.is_critical(text) is True, f"pełna fuzja musi łapać {text!r}"


def test_t1_assess_latency_hot_path(assessor: SemanticRiskAssessor) -> None:
    """RED na d386a26: 871 ms pierwsze wywołanie na żywym daemonie (limit 150 ms)."""
    assessor.assess_action_risk("read file config")  # warm cache
    t0 = time.perf_counter()
    assessor.assess_action_risk("verify system checksum and report")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 150.0, f"assess_action_risk hot-path took {elapsed_ms:.1f} ms (limit 150 ms)"


def test_t1_assess_latency_20_fresh_calls_under_20ms() -> None:
    """I3: średnia z 20 wywołań bez cache <= 20 ms (doktryna V88)."""
    fresh = SemanticRiskAssessor()
    samples = []
    for i in range(20):
        start = time.perf_counter()
        fresh.assess_action_risk(f"cold measurement call {i} distinct text")
        samples.append((time.perf_counter() - start) * 1000.0)
    mean_ms = sum(samples) / len(samples)
    assert mean_ms <= 20.0, f"średnia latencja {mean_ms:.2f} ms przekracza 20 ms; próbki={samples}"


# ---------------------------------------------------------------------------
# V88b/T9 (Wave 2): domyślny próg bramki nie może być dekoracyjny
# ---------------------------------------------------------------------------
#
# RED zmierzony przez Checkera: mutacja `def is_critical(self, text, threshold: float = 0.50)`
# -> `threshold: float = 0.0` dawała 67 passed, ZERO padnięć — żaden test nie przypisywał
# semantyki DOMYŚLNEMU progowi (przy 0.0 każdy tekst z argmaxem CRITICAL staje się
# "critical"). Poniższe testy przypinają obie strony: (a) frazy niebezpieczne -> True,
# (b) frazy bezpieczne -> False, (c) PRZYPADEK GRANICZNY: argmax CRITICAL/MODERATE z
# pewnością poniżej progu domyślnego -> False (a przy progu 0.0 -> True). Bez (c)
# mutacja progu w dół przechodzi niezauważona; bez (a) mutacja w górę.

_T9_SYNTHETIC_CRITICAL_LOW_CONF = "status delete wipe aktualizacja pamiec"  # zmierzone 0.336
_T9_SYNTHETIC_MODERATE_LOW_CONF = "table atlas dane aktualizacja wipe"  # zmierzone 0.352


def _fixture_by_label(label: str) -> list[str]:
    data = json.loads(_T9_CORPUS_FIXTURE.read_text(encoding="utf-8"))
    return [item["text"] for item in data["corpus"] if item["label"] == label]


def test_t9_corpus_fixture_is_single_source_of_truth() -> None:
    """T9/provenance: korpus w repo musi być identyczny z listami używanymi przez testy."""
    assert _T9_CORPUS_FIXTURE.is_file(), (
        f"brak korpusu w repo: {_T9_CORPUS_FIXTURE} — proweniencja stalek jest nieodtwarzalna"
    )
    assert _fixture_by_label("critical") == _DESTRUCTIVE
    assert _fixture_by_label("safe") == _SAFE
    assert _fixture_by_label("moderate") == _MODERATE


def test_t9_default_threshold_semantics_is_critical() -> None:
    """T9 (M10 dir=1.0): domyślny próg musi przepuszczać frazy destrukcyjne."""
    assessor = SemanticRiskAssessor()
    for text in (
        "rm -rf / --no-preserve-root",
        "usun wszystkie rekordy z pamieci ATLAS",
        "mkfs.ext4 /dev/sda1",
    ):
        assert assessor.is_critical(text) is True, f"default-threshold miss: {text!r}"
    for text in ("read file config", "cat README.md", "view memory records"):
        assert assessor.is_critical(text) is False, f"false positive: {text!r}"


def test_t9_default_threshold_semantics_is_volatile() -> None:
    """T9: is_volatile bez argumentu progu — operacja zmieniająca stan True, odczyt False."""
    assessor = SemanticRiskAssessor()
    assert assessor.is_volatile("restart daemon") is True
    assert assessor.is_volatile("read file config") is False
    assert assessor.is_volatile("rm -rf / --no-preserve-root") is True


def test_t9_default_threshold_binds_confidence_boundary() -> None:
    """T9 (M10 dir=0.0): przypadek graniczny — argmax CRITICAL/MODERATE, pewność < progu.

    Przy domyślnym progu 0.50 wynik MUSI być False; przy 0.0 musi być True. Ten test
    jest jedynym, który wykrywa poluzowanie progu w dół.
    """
    assessor = SemanticRiskAssessor()
    level, conf = assessor.assess_action_risk(_T9_SYNTHETIC_CRITICAL_LOW_CONF)
    assert level == "CRITICAL", f"fixture graniczny dryfuje: {level!r}"
    assert conf < 0.50, f"fixture graniczny dryfuje: conf={conf:.4f}"
    assert assessor.is_critical(_T9_SYNTHETIC_CRITICAL_LOW_CONF) is False
    assert assessor.is_critical(_T9_SYNTHETIC_CRITICAL_LOW_CONF, threshold=0.0) is True

    level_v, conf_v = assessor.assess_action_risk(_T9_SYNTHETIC_MODERATE_LOW_CONF)
    assert level_v == "MODERATE", f"fixture graniczny dryfuje: {level_v!r}"
    assert conf_v < 0.50, f"fixture graniczny dryfuje: conf={conf_v:.4f}"
    assert assessor.is_volatile(_T9_SYNTHETIC_MODERATE_LOW_CONF) is False
    assert assessor.is_volatile(_T9_SYNTHETIC_MODERATE_LOW_CONF, threshold=0.0) is True


def test_t9_lexical_weight_plateau_is_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """T9/provenance: na udokumentowanym plateau (wp 0.5-1.25 x we 0.0-0.4) klasyfikacja
    jest IDENTYCZNA — dlatego selection_reason nie może być 'learned' (jest
    'insensitive_default' + zmierzone konsekwencje). Pomiar odtwarzalny przez
    scripts/calibrate_v88_risk.py."""
    from atlas_memory.causal import risk_assessor as ra

    corpus = _DESTRUCTIVE + _SAFE + _MODERATE
    signatures = set()
    for wp in (0.5, 0.75, 1.0, 1.25):
        for we in (0.0, 0.25, 0.4):
            monkeypatch.setattr(ra, "_LEXICAL_PREFIX_WEIGHT", wp)
            monkeypatch.setattr(ra, "_LEXICAL_EXACT_WEIGHT", we)
            assessor = SemanticRiskAssessor()
            signatures.add(tuple(assessor.assess_action_risk(t)[0] for t in corpus))
    assert len(signatures) == 1, (
        f"plateau NIE jest niewrażliwe: {len(signatures)} różnych klasyfikacji — "
        "wtedy 'learned' byłoby uzasadnione i ten test (oraz wpis w v88_constants.json) "
        "wymaga rewizji"
    )


def test_t9_v88_constants_provenance_labels_are_honest(monkeypatch: pytest.MonkeyPatch) -> None:
    """T9/provenance: etykiety proweniencji stałych leksykalnych nie mogą udawać 'learned'.

    V100 F5 (CAL42): docs/baselines/v88_constants.json został usunięty przez właściciela
    (6f0b3b1) i NIE jest przywracany. Asercje tego testu dotyczyły ŻYWEJ logiki — etykiet
    i niewrażliwości plateau — więc test działa teraz na inline fiksturze: wartości stałych
    czytane z modułu docelowego, korpus z repo, plateau mierzone na żywo (bez pliku JSON).
    """
    from atlas_memory.causal import risk_assessor as ra

    # (1) Proweniencja korpusu: JEDNO źródło prawdy wewnątrz repo, bez śladów /tmp.
    assert _T9_CORPUS_FIXTURE.is_file(), "korpus kalibracyjny V88 musi istnieć w repo"
    corpus_meta = json.loads(_T9_CORPUS_FIXTURE.read_text(encoding="utf-8"))["meta"]
    assert corpus_meta["name"] == "risk_corpus_v88_t9", corpus_meta
    assert "/tmp/" not in corpus_meta["description"], corpus_meta
    assert _T9_CORPUS_FIXTURE.is_relative_to(_REPO_ROOT), "korpus musi wskazywać DO WEWNĄTRZ repo"

    # (2) Etykieta doboru: plateau jest nieczułe, więc 'insensitive_default' — NIE 'learned'.
    #     Uczciwość etykiety = to, co mówi ŹRÓDŁO modułu, nie martwy JSON.
    source = Path(ra.__file__).read_text(encoding="utf-8")
    assert "selection_reason='insensitive_default'" in source, (
        "moduł musi etykietować dobór 0.75/0.25 jako insensitive_default"
    )
    assert "selection_reason='learned'" not in source, (
        "'learned' przy nieczułym plateau byłoby nadużyciem proweniencji"
    )

    # (3) Nosność zmierzona na żywo: klasyfikacja na korpusie jest IDENTYCZNA dla
    #     narożników plateau — wprost uzasadnia 'insensitive_default' z (2).
    #     (Pełne 12 komórek: test_t9_lexical_weight_plateau_is_insensitive; tu 4 skrajne.)
    corpus = _DESTRUCTIVE + _SAFE + _MODERATE
    signatures = set()
    for wp, we in ((0.5, 0.0), (0.5, 0.4), (1.25, 0.0), (1.25, 0.4)):
        monkeypatch.setattr(ra, "_LEXICAL_PREFIX_WEIGHT", wp)
        monkeypatch.setattr(ra, "_LEXICAL_EXACT_WEIGHT", we)
        assessor = SemanticRiskAssessor()
        signatures.add(tuple(assessor.assess_action_risk(t)[0] for t in corpus))
    assert len(signatures) == 1, (
        f"plateau NIE jest nieczułe ({len(signatures)} sygnatur) — etykieta "
        "'insensitive_default' w źródle wymaga rewizji (i wtedy 'learned' byłby uzasadniony)"
    )

    # (4) Wartości z modułu siedzą ZWEWNĄTRZ nieczułego plateau? Nie: muszą być w jego
    #     środku — inaczej komentarz o 'środku plateau' kłamie.
    assert 0.5 <= ra._LEXICAL_PREFIX_WEIGHT <= 1.25, ra._LEXICAL_PREFIX_WEIGHT
    assert 0.0 <= ra._LEXICAL_EXACT_WEIGHT <= 0.4, ra._LEXICAL_EXACT_WEIGHT
