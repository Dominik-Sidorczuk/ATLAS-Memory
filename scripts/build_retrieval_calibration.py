#!/usr/bin/env python3
"""Build Retrieval Calibration Dataset (V92.1 — realna pula hybrydowa z Qdranta).

DEFECT (naprawiany w V92.1): do V92 pula kandydatów była SZTUCZNA — generator liczył
`doc_matrix @ q_vec` po WSZYSTKICH 136 faktach i dopiero to wkładał do
`score_candidates(fusion_strategy="dbsf")`. Demon zbiera pulę zupełnie inaczej:
`candidates.gather_candidates` → `QdrantVectorStore.search_hybrid_async`
(prefetch dense+sparse, `Fusion.DBSF` po stronie silnika, limit ~50, bramka min_vec,
filtr `_is_filtered`). `score_top1` z datasecia nie był więc rozdzielony jak
`score_top1` demona (covariate shift) i learned progi były nieprzenoszalne.

V92.1: pula kandydatów = DOKŁADNIE produkcyjna ścieżka na kopii magazynu prod
(dawniej `/tmp/qs_prod_after`; od V97 FALLBACK_QDRANT_PATH leży w trwałym
`~/atlas-run/calib_cache`, patrz F5), kolekcja `hermes_episodic_memory_68m`: realny wynik Qdranta
jest `vector_score` kandydata (tak jak w candidates.py:204), potem ten sam
`score_candidates(dbsf)`. Cosinus zostaje WYŁĄCZNIE do identyfikacji bliźniaków
(zbiór eq_sets) przy doborze twardych negatywów — nigdy do wartości `score_top1`.

Zachowane bramki (bez zmian):
  C1 GroupKFold po target_key, seed=42, zerowe przecięcie kluczów train/test;
  C2 query = first_sentence(faktu) dla prose, czysty klucz dla symbolic (self_query
     wykluczone);
  C3 deterministyczne twarde negatywy (cross-family / leksykalny near-miss) spoza
     eq_sets, test negatives >= 30 na typ, udział pozytywów 40-60%, 5 foldów.

Skala progu: 0.05-1.10 (patrz ADR-011; docs/adr/ADR-011-calibration-dbsf-rescale.md).

Tryb `--source matrix` odtwarza STARY generator (sztuczna macierz) wyłącznie jako
kontrolę/dowód dystrybucyjny — NIE jest ścieżką produkcyjną.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.append(str(REPO / "src"))

import numpy as np

from atlas_memory.cognitive.calibration_engine import is_symbolic_or_identifier
from atlas_memory.cognitive.retrieval.candidates import (
    PROBE_PREFIXES,
    _is_filtered,
    gather_candidates,
    is_conversational_churn,
)
from atlas_memory.cognitive.retrieval.score import (
    _bilingual_stem,
    compute_content_fingerprint,
    get_record_authority_score,
    score_candidates,
)
from atlas_memory.l1_working.hot_buffer import HotBufferPolicy
from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore
from atlas_memory.l2_semantic.onnx_encoder import OnnxDenseEncoder, resolve_model_dir
from atlas_memory.l2_semantic.qdrant_store import QdrantVectorStore
from atlas_memory.l2_semantic.textnorm import normalize_stored_text
from atlas_memory.l2_semantic.vector_backend import load_vector_backend

DOCS_CALIB_DIR = REPO / "docs" / "calib"
DOCS_CALIB_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DB = Path.home() / ".hermes" / "atlas" / "atlas.db"

# F5 (V97, BUG-CAL5/CAL-8): cache kalibracji NIE wolno trzymać na tmpfs — reboot kasuje
# /tmp i historia zna już utratę ~50 h pracy GPU (18.09). Wszystkie DOMYŚLNE ścieżki cache
# idą do trwałego katalogu pod $HOME; każdy da się nadpisać zmienną środowiskową albo flagą CLI.
# Zero bezwzględnych literałów katalogu domowego — ścieżka liczona z Path.home() (wymóg repo).
CALIB_CACHE_DIR = Path.home() / "atlas-run" / "calib_cache"
try:
    CALIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
except OSError:  # środowisko tylko do odczytu: zapis i tak dostanie jawny błąd przy użyyciu
    pass


def _migrate_legacy_cache(dest: Path, legacy_path: str) -> None:
    """Jednorazowo KOPIUJE stary cache z tmpfs do trwałego katalogu (F5 V97).

    Tylko gdy `dest` jeszcze nie istnieje, a legacy plik jest na dysku: ciepzy cache
    (10 MB puł hybrydowych, 14 MB kopii Kùzu) nie przepada przy zmianie domyślnej
    ścieżki. Kopiowane, nie przenoszone — /tmp i tak wygaśnie samo. Zero LLM/GPU/sieci.
    """
    try:
        legacy = Path(legacy_path)
        if dest.exists() or not legacy.is_file():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, dest)
        print(f"[F5] zmigrowano cache z tmpfs: {legacy} -> {dest}")
    except Exception:
        pass  # migracja jest Convenience; brak starego pliku nie może wywrzeć biegu


DEFAULT_DATASET_V93 = DOCS_CALIB_DIR / "calibration_dataset_v93.jsonl"
OUTPUT_JSONL = DEFAULT_DATASET_V93
DEFAULT_HYBRID_CACHE = CALIB_CACHE_DIR / "atlas_calib_hybrid_pool_v92.json"
HYBRID_CACHE = Path(os.getenv("ATLAS_CALIB_HYBRID_CACHE", str(DEFAULT_HYBRID_CACHE)))
_migrate_legacy_cache(HYBRID_CACHE, "/tmp/atlas_calib_hybrid_pool_v92.json")
JUDGE_ERRORS_FILE = DOCS_CALIB_DIR / "judge_errors.jsonl"
JUDGE_STATS_FILE = DOCS_CALIB_DIR / "judge_stats.json"
EQUIV_THRESHOLD = 0.95



def _decode_value(raw: Any) -> str:
    """Value bywa: (a) JSON-obiektem z 'value', (b) surowym JSON-stringiem z escapem
    unicode (backslash-u + 4 cyfry), (c) plain text. Oddaje tekst bez literalnego escape."""
    text = str(raw)
    for _ in range(3):
        try:
            d = json.loads(text)
        except Exception:
            return text.strip()
        if isinstance(d, dict):
            inner = str(d.get("value") or d.get("text") or d.get("content") or "")
            if inner == text:
                return inner.strip()
            text = inner
            continue
        if isinstance(d, str):
            if d == text:
                return d.strip()
            text = d
            continue
        return text.strip()
    return text.strip()


def load_facts(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = db_path or DEFAULT_DB
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = con.execute("SELECT key, value, confidence FROM state_variables").fetchall()
    con.close()
    out = []
    for key, raw, conf in rows:
        text = _decode_value(raw)
        if len(text.split()) < 8 or "." not in text:
            continue
        out.append({"key": key, "text": text, "confidence": conf})
    return out


def first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text)
    for p in parts:
        clean = p.strip()
        if len(clean.split()) >= 5 and clean != text:
            return clean
    words = text.split()
    head = " ".join(words[:18])
    return head if head != text else " ".join(words[:12])


def equiv_sets(doc_vecs: np.ndarray, threshold: float = EQUIV_THRESHOLD) -> List[set[int]]:
    """Dla każdego dokumentu zbiór indeksów bliźniaków (cos > threshold), z nim samym."""
    m = np.asarray(doc_vecs, dtype=np.float64)
    sim = m @ m.T
    np.fill_diagonal(sim, -1.0)
    return [set(np.nonzero(sim[i] > threshold)[0].tolist()) | {i} for i in range(len(m))]


# ==============================================================================
# SĘDZIA METROLOGICZNY V96 ONE-SHOT (5 Wymiarów + Przykłady Wzorcowe + Weta)
# ==============================================================================
METROLOGICAL_SYSTEM_PROMPT = """Jesteś rygorystycznym, metrologicznym sędzią adekwatności wyszukiwania semantycznego (Metrological LLM Judge) dla systemu pamięci epistemicznej agenta AI.
Twoim zadaniem jest precyzyjna, wieloaspektowa ewaluacja stopnia przydatności zwróconego faktu (kandydata) z bazy wiedzy względem zadanego zapytania.

ZASADA NADRZĘDNA — OBSERWOWALNOŚĆ I ZAKAZ HALLUCYNOWANIA:
(a) Oceniasz WYŁĄCZNIE na podstawie dwóch tekstów, które widzisz: [ZAPYTANIE] i [KANDYDAT]. Nie masz dostępu do świata zewnętrznego, bazy wiedzy ani metadanych — NIE weryfikuj prawdziwości faktów względem wiedzy o świecie; karz wyłącznie za sprzeczności WEWNĘTRZNE (kandydat kontra zapytanie, kandydat kontra sam siebie).
(b) Jeżeli wymiar jest nierozstrzygalny z tekstu, przyznaj wartość neutralną (środek skali) i napisz to wprost w rationale. Zgadywanie to błąd pomiaru, nie ostrożność.
(c) Rationale: maksimum jedno zdanie, zakotwiczone w cytowalnym fragmencie tekstu. Ocena bez tekstowego uzasadnienia jest nieważna.

Dokonaj oceny w 5 ortogonalnych wymiarach oraz sprawdź twardą bramkę weta:

1. semantic_entailment (skala 0-3):
   - 0: Całkowity brak związku lub powierzchowna zbieżność pojedynczych słów (keyword overlap).
   - 1: Zbieżność tematyczna, ale fakt NIE odpowiada na pytanie (inny wątek w tej samej dziedzinie).
   - 2: Odpowiedź częściowa lub implikacja pośrednia (wymaga dedukcji).
   - 3: Pełne, bezpośrednie wynikanie logiczne (fakt wprost odpowiada na pytanie).

2. operational_actionability (skala 0-3, warunkowane typem zapytania):
   - Dla zapytań wykonawczych/symbolicznych:
     - 0: Pusta narracja, brak jakiejkolwiek wartości operacyjnej.
     - 1: Ogólnikowa deklaracja intencji bez konkretnych parametrów.
     - 2: Częściowa instrukcja (brak kluczowych flag, ścieżek lub wartości).
     - 3: Ścisła reguła wykonawcza, deterministyczny invariant, komenda z flagami lub precyzyjny parametr.
   - Dla zapytań informacyjnych/prose:
     - 0: Brak treści informacyjnej, niespójny fragment.
     - 1: Kontekst ogólny bez konkretnych danych merytorycznych.
     - 2: Przydatna informacja merytoryczna lub kontekstowa.
     - 3: Pełna, wyczerpująca informacja faktograficzna wprost odpowiadająca na intencję.

3. entity_scope_grounding (skala 0-2):
   - 0: Pomylona encja bazowa lub obcy moduł (np. reguła demona w zapytaniu o providera).
   - 1: Pokrewna encja w tym samym podsystemie (np. caller vs callee).
   - 2: Ścisła tożsamość encji, interfejsu, prefiksu i warstwy logicznej.

4. polarity_fidelity (skala 0-2):
   - 0: INWERSJA REGUŁY / PRZEKŁAMANIE MODALNE (zakaz zamieniony w nakaz, negacja warunku brzegowego).
   - 1: Zgodność warunkowa (reguła prawdziwa tylko pod specyficznymi warunkami brzegowymi).
   - 2: Bezwzględna spójność modalna i logiczna (zakaz pozostaje zakazem, ograniczenie jest zachowane).

5. epistemic_freshness (skala 0-2, WYŁĄCZNIE sygnały widoczne w tekście):
   - 0: Tekst sam deklaruje wycofanie/unieważnienie/roboczy status (np. "wycofane", "deprecated", "unieważnione", "scratchpad", "draft") LUB kandydat wprost odwołuje fakt, który zapytanie traktuje jako obowiązujący.
   - 1: Brak jakichkolwiek sygnałów wersjonowania w tekście — DOMYŚLNA wartość neutralna.
   - 2: Tekst wprost deklaruje aktualność (wskazuje bieżący stan/mechanizm systemu i nie zawiera sygnałów wycofania).

6. hard_veto:
   - detected: true jeśli fakt zawiera jawną sprzeczność wewnętrzną z treścią zapytania (np. zapytanie zakłada zakaz, kandydat go odwraca) LUB kandydat sam w sobie jest wewnętrznie sprzeczny modalnie. W przeciwnym razie false.
   - reason: zwięzłe wyjaśnienie weta lub 'brak sprzeczności'.

ZASADA WAG (informacyjnie; wagi liczy pipeline): największy wpływ ma semantic_entailment, potem operational_actionability; epistemic_freshness ma wpływ najmniejszy — NIE podbijaj świeżości kosztem semantyki i nie wynajduj sygnałów świeżości, których nie ma w tekście.

---

PRZYKŁADY WZORCOWE (FEW-SHOT):

Przykład 1 — Pełny pozytyw (semantic_entailment = 3):
[TYP ZAPYTANIA]: prose
[ZAPYTANIE]:
Jakie porty UDS i RPC są używane przez demona ATLAS?
[KANDYDAT]:
Demon ATLAS nasłuchuje wyłącznie na gnieździe Unix Domain Socket ~/.hermes/atlas.sock (IPC przez UDS, bez portu TCP).
[ODPOWIEDŹ]:
{
  "semantic_entailment": {"score": 3, "rationale": "Kandydat wprost i kompletnie odpowiada na pytanie o gniazdo nasłuchu demona ATLAS."},
  "operational_actionability": {"score": 3, "rationale": "Podaje dokładną ścieżkę socketu UDS gotową do użycia w konfiguracji."},
  "entity_scope_grounding": {"score": 2, "rationale": "Ścisła tożsamość encji (demon ATLAS) i warstwy komunikacji IPC."},
  "polarity_fidelity": {"score": 2, "rationale": "Zakaz portu TCP zachowany bez inwersji."},
  "epistemic_freshness": {"score": 2, "rationale": "Tekst deklaruje bieżący mechanizm IPC, brak sygnałów wycofania."},
  "hard_veto": {"detected": false, "reason": "brak sprzeczności"}
}

Przykład 2 — Twardy negatyw z tej samej dziedziny (semantic_entailment = 1):
[TYP ZAPYTANIA]: prose
[ZAPYTANIE]:
Jakie porty UDS i RPC są używane przez demona ATLAS?
[KANDYDAT]:
Prefetch pamięci w atlas_provider.py odpytuje gniazdo UDS z domyślnym limitem rekordów i budżetem tokenów.
[ODPOWIEDŹ]:
{
  "semantic_entailment": {"score": 1, "rationale": "Ta sama dziedzina (UDS w ATLAS), ale opisuje prefetch providera, a nie adres nasłuchu demona."},
  "operational_actionability": {"score": 2, "rationale": "Kontekst operacyjny, lecz nie odpowiada na zadane pytanie."},
  "entity_scope_grounding": {"score": 1, "rationale": "Pokrewna encja (provider zamiast demona) w tym samym podsystemie IPC."},
  "polarity_fidelity": {"score": 2, "rationale": "Brak inwersji modalnej."},
  "epistemic_freshness": {"score": 1, "rationale": "Brak sygnałów wersjonowania w tekście — wartość neutralna."},
  "hard_veto": {"detected": false, "reason": "brak sprzeczności"}
}
---

Zwróć WYŁĄCZNIE poprawny obiekt JSON w formacie:
{
  "semantic_entailment": {"score": 0..3, "rationale": "krótkie uzasadnienie"},
  "operational_actionability": {"score": 0..3, "rationale": "krótkie uzasadnienie"},
  "entity_scope_grounding": {"score": 0..2, "rationale": "krótkie uzasadnienie"},
  "polarity_fidelity": {"score": 0..2, "rationale": "krótkie uzasadnienie"},
  "epistemic_freshness": {"score": 0..2, "rationale": "krótkie uzasadnienie"},
  "hard_veto": {"detected": false, "reason": "brak sprzeczności"}
}"""

METROLOGICAL_USER_PROMPT = """[TYP ZAPYTANIA]: {qt}

[ZAPYTANIE]:
{q}

[KANDYDAT (ZAPAMIĘTANY FAKT)]:
{c}

Oceń przydatność faktu wg 5 wymiarów metrologicznych i zwróć poprawny obiekt JSON."""

DEFAULT_JUDGE_CACHE = DOCS_CALIB_DIR / "cache_v93_sc.jsonl"
LEGACY_JUDGE_CACHE = DOCS_CALIB_DIR / "cache_v93.jsonl"
JUDGE_CACHE_FILE = Path(os.getenv("ATLAS_JUDGE_CACHE", str(DEFAULT_JUDGE_CACHE)))


def resolve_judge_cache_path(cache_path: Optional[Path] = None) -> Path:
    """JEDYNE źródło prawdy dla ścieżki cache sedziego (V97 F7 / BUG-CAL34).

    Przyczyna bugu: `AsyncRetrievalJudge.__init__` czytal `os.getenv("ATLAS_JUDGE_CACHE",
    DEFAULT_JUDGE_CACHE)` bezposrednio, wiec `patch("...JUDGE_CACHE_FILE", tmp_path)` w
    testach był dekoracyjny — sam bieg pytesta dopisywał syntetyczne wpisy
    (`q:"query_t10"`, `model:"local"`) do TRACKOWANEGO `docs/calib/cache_v93_sc.jsonl`,
    czyli do artefaktu proweniencji kalibracji.

    Priorytet: jawny argument `cache_path` -> env `ATLAS_JUDGE_CACHE` -> BIEŻĄCA wartość
    atrybutu modułu `JUDGE_CACHE_FILE` (patchowalnego w testach) -> default repo.
    """
    if cache_path:
        return Path(cache_path)
    env_path = os.getenv("ATLAS_JUDGE_CACHE")
    if env_path:
        return Path(env_path)
    return Path(globals().get("JUDGE_CACHE_FILE", DEFAULT_JUDGE_CACHE))


JUDGE_ENDPOINT = os.getenv("ATLAS_JUDGE_URL", "http://127.0.0.1:20122/v1/chat/completions")
JUDGE_MODEL = os.getenv("ATLAS_JUDGE_MODEL", "local")
JUDGE_PROMPT_VERSION = "v102_observable_neutral_r4"
# V100 F1 (CAL37): logger do głośnego odrzucania proweniencji spoza wersji promptu.
logger = logging.getLogger("atlas.calibration.retrieval")
DEFAULT_JUDGE_TIMEOUT = 420
DEFAULT_SC_MODE = os.getenv("ATLAS_SC_MODE", "batch")


# T13/F4 (V97): próg wariancji unstable dla K=3 self-consistency — wartość policzona
# z ROZKŁADU ZAPISANEGO NA DYSKU, nie z przepisanej stałej. Poprzednia wartość 0.025
# miała fałszywą proweniencję ("P95, n=40, docs/calib/sc_sigma_smoke40.json"): plik ma
# w rzeczywistości N=11 (a jego P95 = 0.137), artefakt n=117 ani wartość 0.0546 nie
# istnieją na dysku. Jako proweniencja NIE używam cache_v93_sc.jsonl (jego N rośnie przy
# każdym biegu sędziego — 70 w bazie commita, 76 w tree), tylko zapisanego artefaktu:
# z pliku docs/calib/sc_sigma_smoke40.json, N=11, P95=0.13705
# (P90=0.0033, max=0.2708; test: test_v97_f4_threshold_equals_independent_p95_from_disk_artifact)
# Uwaga: przy N=11 P95 to efektywnie niemal max — to UCZCIWA wartość z dysku, a nie 0.0546.
#
# V98 F9 (BUG-CAL35): powyższa stała jest teraz TYLKO AWARYJNYM FALLBACKIEM — opisuje
# rozkład N=11, którego nazwa pliku kłamie. Realny próg pochodzi z pliku proweniencji
# `docs/calib/sigma2_threshold_provenance.json` generowanego przez
# `scripts/recompute_sigma2_threshold.py` (liczy P95 z pełnego korpusu cache i zapisuje
# source_file/n/p95/computed_at/sha256). Jeśli pliku proweniencji NIE MA, runner
# świadomie używa poniższej stałej i mówi o tym wprost (brak świeżego pomiaru).
UNSTABLE_SIGMA2_THRESHOLD_DEFAULT = 0.13705
# Artefakt proweniencji progu (F4 V97): zapisany plik rozkładu, nieapendowany biegami.
SIGMA2_PROVENANCE_FILE = DOCS_CALIB_DIR / "sc_sigma_smoke40.json"
# V98 F9: proweniencja PROGU (P95 z realnego pomiaru) — wynik skryptu, nie ręczna liczba.
SIGMA2_THRESHOLD_PROVENANCE_FILE = DOCS_CALIB_DIR / "sigma2_threshold_provenance.json"
SIGMA2_THRESHOLD_PROVENANCE_FIELDS: Tuple[str, ...] = (
    "source_file", "n", "p95", "computed_at", "sha256",
)


def load_sigma2_distribution(path: Optional[Path] = None) -> List[float]:
    """Rozkład σ² self-consistency Z ARTEFAKTU NA DYSKU (zero LLM, zero GPU).

    Czyta wpisy `sigma2` (K=3 independent, latenty w `indep_latents`) w kolejności pliku.
    Służy do odtworzenia progu `UNSTABLE_SIGMA2_THRESHOLD_DEFAULT` razem z realnym N —
    dawniej komentarz w runnerze podawał N=40/117, których na dysku nie ma.

    V98 F9: `path` może wskazywać na plik proweniencji progu (dict z `p95`); jeśli niósł
    on zapisany rozkład (`distribution`), zwracamy go — inaczej czytamy klasyczny plik
    z listą rekordów (`[{... "sigma2": ...}]`).
    """
    src = Path(path or SIGMA2_PROVENANCE_FILE)
    if not src.is_file():
        return []
    try:
        rows = json.loads(src.read_text(encoding="utf-8"))
    except Exception:
        return []
    # Plik proweniencji (F9): dict — rozkład tylko wtedy, gdy został w nim zapisany.
    if isinstance(rows, dict):
        dist = rows.get("distribution")
        return [float(v) for v in dist] if isinstance(dist, list) else []
    return [float(r["sigma2"]) for r in rows if isinstance(r, dict) and r.get("sigma2") is not None]


def sha256_file(path: Path) -> str:
    """SHA-256 pliku (do weryfikacji, czy proweniencja nadal opisuje ten sam corpus)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sigma2_threshold_provenance(
    path: Optional[Path] = None, verify_source: bool = True
) -> Optional[Dict[str, Any]]:
    """Proweniencja progu σ² (wynik `scripts/recompute_sigma2_threshold.py`) lub None.

    None = Brak pomiaru na dysku: plik nie istnieje, jest niekompletny (brak któregoś
    z pól SIGMA2_THRESHOLD_PROVENANCE_FIELDS) albo jego `sha256` nie opisuje już pliku
    źródłowego (korpus urósł od czasu pomiaru — próg jest PRZETERMINOWANY i nie ufac mu).
    """
    src = Path(path or globals().get("SIGMA2_THRESHOLD_PROVENANCE_FILE", SIGMA2_THRESHOLD_PROVENANCE_FILE))
    if not src.is_file():
        return None
    try:
        doc = json.loads(src.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(doc, dict):
        return None
    if any(k not in doc or doc[k] is None for k in SIGMA2_THRESHOLD_PROVENANCE_FIELDS):
        return None
    if doc.get("percentile") not in (None, 95, 95.0):
        return None
    if float(doc["n"]) <= 0 or doc.get("p95") is None:
        return None
    if verify_source:
        source = Path(doc["source_file"])
        if not source.is_absolute():
            source = REPO / doc["source_file"]
        if not source.is_file():
            return None
        if sha256_file(source) != doc["sha256"]:
            return None
    return doc


def resolve_unstable_sigma2_threshold(path: Optional[Path] = None) -> Optional[float]:
    """Próg unstable: P95 Z PLIKU PROWENIENCJI (V98 F9), awaryjnie stała sprzed pomiaru.

    NIE ustawiamy progu ręcznie — bierzemy `p95` zmierzony i zapisany przez
    `scripts/recompute_sigma2_threshold.py`. Fallback na `UNSTABLE_SIGMA2_THRESHOLD_DEFAULT`
    oznacza BRAK ŚWIEŻEGO POMIARU i jest odnotowany na stderr, żeby nie udawać metryki.

    V100 F1 (CAL37): próg σ² jest wiązany z wersją promptu sędziego. Proweniencja bez
    pola `judge_prompt_version` albo z niezgodną wersją jest ODRZUCONA — resolver zwraca
    None z głośnym logger.warning; konsumenci wtedy świadomie spadają na stałą
    awaryjną (UNSTABLE_SIGMA2_THRESHOLD_DEFAULT), zamiast bezszelestnie ufać pomiarowi
    z innej generacji promptu.
    """
    doc = load_sigma2_threshold_provenance(path)
    if doc is None:
        if path is None:
            print(
                "[sigma2-prog] brak ważnej proweniencji "
                f"({Path(SIGMA2_THRESHOLD_PROVENANCE_FILE).name}) — używam stałej awaryjnej "
                f"{UNSTABLE_SIGMA2_THRESHOLD_DEFAULT} (NIE ZMIERZONEJ na bieżącym korpusie; "
                "uruchom scripts/recompute_sigma2_threshold.py)",
                file=sys.stderr,
            )
        return float(UNSTABLE_SIGMA2_THRESHOLD_DEFAULT)
    if doc.get("judge_prompt_version") != JUDGE_PROMPT_VERSION:
        logger.warning("próg σ² odrzucony: proweniencja z innej wersji promptu")
        return None
    return float(doc["p95"])


UNSTABLE_SIGMA2_THRESHOLD = float(
    os.getenv("ATLAS_UNSTABLE_THRESHOLD", "")
    or (
        _v100_resolved
        if (_v100_resolved := resolve_unstable_sigma2_threshold()) is not None
        else str(UNSTABLE_SIGMA2_THRESHOLD_DEFAULT)
    )
)


def sigma2_percentile(values: List[float], q: float = 95.0) -> float:
    """Percentyl (interpolacja liniowa) rozkładu σ² — ta sama definicja co w teście F4."""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _sha_pair(prompt_version: str, query: str, cand_key: str, model: str = "") -> str:
    model_part = f"||{model}" if model else ""
    return hashlib.sha1(f"{prompt_version}||{query}||{cand_key}{model_part}".encode("utf-8")).hexdigest()


@dataclass
class JudgeVerdict:
    label: Optional[int]                  # 0, 1 lub None przy błędzie sędziego (T6)
    latent_score: float                   # y_latent in [0.0, 1.0]
    rubric: Dict[str, Any]                # słownik 5 wymiarów + hard_veto
    reasoning: str                        # syntetyczne podsumowanie
    thinking: Optional[str]               # thinking model z <think>...</think>
    source: str                           # np. 'llm_judge_live', 'llm_judge_cache', 'self_consistency_vote', 'judge_error'
    self_consistency: Optional[Dict[str, Any]] = None
    prior_used: bool = False



def compute_latent_score(
    rubric: Dict[str, Any],
    measured_dimensions: Optional[Sequence[str]] = None,
) -> Tuple[float, int, str]:
    """Oblicza ciągły wynik użyteczności y_latent in [0.0, 1.0] oraz etykietę binarną.

    Gdy podano `measured_dimensions` (np. dla ortogonalnej sondy P1/P2/P3 z SC_PROBE_SPECS),
    wynik `y_latent` jest liczony wyłącznie z wymiarów rozstrzygających danej sondy
    (znormalizowany przez sumę ich wag), dzięki czemu wariancja między sondami (sigma2)
    mierzy rzeczywistą rozbieżność między wymiarami P1/P2/P3 zamiast zwracać 0.0.
    """
    def _get_val(field: str) -> float:
        val = rubric.get(field)
        if isinstance(val, dict):
            return float(val.get("score", 0.0) or 0.0)
        try:
            return float(val or 0.0)
        except (ValueError, TypeError):
            return 0.0

    s_ent = _get_val("semantic_entailment")
    o_act = _get_val("operational_actionability")
    e_scope = _get_val("entity_scope_grounding")
    p_pol = _get_val("polarity_fidelity")
    t_fresh = _get_val("epistemic_freshness")

    veto_obj = rubric.get("hard_veto")
    veto_detected = False
    veto_reason = ""
    if isinstance(veto_obj, dict):
        veto_detected = bool(veto_obj.get("detected", False))
        veto_reason = str(veto_obj.get("reason", ""))
    elif isinstance(veto_obj, bool):
        veto_detected = veto_obj

    # Bramka weta: bezpośrednia sprzeczność logiczna LUB inwersja polaryzacji (p_pol == 0)
    if veto_detected:
        return 0.0, 0, f"hard_veto: {veto_reason or 'detected'}"
    if "polarity_fidelity" in rubric and p_pol == 0.0:
        return 0.0, 0, "polarity_inversion_veto (p_pol=0)"

    norm_by_dim = {
        "semantic_entailment": (0.30, min(1.0, max(0.0, s_ent / 3.0))),
        "operational_actionability": (0.25, min(1.0, max(0.0, o_act / 3.0))),
        "entity_scope_grounding": (0.20, min(1.0, max(0.0, e_scope / 2.0))),
        "polarity_fidelity": (0.15, min(1.0, max(0.0, p_pol / 2.0))),
        "epistemic_freshness": (0.10, min(1.0, max(0.0, t_fresh / 2.0))),
    }

    if measured_dimensions:
        active_dims = [d for d in measured_dimensions if d in norm_by_dim]
        total_w = sum(norm_by_dim[d][0] for d in active_dims)
        if total_w > 0.0:
            y_latent = sum(norm_by_dim[d][0] * norm_by_dim[d][1] for d in active_dims) / total_w
        else:
            y_latent = sum(w * v for w, v in norm_by_dim.values())
    else:
        y_latent = sum(w * v for w, v in norm_by_dim.values())

    y_latent = round(float(y_latent), 4)

    # Dyskretna etykieta binarna:
    # - dla pełnej rubryki: y_latent >= 0.50 oraz semantyka >= 1.8 oraz polaryzacja >= 1.0
    # - dla pojedynczej sondy częściowej (bez semantic_entailment w measured_dimensions): próg na y_latent >= 0.50
    # V109.5 (Strict AGI Entailment & Entity Scope Gate):
    # Hard negatives in the same domain routinely get s_ent=2 ("partially related"), e_scope=1 ("related entity, not exact"),
    # p_pol=2, t_fresh=2 -> y_latent=0.7167, which previously misclassified hard negatives as label=1 (synonym_positive)!
    # A candidate is label=1 ONLY if it genuinely entails the answer (s_ent >= 2.5, or s_ent >= 2.0 with exact entity scope e_scope >= 2.0 and y_latent >= 0.78).
    if measured_dimensions:
        # Partial orthogonal probe (P1, P2, P3): evaluate ONLY dimensions measured by this probe
        label = 1 if y_latent >= 0.75 else 0
        if "semantic_entailment" in measured_dimensions and s_ent < 2.0:
            label = 0
        if "entity_scope_grounding" in measured_dimensions and e_scope < 1.5:
            label = 0
        if "polarity_fidelity" in measured_dimensions and p_pol < 1.0:
            label = 0
    else:
        # Full composite rubric
        is_entailed = (s_ent >= 2.5 and e_scope >= 1.5) or (s_ent >= 2.0 and e_scope >= 2.0 and o_act >= 2.5)
        label = 1 if (y_latent >= 0.75 and is_entailed and (p_pol >= 1.0 or "polarity_fidelity" not in rubric)) else 0
    reason = f"y={y_latent:.3f} (ent={s_ent:.0f}/3, act={o_act:.0f}/3, scp={e_scope:.0f}/2, pol={p_pol:.0f}/2, frs={t_fresh:.0f}/2)"
    return y_latent, label, reason



def _extract_json_block(text: str) -> Optional[dict]:
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    candidate_strs: List[str] = []
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, flags=re.DOTALL)
    if match:
        candidate_strs.append(match.group(1))
    i = clean.find("{")
    j = clean.rfind("}")
    if i != -1:
        if j != -1 and j > i and clean[i : j + 1].count("{") == clean[i : j + 1].count("}"):
            candidate_strs.append(clean[i : j + 1])
        candidate_strs.append(clean[i:])
        if j != -1 and j > i:
            candidate_strs.append(clean[i : j + 1])

    for raw in candidate_strs:
        r = raw.strip()
        try:
            return json.loads(r)
        except Exception:
            pass
        cleaned = re.sub(r",\s*([\]}])", r"\1", r)
        cleaned = re.sub(r",\s*$", "", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        for suffix in ["}", "}}", "}}}", "\"", "\"}", "\"}}", "\"}}}"]:
            try:
                return json.loads(cleaned + suffix)
            except Exception:
                pass
    return None


def _extract_json_batch(text: str) -> List[Dict[str, Any]]:
    """Wyodrębnia listę obiektów JSON z odpowiedzi modelu dla batchingu K=3 (T1)."""
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 1. Próba parsowania całego tekstu jako listy lub słownika z listą
    try:
        data = json.loads(clean)
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return data
        if isinstance(data, dict):
            for k in ("evaluations", "variants", "items", "results"):
                if isinstance(data.get(k), list) and all(isinstance(item, dict) for item in data[k]):
                    return data[k]
    except Exception:
        pass

    # 2. Próba z bloków markdown ```json ... ```
    matches = re.findall(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", clean, flags=re.DOTALL)
    for m in matches:
        try:
            data = json.loads(m.strip())
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                return data
            if isinstance(data, dict):
                for k in ("evaluations", "variants", "items", "results"):
                    if isinstance(data.get(k), list) and all(isinstance(item, dict) for item in data[k]):
                        return data[k]
        except Exception:
            pass

    # 3. Zbalansowane wyszukiwanie kolejnych obiektów {...}
    objs = []
    pos = 0
    while pos < len(clean):
        start = clean.find("{", pos)
        if start == -1:
            break
        depth = 0
        in_string = False
        escape = False
        end = -1
        for i in range(start, len(clean)):
            ch = clean[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
        if end != -1:
            block = clean[start : end + 1]
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict) and any(
                    k in parsed for k in ("semantic_entailment", "operational_actionability", "score")
                ):
                    objs.append(parsed)
            except Exception:
                pass
            pos = end + 1
        else:
            break

    if objs:
        return objs

    single = _extract_json_block(text)
    return [single] if single else []


METROLOGICAL_BATCH_SYSTEM_PROMPT = METROLOGICAL_SYSTEM_PROMPT + """

UWAGA - TRYB BATCH K=3:
Otrzymasz 3 SONDY o ROZŁĄCZNYCH celach pomiarowych (entailment / zakres encji /
polaryzacja+świeżość) skierowane do tego samego kandydata. Każda sonda wskazuje,
które wymiary są w niej rozstrzygające — oceń każdą sondę osobno, zgodnie z jej celem.
Kolejność sond NIE niesie informacji — oceń każdą sondę niezależnie, nie przenoś ocen
pomiędzy wariantami (brak kotwiczenia pozycyjnego).
Dokonaj niezależnej oceny każdej sondy i zwróć tablicę JSON zawierającą dokładnie 3 obiekty ocen w formacie:
[
  {
    "variant_id": 1,
    "semantic_entailment": {"score": 0..3, "rationale": "krótkie uzasadnienie"},
    "operational_actionability": {"score": 0..3, "rationale": "krótkie uzasadnienie"},
    "entity_scope_grounding": {"score": 0..2, "rationale": "krótkie uzasadnienie"},
    "polarity_fidelity": {"score": 0..2, "rationale": "krótkie uzasadnienie"},
    "epistemic_freshness": {"score": 0..2, "rationale": "krótkie uzasadnienie"},
    "hard_veto": {"detected": false, "reason": "brak sprzeczności"}
  },
  {
    "variant_id": 2, ...
  },
  {
    "variant_id": 3, ...
  }
]
"""


# ==============================================================================
# V98 F8 (BUG-CAL36) — sondy self-consistency o ROZŁĄCZNYCH celach pomiarowych.
#
# DEFEKT (dowód pomiarowy): stary generator zwracał 3 "warianty" będące TYM SAMYM
# tekstem w 3 prefiksach szablonu:
#   'X' | 'Weryfikacja epistemiczna reguły pamięci: X' |
#   'Czy poniższy fakt odpowiada merytorycznie na zapytanie: X'
# W docs/calib/cache_v93_sc.jsonl 42/60 rekordów ma identyczne treści we wszystkich
# wariantach, więc σ² mierzył wrażliwość sędziego na SZABLON (efekt promptu), a nie
# niepewność werdyktu dla zapytania; skutkiem było sigma2 = 0.0.
#
# POPRAWKA: każda sonda ma INNY CEL POMIAROWY i inny, rozłączny zestaw wymiarów
# rozstrzygających. Kontrakt odpowiedzi JSON (5 wymiarów + hard_veto) oraz wagi
# latentu (compute_latent_score) pozostają BEZ ZMIAN — zmienia się tylko to, na czym
# sonda każe sędziemu się skupić, więc σ² mierzy niepewność pomiaru, nie szablonu.
# ==============================================================================
SC_PROBE_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "probe_id": "p1_entailment",
        "purpose": "materialny entailment: czy kandydat wprost odpowiada na zapytanie",
        "measured_dimensions": ("semantic_entailment", "operational_actionability"),
        "focus": (
            "SONDA P1 (entailment merytoryczny) — rozstrzygnij przede wszystkim, czy kandydat "
            "WPROST odpowiada na zapytanie i czy jego treść jest operacyjnie użyteczna. "
            "Wymiary rozstrzygające: semantic_entailment, operational_actionability."
        ),
    },
    {
        "probe_id": "p2_entity_scope",
        "purpose": "zakres i pochodzenie encji: czy kandydat dotyczy tej samej encji i warstwy",
        "measured_dimensions": ("entity_scope_grounding",),
        "focus": (
            "SONDA P2 (zakres i pochodzenie encji) — rozstrzygnij przede wszystkim, czy kandydat "
            "dotyczy TEJ SAMEJ encji, modułu, prefiksu i warstwy logicznej co zapytanie "
            "(caller vs callee, provider vs demon). Wymiar rozstrzygający: entity_scope_grounding."
        ),
    },
    {
        "probe_id": "p3_polarity_freshness",
        "purpose": "polaryzacja modalna i świeżość epistemiczna: zakaz nieodwrócony, wiedza aktualna",
        "measured_dimensions": ("polarity_fidelity", "epistemic_freshness"),
        "focus": (
            "SONDA P3 (polaryzacja i świeżość epistemiczna) — rozstrzygnij przede wszystkim, czy "
            "kandydat NIE odwraca nakazu w zakaz (ani odwrotnie) oraz czy opisuje stan aktualny, "
            "a nie wersję wycofaną. Wymiary rozstrzygające: polarity_fidelity, epistemic_freshness."
        ),
    },
)

# Wszystkie wymiary rubryki, które muszą być rozłącznie przypisane sondom (5 wymiarów).
SC_PROBE_RUBRIC_DIMENSIONS: Tuple[str, ...] = (
    "semantic_entailment",
    "operational_actionability",
    "entity_scope_grounding",
    "polarity_fidelity",
    "epistemic_freshness",
)


def self_consistency_probe(query: str, probe_index: int) -> Dict[str, Any]:
    """Jedna sonda self-consistency: cel pomiarowy + tekst podany sędziemu.

    Sondy różnią się CELEM (rozłączny zestaw wymiarów rozstrzygających), a nie prefiksem
    tego samego pytania (BUG-CAL36). Zwracany `text` trafia do `[ZAPYTANIE]` promptu sędziego.
    """
    spec = SC_PROBE_SPECS[probe_index % len(SC_PROBE_SPECS)]
    q = " ".join((query or "").split())
    return {
        "probe_id": spec["probe_id"],
        "purpose": spec["purpose"],
        "measured_dimensions": list(spec["measured_dimensions"]),
        "text": f"{spec['focus']}\nZAPYTANIE: {q}",
    }


def self_consistency_probes(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """K niezależnych sond SC (K domyślnie 3 = trzy rozłączne cele pomiarowe)."""
    return [self_consistency_probe(query, i) for i in range(max(0, k))]


def self_consistency_probe_dimensions(probes: Sequence[Dict[str, Any]]) -> List[Set[str]]:
    """Zestawy wymiarów rozstrzygających per sonda (do weryfikacji rozłączności)."""
    return [set(p.get("measured_dimensions", ())) for p in probes]


class AsyncRetrievalJudge:
    def __init__(
        self,
        endpoint: str = JUDGE_ENDPOINT,
        enabled: bool = True,
        concurrency: int = 1,
        sc_mode: Optional[str] = None,
        model: Optional[str] = None,
        unstable_threshold: Optional[float] = None,
        cache_path: Optional[Path] = None,
    ):
        self.endpoint = endpoint
        self.enabled = enabled
        self.sc_mode = (sc_mode or DEFAULT_SC_MODE).lower()
        self.model = model or os.getenv("ATLAS_JUDGE_MODEL", JUDGE_MODEL)
        # Kolejność progu (F9 V98): jawny argument -> env ATLAS_UNSTABLE_THRESHOLD ->
        # P95 z pliku proweniencji -> stała awaryjna (patrz resolve_unstable_sigma2_threshold).
        env_threshold = os.getenv("ATLAS_UNSTABLE_THRESHOLD")
        if unstable_threshold is not None:
            self.unstable_threshold = float(unstable_threshold)
        elif env_threshold:
            self.unstable_threshold = float(env_threshold)
        else:
            # V100 F1 (CAL37): None = proweniencja odrzucona (inna wersja promptu) —
            # głośny fallback na stałą awaryjną, NIE na cudzy pomiar.
            _resolved = resolve_unstable_sigma2_threshold()
            self.unstable_threshold = float(
                UNSTABLE_SIGMA2_THRESHOLD_DEFAULT if _resolved is None else _resolved
            )
        # V97 F7 (BUG-CAL34): sciezka cache PRZEZ resolver, nie bezposrednio z env/default.
        # Wczesniej `os.getenv("ATLAS_JUDGE_CACHE", DEFAULT_JUDGE_CACHE)` omijal patch
        # `JUDGE_CACHE_FILE` w testach -> pytest dopisywal wpisy do trackowanego
        # docs/calib/cache_v93_sc.jsonl (artefakt proweniencji +70 linii syntetycznych).
        self.cache_file = Path(cache_path) if cache_path else resolve_judge_cache_path()
        self.cache: Dict[str, dict] = {}
        self._load_cache()
        self.cache_fh = None
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(1)
        self.unstable_count = 0
        self.judge_errors_count = 0
        if self.enabled:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_fh = self.cache_file.open("a", encoding="utf-8")

    def _load_cache(self) -> None:
        if not self.cache_file.exists():
            return
        for ln in self.cache_file.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
                k = r.get("k")
                if not k:
                    continue
                if "label" in r and r.get("label") is not None:
                    self.cache[k] = r
                    continue
                raw_s = r.get("s") if r.get("s") is not None else r.get("score")
                if raw_s is not None:
                    self.cache[k] = r
            except Exception:
                pass

    def _blocking_http_call(self, body: bytes, timeout: Optional[int] = None) -> dict:
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        eff_timeout = timeout if timeout is not None else int(os.getenv("ATLAS_JUDGE_TIMEOUT", str(DEFAULT_JUDGE_TIMEOUT)))
        with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    async def judge_batch_self_consistency(
        self,
        query: str,
        cand_key: str,
        cand_text: str,
        k: int = 3,
        temperature: float = 0.7,
        sc_mode: Optional[str] = None,
        query_type: str = "prose",
    ) -> JudgeVerdict:
        """Self-consistency K=3 w trybie independent (domyślny) lub batch (T10/T11)."""
        if not self.enabled:
            return JudgeVerdict(0, 0.0, {}, "judge_disabled", None, "mechanic", prior_used=False)

        mode = (sc_mode or self.sc_mode or os.getenv("ATLAS_SC_MODE", DEFAULT_SC_MODE)).lower()
        model_name = self.model

        indep_cache_k = hashlib.sha1(f"{JUDGE_PROMPT_VERSION}_sc_indep_k{k}_t{temperature:.1f}||{query}||{cand_key}||{model_name}".encode("utf-8")).hexdigest()
        batch_cache_k = hashlib.sha1(f"{JUDGE_PROMPT_VERSION}_k{k}_t{temperature:.1f}||{query}||{cand_key}||{model_name}".encode("utf-8")).hexdigest()
        cache_k = indep_cache_k if mode == "independent" else batch_cache_k

        async with self._lock:
            entry = self.cache.get(cache_k)
            if entry and entry.get("label") is not None:
                lbl = entry.get("label")
                lat = float(entry.get("latent_score", 0.0))
                rub = entry.get("rubric", {})
                rsn = str(entry.get("r", "cache"))
                thk = entry.get("th")
                sc = entry.get("self_consistency")
                return JudgeVerdict(lbl, lat, rub, rsn, thk, "llm_judge_cache", self_consistency=sc, prior_used=False)

        # V98 F8 (BUG-CAL36): trzy sondaże o rozłącznych celach pomiarowych zamiast
        # trzech prefiksów szablonu tego samego zdania (patrz SC_PROBE_SPECS).
        probes = self_consistency_probes(query, k=k)
        variants = [p["text"] for p in probes]

        loop = asyncio.get_running_loop()

        if mode == "independent":
            # T10a: K niezależnych wywołań HTTP przy temperature > 0; cache per sample_idx
            labels = []
            latents = []
            rubrics = []
            thinkings = []

            for s in range(k):
                var_q = variants[s] if s < len(variants) else query.strip()
                sample_cache_k = hashlib.sha1(
                    f"{JUDGE_PROMPT_VERSION}_indep_sample{s}_k{k}_t{temperature:.1f}||{query}||{cand_key}||{model_name}".encode("utf-8")
                ).hexdigest()

                async with self._lock:
                    s_cached = self.cache.get(sample_cache_k)
                if s_cached and s_cached.get("label") is not None:
                    latents.append(float(s_cached.get("latent_score", 0.0)))
                    labels.append(int(s_cached.get("label", 0)))
                    rubrics.append(s_cached.get("rubric", {}))
                    thinkings.append(s_cached.get("th"))
                    continue

                user_content = METROLOGICAL_USER_PROMPT.format(q=var_q, c=cand_text.strip(), qt=query_type)
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": METROLOGICAL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": temperature,
                    "max_tokens": 4096,
                    "chat_template_kwargs": {"enable_thinking": True},
                }
                body = json.dumps(payload).encode("utf-8")

                async with self._semaphore:
                    for attempt in range(2):
                        try:
                            data = await loop.run_in_executor(None, self._blocking_http_call, body)
                            msg = data["choices"][0]["message"]
                            content = msg.get("content", "").strip()
                            thinking = msg.get("reasoning_content")
                            if not thinking and "<think>" in content:
                                m_think = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
                                if m_think:
                                    thinking = m_think.group(1).strip()

                            obj = _extract_json_block(content)
                            if not obj or not isinstance(obj, dict):
                                raise ValueError(f"Błąd parsowania JSON rubryki próbki {s}: {content[:150]}")

                            probe_dims = probes[s].get("measured_dimensions") if s < len(probes) else None
                            l_score, lbl, _ = compute_latent_score(obj, measured_dimensions=probe_dims)
                            latents.append(l_score)
                            labels.append(lbl)
                            rubrics.append(obj)
                            thinkings.append(thinking)

                            s_entry = {
                                "k": sample_cache_k,
                                "sample_idx": s,
                                "q": query,
                                "cand_key": cand_key,
                                "label": lbl,
                                "latent_score": l_score,
                                "rubric": obj,
                                "th": thinking,
                                "source": "llm_judge_sample",
                                "model": model_name,
                                "timestamp": time.time(),
                            }
                            async with self._lock:
                                self.cache[sample_cache_k] = s_entry
                                if self.cache_fh:
                                    self.cache_fh.write(json.dumps(s_entry, ensure_ascii=False) + "\n")
                                    self.cache_fh.flush()

                            break
                        except Exception as exc:
                            if attempt == 1:
                                self.judge_errors_count += 1
                                err_entry = {
                                    "timestamp": time.time(),
                                    "query": query,
                                    "cand_key": cand_key,
                                    "sample_idx": s,
                                    "error": str(exc),
                                }
                                try:
                                    JUDGE_ERRORS_FILE.parent.mkdir(parents=True, exist_ok=True)
                                    with JUDGE_ERRORS_FILE.open("a", encoding="utf-8") as efh:
                                        efh.write(json.dumps(err_entry, ensure_ascii=False) + "\n")
                                except Exception:
                                    pass
                                return JudgeVerdict(
                                    label=None,
                                    latent_score=0.0,
                                    rubric={},
                                    reasoning=f"judge_error: {exc}",
                                    thinking=None,
                                    source="judge_error",
                                    prior_used=False,
                                )
                            await asyncio.sleep(0.5)

            if len(latents) < k:
                raise ValueError(f"Niekompletne próbki z sędziego: otrzymano {len(latents)}/{k} próbek (F-A)")

            composite_rubric: Dict[str, Any] = dict(rubrics[0]) if rubrics else {}
            for obj_s, probe_s in zip(rubrics[:k], probes[:k], strict=True):
                for dim_k in probe_s.get("measured_dimensions") or ():
                    if dim_k in obj_s:
                        composite_rubric[dim_k] = obj_s[dim_k]
                hv_s = obj_s.get("hard_veto", False)
                is_veto_s = bool(hv_s.get("detected", False)) if isinstance(hv_s, dict) else bool(hv_s)
                if is_veto_s:
                    composite_rubric["hard_veto"] = True
                    if isinstance(hv_s, dict) and hv_s.get("reason"):
                        composite_rubric["veto_reason"] = hv_s["reason"]
                    elif obj_s.get("veto_reason"):
                        composite_rubric["veto_reason"] = obj_s["veto_reason"]

            composite_latent, composite_lbl, _ = compute_latent_score(composite_rubric)
            majority_label = 1 if (composite_lbl == 1 and sum(labels) >= (len(labels) / 2.0)) else 0
            mean_latent = composite_latent
            sigma2 = round(float(np.var(latents, ddof=1) if len(latents) > 1 else 0.0), 4)
            unstable = bool(sigma2 > self.unstable_threshold)
            if unstable:
                self.unstable_count += 1

            sc_info = {
                "mode": "independent",
                "k": len(labels),
                "variants": variants,
                # V98 F8: proweniencja CELU każdej sondy (wymiary rozstrzygające per sonda).
                "probes": [
                    {k2: v2 for k2, v2 in p.items() if k2 != "text"} for p in probes[: len(labels)]
                ],
                "labels": labels,
                "unanimous": len(set(labels)) == 1,
                "latents": latents,
                "sigma2": sigma2,
                "unstable": unstable,
            }
            summary_reason = f"k={len(labels)} vote={majority_label} mean_latent={mean_latent:.3f} sigma2={sigma2:.4f} unstable={unstable} mode=independent"
            best_rubric = composite_rubric
            thk_summary = thinkings[0] if thinkings else None

            agg_entry = {
                "k": cache_k,
                "q": query,
                "cand_key": cand_key,
                "label": majority_label,
                "latent_score": mean_latent,
                "rubric": best_rubric,
                "r": summary_reason,
                "th": thk_summary,
                "source": "self_consistency_vote",
                "self_consistency": sc_info,
                "model": model_name,
                "timestamp": time.time(),
            }
            async with self._lock:
                self.cache[cache_k] = agg_entry
                if self.cache_fh:
                    self.cache_fh.write(json.dumps(agg_entry, ensure_ascii=False) + "\n")
                    self.cache_fh.flush()

            return JudgeVerdict(
                label=majority_label,
                latent_score=mean_latent,
                rubric=best_rubric,
                reasoning=summary_reason,
                thinking=thk_summary,
                source="self_consistency_vote",
                self_consistency=sc_info,
                prior_used=False,
            )

        else:
            # Batch mode (1 HTTP call evaluating K variants)
            # V98 F8 (BUG-CAL36): sondy mają rozłączne cele — batch listuje je jako
            # sondy P1/P2/P3 z celem, nie jako parafrazy tego samego zdania.
            user_content = "[KANDYDAT (ZAPAMIĘTANY FAKT)]:\n" + cand_text.strip() + "\n\n[SONDY ZAPYTANIA DO OCENY]:\n"
            for idx, (var, probe) in enumerate(zip(variants, probes, strict=True), 1):
                user_content += f"Sonda {idx} [{probe['probe_id']}, cel: {probe['purpose']}]: {var}\n"
            user_content += "\nOceń każdą z powyższych sond i zwróć tablicę JSON z 3 obiektami ocen."

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": METROLOGICAL_BATCH_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "temperature": temperature,
                "max_tokens": 4096,
                "chat_template_kwargs": {"enable_thinking": True},
            }
            body = json.dumps(payload).encode("utf-8")

            async with self._semaphore:
                for attempt in range(2):
                    try:
                        data = await loop.run_in_executor(None, self._blocking_http_call, body)
                        msg = data["choices"][0]["message"]
                        content = msg.get("content", "").strip()
                        thinking = msg.get("reasoning_content")
                        if not thinking and "<think>" in content:
                            m_think = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
                            if m_think:
                                thinking = m_think.group(1).strip()

                        eval_objs = _extract_json_batch(content)
                        if not eval_objs:
                            raise ValueError(f"Błąd parsowania JSON batcha: {content[:150]}")
                        if len(eval_objs) < k:
                            raise ValueError(f"Niekompletny batch z sędziego: otrzymano {len(eval_objs)}/{k} obiektów (F-A)")

                        labels = []
                        latents = []
                        rubrics = []
                        for idx_s, obj in enumerate(eval_objs[:k]):
                            probe_dims = probes[idx_s].get("measured_dimensions") if idx_s < len(probes) else None
                            l_score, lbl, _ = compute_latent_score(obj, measured_dimensions=probe_dims)
                            latents.append(l_score)
                            labels.append(lbl)
                            rubrics.append(obj)

                        if not labels:
                            raise ValueError("Brak poprawnych ocen w batchu")

                        composite_rubric = dict(rubrics[0]) if rubrics else {}
                        for obj_s, probe_s in zip(rubrics[:k], probes[:k], strict=True):
                            for dim_k in probe_s.get("measured_dimensions") or ():
                                if dim_k in obj_s:
                                    composite_rubric[dim_k] = obj_s[dim_k]
                            hv_s = obj_s.get("hard_veto", False)
                            is_veto_s = bool(hv_s.get("detected", False)) if isinstance(hv_s, dict) else bool(hv_s)
                            if is_veto_s:
                                composite_rubric["hard_veto"] = True
                                if isinstance(hv_s, dict) and hv_s.get("reason"):
                                    composite_rubric["veto_reason"] = hv_s["reason"]
                                elif obj_s.get("veto_reason"):
                                    composite_rubric["veto_reason"] = obj_s["veto_reason"]

                        composite_latent, composite_lbl, _ = compute_latent_score(composite_rubric)
                        majority_label = 1 if (composite_lbl == 1 and sum(labels) >= (len(labels) / 2.0)) else 0
                        mean_latent = composite_latent
                        sigma2 = round(float(np.var(latents, ddof=1) if len(latents) > 1 else 0.0), 4)
                        unstable = bool(sigma2 > self.unstable_threshold)

                        if unstable:
                            self.unstable_count += 1

                        sc_info = {
                            "mode": "batch",
                            "k": len(labels),
                            "variants": variants,
                            # V98 F8: proweniencja CELU każdej sondy (wymiary rozstrzygające per sonda).
                            "probes": [
                                {k2: v2 for k2, v2 in p.items() if k2 != "text"}
                                for p in probes[: len(labels)]
                            ],
                            "labels": labels,
                            "unanimous": len(set(labels)) == 1,
                            "latents": latents,
                            "sigma2": sigma2,
                            "unstable": unstable,
                        }

                        summary_reason = f"k={len(labels)} vote={majority_label} mean_latent={mean_latent:.3f} sigma2={sigma2:.4f} unstable={unstable} mode=batch"
                        best_rubric = composite_rubric

                        cache_entry = {
                            "k": cache_k,
                            "q": query,
                            "cand_key": cand_key,
                            "label": majority_label,
                            "latent_score": mean_latent,
                            "rubric": best_rubric,
                            "r": summary_reason,
                            "th": thinking,
                            "model": model_name,
                            "source": "self_consistency_vote",
                            "self_consistency": sc_info,
                            "timestamp": time.time(),
                        }

                        async with self._lock:
                            self.cache[cache_k] = cache_entry
                            if self.cache_fh:
                                self.cache_fh.write(json.dumps(cache_entry, ensure_ascii=False) + "\n")
                                self.cache_fh.flush()

                        return JudgeVerdict(
                            label=majority_label,
                            latent_score=mean_latent,
                            rubric=best_rubric,
                            reasoning=summary_reason,
                            thinking=thinking,
                            source="self_consistency_vote",
                            self_consistency=sc_info,
                            prior_used=False,
                        )
                    except Exception as exc:
                        if attempt == 1:
                            self.judge_errors_count += 1
                            err_entry = {
                                "timestamp": time.time(),
                                "query": query,
                                "cand_key": cand_key,
                                "error": str(exc),
                            }
                            try:
                                JUDGE_ERRORS_FILE.parent.mkdir(parents=True, exist_ok=True)
                                with JUDGE_ERRORS_FILE.open("a", encoding="utf-8") as efh:
                                    efh.write(json.dumps(err_entry, ensure_ascii=False) + "\n")
                            except Exception:
                                pass
                            return JudgeVerdict(
                                label=None,
                                latent_score=0.0,
                                rubric={},
                                reasoning=f"judge_error: {exc}",
                                thinking=None,
                                source="judge_error",
                                prior_used=False,
                            )
                        await asyncio.sleep(0.5)

            return JudgeVerdict(label=None, latent_score=0.0, rubric={}, reasoning="judge_error: unknown", thinking=None, source="judge_error", prior_used=False)

    async def judge(
        self,
        query: str,
        cand_key: str,
        cand_text: str,
        query_type: str = "prose",
    ) -> JudgeVerdict:
        """Zwraca JudgeVerdict (label, latent_score, rubric, reasoning, thinking, source)."""
        if not self.enabled:
            return JudgeVerdict(0, 0.0, {}, "judge_disabled", None, "mechanic", prior_used=False)

        k = _sha_pair(JUDGE_PROMPT_VERSION, query, cand_key, self.model)
        legacy_k = _sha_pair(JUDGE_PROMPT_VERSION, query, cand_key) if self.model == "local" else None
        async with self._lock:
            entry = self.cache.get(k) or (self.cache.get(legacy_k) if legacy_k else None)
            if entry:
                if "label" in entry and entry.get("label") is not None:
                    try:
                        lbl = int(entry["label"])
                        lat = float(entry.get("latent_score", lbl))
                        rub = entry.get("rubric", {})
                        rsn = str(entry.get("r", entry.get("reason", entry.get("reasoning", "cache"))))
                        thk = entry.get("th", entry.get("thinking"))
                        sc = entry.get("self_consistency")
                        return JudgeVerdict(lbl, lat, rub, rsn, thk, "llm_judge_cache", self_consistency=sc, prior_used=False)
                    except (ValueError, TypeError):
                        pass
                raw_score = entry.get("s") if entry.get("s") is not None else entry.get("score")
                if raw_score is not None:
                    try:
                        lbl = int(raw_score)
                        lat = float(lbl)
                        rsn = str(entry.get("r", entry.get("reason", entry.get("reasoning", "cache"))))
                        thk = entry.get("th", entry.get("thinking", entry.get("reasoning_content")))
                        return JudgeVerdict(lbl, lat, {}, rsn, thk, "llm_judge_cache", prior_used=False)
                    except (ValueError, TypeError):
                        pass

        user_content = METROLOGICAL_USER_PROMPT.format(q=query.strip(), c=cand_text.strip(), qt=query_type)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": METROLOGICAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        body = json.dumps(payload).encode("utf-8")

        loop = asyncio.get_running_loop()
        async with self._semaphore:
            for attempt in range(2):
                try:
                    data = await loop.run_in_executor(None, self._blocking_http_call, body)
                    msg = data["choices"][0]["message"]
                    content = msg.get("content", "").strip()
                    thinking = msg.get("reasoning_content")
                    if not thinking and "<think>" in content:
                        m_think = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
                        if m_think:
                            thinking = m_think.group(1).strip()

                    obj = _extract_json_block(content)
                    if not obj or not isinstance(obj, dict):
                        raise ValueError(f"Błąd parsowania JSON rubryki: {content[:150]}")

                    latent_score, label, summary_reason = compute_latent_score(obj)

                    cache_entry = {
                        "k": k,
                        "q": query,
                        "cand_key": cand_key,
                        "label": label,
                        "latent_score": latent_score,
                        "rubric": obj,
                        "r": summary_reason,
                        "th": thinking,
                        "source": "llm_judge_live",
                        "model": self.model,
                        "timestamp": time.time(),
                    }
                    async with self._lock:
                        self.cache[k] = cache_entry
                        if self.cache_fh:
                            self.cache_fh.write(json.dumps(cache_entry, ensure_ascii=False) + "\n")
                            self.cache_fh.flush()

                    return JudgeVerdict(label, latent_score, obj, summary_reason, thinking, "llm_judge_live", prior_used=False)
                except Exception as exc:
                    if attempt == 1:
                        # T6: Awaria sędziego = judge_error, rejestracja błędu
                        self.judge_errors_count += 1
                        err_entry = {
                            "timestamp": time.time(),
                            "query": query,
                            "cand_key": cand_key,
                            "error": str(exc),
                        }
                        try:
                            JUDGE_ERRORS_FILE.parent.mkdir(parents=True, exist_ok=True)
                            with JUDGE_ERRORS_FILE.open("a", encoding="utf-8") as efh:
                                efh.write(json.dumps(err_entry, ensure_ascii=False) + "\n")
                        except Exception:
                            pass
                        return JudgeVerdict(
                            label=None,
                            latent_score=0.0,
                            rubric={},
                            reasoning=f"judge_error: {exc}",
                            thinking=None,
                            source="judge_error",
                            prior_used=False,
                        )
                    await asyncio.sleep(0.5)

        return JudgeVerdict(label=None, latent_score=0.0, rubric={}, reasoning="judge_error: unknown", thinking=None, source="judge_error", prior_used=False)

    def close(self) -> None:
        if self.cache_fh:
            try:
                self.cache_fh.close()
            except Exception:
                pass



# Kompatybilność wsteczna:
RetrievalJudge = AsyncRetrievalJudge

DEFAULT_PROD_QDRANT = Path.home() / ".hermes" / "atlas" / "qdrant"

# F5 (V97): domyślna KOPIA korpusu Qdranta mieszka w trwałym katalogu cache (jak reszta),
# nie na tmpfs. Override: --qdrant-path (CLI, patrz resolve_qdrant_path) — bez zmian.
FALLBACK_QDRANT_PATH = str(CALIB_CACHE_DIR / "qs_prod_after")
# Domyślna kopia Kùzu dla kalibracji (też trwały katalog; nadpisaće ją --kuzu-path).
KUZU_CALIB_COPY = CALIB_CACHE_DIR / "kuzu_calib"


def resolve_qdrant_path(preferred_path: Optional[str] = None, allow_prod_store: bool = False) -> str:
    """Dynamiczny resolver ścieżki magazynu Qdrant (T7/D16/D20).

    Wymaga jawnego --qdrant-path poza magazynem produkcyjnym ~/.hermes/atlas/qdrant.
    Cichy powrót do produkcyjnego magazynu jest zabroniony.
    """
    prod_path = DEFAULT_PROD_QDRANT.resolve()
    if preferred_path:
        p = Path(preferred_path).resolve()
        if p == prod_path and not allow_prod_store:
            raise ValueError(
                "BŁĄD (T7): Podana ścieżka wskazuje na produkcyjny magazyn ~/.hermes/atlas/qdrant.\n"
                "Użycie produkcyjnego magazynu grozi kolizją locków z daemonem ATLAS.\n"
                "Podaj kopię magazynu (--qdrant-path /sciezka/do/kopii) lub użyj jawnej flagi --allow-prod-store."
            )
        return str(p)
    if not allow_prod_store:
        raise ValueError(
            "BŁĄD (T7): Nie podano --qdrant-path. Cichy powrót do magazynu produkcyjnego ~/.hermes/atlas/qdrant jest zabroniony.\n"
            "Podaj ścieżkę do kopii magazynu (--qdrant-path /sciezka) lub użyj jawnej flagi --allow-prod-store."
        )
    return str(prod_path)


_resolve_qdrant_path = resolve_qdrant_path
DEFAULT_QDRANT_PATH = str(DEFAULT_PROD_QDRANT)

# V109.7: Ujednolicenie pool_limit=50 z produkcyjnym prefetch.py:27 default_limit=50
DEFAULT_POOL_LIMIT = int(os.getenv("ATLAS_CALIB_POOL_LIMIT", "50"))
# Bramka min_vec: STAŁE bezpieceńskie z candidates.py:132-150 (ścieżka bez kalibracji).
# Uczenie progu na puli przefiltrowanej WŁASNYM, dopiero co wyuczonym progiem byłoby
# pętlą (dataset zawężałby sobie własną dystrybucję), więc celowo NIE czytamy tu
# artefaktu kalibracji.
MIN_VEC_SYMBOLIC = 0.70
MIN_VEC_PROSE = 0.35
# F5 (V97): duplikat przypisania HYBRID_CACHE (dawniej "/tmp/atlas_calib_hybrid_pool_v92.json",
# tmpfs!) USUNIĘTY — jedyna definicja jest w bloku stałych na górze pliku (CALIB_CACHE_DIR).


def extract_family(key: str) -> str:
    """Extracts semantic key family for per-family calibration."""
    if key.startswith("hermes:"):
        parts = key.split(":")
        return ":".join(parts[:2])
    return key.split(":")[0]


def _clean_key(raw: str) -> str:
    k = str(raw or "")
    return k[3:] if k.startswith("kv:") else k


def _hit_to_row(hit: Dict[str, Any]) -> Dict[str, Any]:
    """Wyciąga z trafienia Qdranta pola potrzebne do odtworzenia kandydata (paraphiszing)."""
    rec = hit.get("record") or {}
    meta = rec.get("metadata") or hit.get("metadata") or {}
    return {
        "score": float(hit.get("score", 0.0) or 0.0),
        "id": str(hit.get("id", "")),
        "kv_key": hit.get("kv_key") or meta.get("kv_key") or "",
        "upsert_key": hit.get("upsert_key") or "",
        "subject": str(rec.get("subject") or hit.get("subject") or ""),
        "object": str(rec.get("object") or hit.get("text") or ""),
        "text": str(hit.get("text") or rec.get("object") or ""),
        "source_type": str(meta.get("source_type") or rec.get("source_type") or "user_explicit"),
        "origin": str(meta.get("origin") or ""),
        "via_sparse": bool(hit.get("via_sparse")),
        "via_dense": bool(hit.get("via_dense")),
    }


DEFAULT_PROD_KUZU = Path.home() / ".hermes" / "atlas" / "kuzu"


def resolve_kuzu_path(preferred_path: Optional[str] = None) -> Optional[str]:
    """Zwraca ścieżkę do kopii bazy Kùzu dla procesu kalibracji bez kolizji locków."""
    if preferred_path:
        p = Path(preferred_path).resolve()
        return str(p) if p.exists() else None
    calib_copy = KUZU_CALIB_COPY
    _migrate_legacy_cache(calib_copy, "/tmp/kuzu_calib")  # F5: jednorazowo z tmpfs, potem trwale
    if calib_copy.exists():
        try:
            if calib_copy.is_file() and calib_copy.stat().st_size > 100_000:
                return str(calib_copy)
            elif calib_copy.is_dir() and any(calib_copy.iterdir()):
                return str(calib_copy)
        except Exception:
            pass
    if DEFAULT_PROD_KUZU.exists():
        try:
            import shutil
            if DEFAULT_PROD_KUZU.is_dir():
                shutil.copytree(DEFAULT_PROD_KUZU, calib_copy, dirs_exist_ok=True)
            else:
                shutil.copy2(DEFAULT_PROD_KUZU, calib_copy)
            return str(calib_copy)
        except Exception:
            return None
    return None


class ReadOnlyKV:
    """Zero-risk, read-only adapter dla atlas.db zduck-typowany pod candidates.gather_candidates.

    Otwiera bazę SQLite w trybie file:{path}?mode=ro (uri=True) bez dotykania produkcyjnego
    VerifiedKVStore i bez wykonywania jakichkolwiek operacji zapisu (WAL checkpoint, triggery, tabele).
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        resolved = Path(db_path).resolve()
        uri = f"file:{resolved}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def search_candidates_sync(
        self,
        query: str,
        limit: int = 100,
        words: Optional[List[str]] = None,
        stems: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        cursor = self._conn.cursor()
        q_clean = query.strip()
        # 1. Exact match (P0 dla zapytań symbolicznych -> importance_score = 1.0 w candidates.py:117)
        row = cursor.execute(
            "SELECT key, value, confidence, timestamp, metadata FROM state_variables WHERE key = ?",
            (q_clean,),
        ).fetchone()
        if row:
            meta = {}
            if row["metadata"] and str(row["metadata"]).strip().startswith("{"):
                try:
                    meta = json.loads(row["metadata"])
                except Exception:
                    pass
            results[row["key"]] = {
                "key": row["key"],
                "value": _decode_value(row["value"]),
                "confidence": float(row["confidence"] or 1.0),
                "timestamp": float(row["timestamp"] or 0.0),
                "metadata": meta,
            }

        # 2. Token / word prefix lookup na state_variables
        tokens_to_match = list(dict.fromkeys((words or []) + (stems or [])))
        if tokens_to_match and len(results) < limit:
            for tok in tokens_to_match[:5]:
                if len(tok) < 3:
                    continue
                match_rows = cursor.execute(
                    "SELECT key, value, confidence, timestamp, metadata FROM state_variables WHERE key LIKE ? LIMIT ?",
                    (f"%{tok}%", limit - len(results)),
                ).fetchall()
                for r in match_rows:
                    k = r["key"]
                    if k not in results:
                        meta = {}
                        if r["metadata"] and str(r["metadata"]).strip().startswith("{"):
                            try:
                                meta = json.loads(r["metadata"])
                            except Exception:
                                pass
                        results[k] = {
                            "key": k,
                            "value": _decode_value(r["value"]),
                            "confidence": float(r["confidence"] or 1.0),
                            "timestamp": float(r["timestamp"] or 0.0),
                            "metadata": meta,
                        }
                    if len(results) >= limit:
                        break
        return results

    def filter_existing_keys_sync(self, keys: Sequence[str]) -> Set[str]:
        if not keys:
            return set()
        cursor = self._conn.cursor()
        placeholders = ",".join("?" for _ in keys)
        rows = cursor.execute(
            f"SELECT key FROM state_variables WHERE key IN ({placeholders})",
            list(keys),
        ).fetchall()
        return {r[0] for r in rows}

    async def filter_existing_keys(self, keys: Sequence[str]) -> Set[str]:
        return self.filter_existing_keys_sync(keys)

    def get_sync(self, key: str) -> Optional[Dict[str, Any]]:
        cursor = self._conn.cursor()
        row = cursor.execute(
            "SELECT key, value, confidence, timestamp, metadata FROM state_variables WHERE key = ?",
            (key,),
        ).fetchone()
        if not row:
            return None
        meta = {}
        if row["metadata"] and str(row["metadata"]).strip().startswith("{"):
            try:
                meta = json.loads(row["metadata"])
            except Exception:
                pass
        return {
            "key": row["key"],
            "value": _decode_value(row["value"]),
            "confidence": float(row["confidence"] or 1.0),
            "timestamp": float(row["timestamp"] or 0.0),
            "metadata": meta,
        }

    def get_all_sync(self, limit: int = 100, paged: bool = True) -> Dict[str, Any]:
        cursor = self._conn.cursor()
        rows = cursor.execute(
            "SELECT key, value, confidence, timestamp, metadata FROM state_variables LIMIT ?",
            (limit,),
        ).fetchall()
        res = {}
        for r in rows:
            meta = {}
            if r["metadata"] and str(r["metadata"]).strip().startswith("{"):
                try:
                    meta = json.loads(r["metadata"])
                except Exception:
                    pass
            res[r["key"]] = {
                "key": r["key"],
                "value": _decode_value(r["value"]),
                "confidence": float(r["confidence"] or 1.0),
                "timestamp": float(r["timestamp"] or 0.0),
                "metadata": meta,
            }
        return res

    async def close(self) -> None:
        self.close_sync()

    def close_sync(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


class CalibrationGraphStore(KuzuGraphStore):
    """Lekki adapter KuzuGraphStore dla procesu kalibracji z O(1) case-normalization."""

    def __init__(self, db_path: str):
        super().__init__(db_path)
        self._lower_to_node: Dict[str, str] = {}

    def hydrate_from_disk_sync(self) -> Dict[str, Any]:
        res = super().hydrate_from_disk_sync()
        with self._graph_lock:
            self._lower_to_node = {str(n).lower(): str(n) for n in self.nx_graph.nodes()}
        return res

    async def get_node_relations(self, subject: str, predicate: Optional[str] = None) -> List[Dict[str, Any]]:
        rels = await super().get_node_relations(subject, predicate)
        if rels:
            return rels
        s_low = subject.lower()
        matched_node = (
            self._lower_to_node.get(s_low)
            or self._lower_to_node.get(s_low.split(":")[-1])
            or self._lower_to_node.get(s_low.split("/")[-1])
        )
        if matched_node and matched_node != subject:
            return await super().get_node_relations(matched_node, predicate)
        return []


class CachedHybridVectorStore:
    """Obejście cache dla search_hybrid_async na QdrantVectorStore z zachowaniem 5 źródeł."""

    def __init__(
        self,
        store: Optional[QdrantVectorStore],
        cache: Dict[str, Any],
        cache_path: Optional[Path] = None,
    ):
        self._store = store
        self._cache = cache
        self._cache_path = cache_path
        self.collection_name = (
            getattr(store, "collection_name", "hermes_episodic_memory_68m")
            if store
            else "hermes_episodic_memory_68m"
        )

    async def search_hybrid_async(
        self,
        query: str,
        limit: int = 15,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        ck = hashlib.sha1(f"{self.collection_name}|{limit}|{query}".encode("utf-8")).hexdigest()
        raw_hits = None
        if ck in self._cache:
            raw_hits = self._cache[ck]
        else:
            ck50 = hashlib.sha1(f"{self.collection_name}|50|{query}".encode("utf-8")).hexdigest()
            if ck50 in self._cache:
                raw_hits = self._cache[ck50]

        if raw_hits is not None:
            res = []
            for h in raw_hits[:limit]:
                h_copy = dict(h)
                meta = dict(h_copy.get("metadata") or {})
                if not meta.get("origin") and h_copy.get("origin"):
                    meta["origin"] = h_copy["origin"]
                if not meta.get("kv_key") and h_copy.get("kv_key"):
                    meta["kv_key"] = h_copy["kv_key"]
                if not meta.get("source_type") and h_copy.get("source_type"):
                    meta["source_type"] = h_copy["source_type"]
                h_copy["metadata"] = meta
                res.append(h_copy)
            return res

        if self._store is None:
            return []
        hits = await self._store.search_hybrid_async(query=query, limit=limit, session_id=session_id)
        rows = [_hit_to_row(h) for h in hits or []]
        self._cache[ck] = rows
        if self._cache_path:
            try:
                self._cache_path.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
        return rows


async def _hybrid_pool(
    store: Optional[QdrantVectorStore],
    query: str,
    limit: int,
    cache: Dict[str, Any],
    cache_path: Path,
) -> List[Dict[str, Any]]:
    """Realna pula hybrydowa z Qdranta z cache na dysku (encje raz, wynik wielokrotnie).

    Cachujemy SUROWE trafienia (przed bramką min_vec), żeby zmiana stałych bramki nie
    unieważniała cache'u i żeby próg dało się przeliczyć bez odpytywania silnika.
    """
    assert store is not None, "hybrid pool wymaga QdrantVectorStore"
    ck = hashlib.sha1(f"{store.collection_name}|{limit}|{query}".encode("utf-8")).hexdigest()
    if ck in cache:
        return cache[ck]
    hits = await store.search_hybrid_async(query=query, limit=limit)
    rows = [_hit_to_row(h) for h in hits or []]
    cache[ck] = rows
    return rows


def _build_candidates(rows: List[Dict[str, Any]], query: str, is_sym: bool) -> List[Dict[str, Any]]:
    """Odtwarza kroki 3-4 candidates.gather_candidates na SUROWYCH trafieniach hybrydy.

    Identyczne jak demon: ekstrakcja klucza (metadata.kv_key → kv_key → upsert_key →
    record.subject → id, stripping prefiksu `kv:`), bramka min_vec z rafinem
    daemon_set/lex-hint, filtr `_is_filtered`, dedup po kluczu z zachowaniem
    najwyższego `vector_score` (candidates.py:266).
    """
    q_clean = query.strip()
    words = [w.lower().strip(".,!?:;\"'()[]{}") for w in q_clean.split() if len(w) >= 3]
    stems = [w[:6] for w in words if len(w) >= 4]
    bilingual_stems = [_bilingual_stem(w) for w in words if len(w) >= 3]
    min_vec = MIN_VEC_SYMBOLIC if is_sym else MIN_VEC_PROSE

    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        k = _clean_key(
            str(row.get("kv_key") or row.get("upsert_key") or row.get("subject") or row.get("id") or "")
        )
        v = row.get("object") or row.get("text") or ""
        score = float(row.get("score", 0.0))
        cand_text = f"{k} {v}".lower()
        has_lex_hint = any(w in cand_text for w in words if len(w) >= 3) or any(
            s in cand_text for s in stems if len(s) >= 4
        )
        if not has_lex_hint and bilingual_stems:
            cand_tokens = [w for w in re.findall(r"\w+", cand_text) if len(w) >= 3]
            cand_stems = [_bilingual_stem(tok) for tok in cand_tokens]
            has_lex_hint = bool(set(cand_stems) & set(bilingual_stems))

        is_sparse_hit = bool(row.get("via_sparse"))
        is_daemon_set = row.get("origin", "") == "daemon_set"
        eff_min = max(0.75, min_vec) if (is_daemon_set and not has_lex_hint) else min_vec
        if not is_sparse_hit and score < eff_min:
            continue
        meta = {"kv_key": row.get("kv_key"), "origin": row.get("origin")}
        if any(k.lower().startswith(p) for p in PROBE_PREFIXES) or _is_filtered(k, v, meta, None) or not v:
            continue

        cand = {
            "key": k,
            "subject": k,
            "object": v,
            # V92.1 KLUCZOWE: vector_score = realny wynik fuzji DBSF z Qdranta,
            # dokładnie tak jak robi to candidates.py:204/266-271. Nie cosine z macierzy.
            "vector_score": score,
            # candidates.py buduje rekord wektorowy jako _build_record(k, v, "vector",
            # {"metadata": meta}, score) → confidence=1.0, source_type z metadanych.
            "confidence": 1.0,
            "source_type": row.get("source_type") or "user_explicit",
        }
        if k not in best or score > float(best[k]["vector_score"]):
            best[k] = cand
    return list(best.values())


def _ranked_scores_by_key(
    query: str, rows: List[Dict[str, Any]], is_sym: bool
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Realna pula → score_candidates(dbsf) → (ranked, najlepszy wynik per klucz)."""
    cands = _build_candidates(rows, query, is_sym)
    if not cands:
        return [], {}
    ranked = score_candidates(query, cands, fusion_strategy="dbsf")
    by_key: Dict[str, float] = {}
    for c in ranked:
        k = str(c.get("key") or c.get("subject") or "")
        by_key[k] = max(by_key.get(k, float("-inf")), float(c["score"]))
    return ranked, by_key


def _cand_for_fact(
    ranked: List[Dict[str, Any]], by_key: Dict[str, float], fact: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Odnajduje obiekt kandydata dla faktu w puli (po kluczu, odcisku treści lub treści).

    V96 (BUG-CAL21 M3):
    1. Dopasowanie po dokładnym kluczu (str(c.key) == fact.key) -> (c, False)
    2. Dopasowanie po klasie równoważności odcisku treści:
       compute_content_fingerprint(cand_object) == compute_content_fingerprint(fact.text)
       -> (c, True) [cel zdeduplikowany jako bliźniak, trafienie klasy równoważności]
    3. Dopasowanie po pełnej treści tekstowej -> (c, True)
    """
    k = fact["key"]
    for c in ranked:
        if str(c.get("key") or c.get("subject") or "") == k:
            return c, False

    tgt_fp = compute_content_fingerprint(fact.get("text", ""))
    if tgt_fp:
        for c in ranked:
            cand_text = str(c.get("object") or c.get("text") or c.get("content") or "")
            if compute_content_fingerprint(cand_text) == tgt_fp:
                return c, True

    txt = fact.get("text", "")
    for c in ranked:
        cand_text = str(c.get("object") or c.get("text") or c.get("content") or "")
        if cand_text == txt:
            return c, True

    return None, False


def _score_for_fact(
    ranked: List[Dict[str, Any]], by_key: Dict[str, float], fact: Dict[str, Any]
) -> float:
    """Wynik DBSF faktu w REALNEJ puli: po kluczu, a gdy dedup po odcisku treści zwinął
    bliźniaka pod innym kluczem — po klasie odcisku treści (M3)."""
    c, _ = _cand_for_fact(ranked, by_key, fact)
    return float(c["score"]) if c else 0.0


def active_sampling_prior(
    cand_score: float,
    cand_margin: float,
    cand_key: str,
    tgt_key: str,
    q_type: str,
    vector_score: float = 0.0,
    store_keys: Optional[set] = None,
) -> Optional[JudgeVerdict]:
    """Active Uncertainty Sampling & Structural Negative Prior (V93.1 - T2).

    ZASADA BEZWZGLĘDNA (T2):
    - Prior NIGDY nie etykietuje pozytywów po score/margin (wycięcie D12/D21).
    - Auto-etykiety dozwolone WYŁĄCZNIE ze sygnałów strukturalnych dla negatywów
      (brak kandydata w puli / cand_key nie istnieje w magazynie).
    """
    if q_type == "symbolic":
        if cand_key == tgt_key:
            return JudgeVerdict(1, 1.0, {}, "Strukturalna zgodność klucza symbolicznego", None, "symbolic_exact_match", prior_used=False)
        else:
            return JudgeVerdict(0, 0.0, {}, "Strukturalna niezgodność klucza symbolicznego", None, "symbolic_structural_negative", prior_used=False)

    # Strukturalny negatyw: cand_key nie istnieje w magazynie wiedzy (T2)
    if store_keys is not None and cand_key and cand_key not in store_keys:
        return JudgeVerdict(0, 0.0, {}, "Strukturalny brak kandydata w magazynie", None, "structural_absent_negative", prior_used=True)

    # Wszystkie inne przypadki prose (w tym cel pozytywny i rywale z puli)
    # MUSZĄ przejść przez sędziego LLM!
    return None


def symbolic_rank_probe(pool: List[Dict[str, Any]], target_key: str) -> Dict[str, Any]:
    """Diagnostyka retrievalowa dla kluczy symbolicznych (T5).

    Oblicza hit-rate top-1 oraz top-3 dla dokładnego klucza w puli hybrydowej.
    """
    if not pool:
        return {"hit_top1": False, "hit_top3": False, "rank": -1, "pool_size": 0}
    clean_tgt = _clean_key(target_key)
    ranks = []
    for idx, cand in enumerate(pool):
        ck = _clean_key(str(cand.get("key") or cand.get("subject") or cand.get("kv_key") or ""))
        if ck == clean_tgt:
            ranks.append(idx)
            break
    rank = ranks[0] if ranks else -1
    return {
        "hit_top1": rank == 0,
        "hit_top3": 0 <= rank < 3,
        "rank": rank,
        "pool_size": len(pool),
    }



FACT_IDX_BY_KEY: Dict[str, int] = {}
FACT_IDX_BY_TEXT: Dict[str, int] = {}


def _is_twin(cand_key: str, cand_text: str, tgt_i: int, fact_vecs_norm: np.ndarray,
             eq: List[set], cand_vecs: Dict[str, np.ndarray], facts: Optional[List[Dict[str, Any]]] = None) -> bool:
    """Czy kandydat jest bliźniakiem celu (cos > EQUIV_THRESHOLD lub ten sam fingerprint) — wtedy NIE może być
    twardym negatywem (C3)."""
    if facts is not None and 0 <= tgt_i < len(facts):
        tgt_txt = facts[tgt_i].get("text", "")
        tgt_fp = compute_content_fingerprint(tgt_txt)
        if tgt_fp and compute_content_fingerprint(cand_text) == tgt_fp:
            return True
        if cand_text and cand_text == tgt_txt:
            return True

    j = FACT_IDX_BY_KEY.get(cand_key, FACT_IDX_BY_TEXT.get(cand_text, -1))
    if j >= 0:
        return j in eq[tgt_i]
    v = cand_vecs.get(cand_text)
    if v is None:
        return False
    return bool(float(np.dot(v, fact_vecs_norm[tgt_i])) > EQUIV_THRESHOLD)


PROVENANCE_ALLOWED_SOURCES = {
    "llm_judge_live",
    "llm_judge_cache",
    "self_consistency_vote",
    "gold_human",
}


def verify_provenance_gate(
    records: List[Dict[str, Any]],
    min_verified_ratio: float = 1.0,
    min_negatives: int = 30,
    gold_set_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Bramka proweniencji V93.1 (T8)."""
    test_recs = [r for r in records if r.get("split") == "test"]
    if not test_recs:
        raise ValueError("Brak rekordów testowych w zbiorze!")

    prose_test = [r for r in test_recs if r.get("query_type") == "prose"]
    verified_prose = [
        r for r in prose_test
        if r.get("label_source") in PROVENANCE_ALLOWED_SOURCES and not r.get("prior_used", False)
    ]
    verified_ratio = len(verified_prose) / max(1, len(prose_test))

    if verified_ratio < min_verified_ratio:
        raise ValueError(
            f"BRAMKA PROWENIENCJI ODRZUCONA: Tylko {verified_ratio * 100:.1f}% rekordów testowych prose "
            f"posiada wiarygodną proweniencję (wymagane: {min_verified_ratio * 100:.1f}%). "
            f"Wykryto rekordy ze źródeł nielicencjonowanych."
        )

    negatives_count = sum(1 for r in test_recs if r.get("label") == 0)
    if negatives_count < min_negatives:
        raise ValueError(
            f"BRAMKA NEGATYWÓW ODRZUCONA: Liczba negatywów w teście {negatives_count} < {min_negatives}."
        )

    if gold_set_path and Path(gold_set_path).exists():
        with open(gold_set_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                gr = json.loads(line)
                if gr.get("label_source") != "gold_human":
                    raise ValueError(
                        f"ZŁOTY ZBIÓR P3 SKAŻONY: Rekord posiada label_source='{gr.get('label_source')}', "
                        f"a wymagane jest wyłącznie 'gold_human'!"
                    )

    return {
        "status": "passed",
        "verified_ratio": verified_ratio,
        "test_records": len(test_recs),
        "negatives_count": negatives_count,
    }


def validate_test_record_provenance(rec: Dict[str, Any], use_judge: bool = True) -> None:
    """T2 & F-D: Weryfikacja że test split nie przyjmuje prioru ani nieautoryzowanego źródła."""
    if rec.get("split") != "test":
        return
    src = rec.get("label_source")
    # T3: Metadata records for policy-filtered targets are exempt from provenance checks
    if src == "target_filtered_by_policy":
        return
    q_t = rec.get("query_type")
    if q_t == "prose":
        allowed_sources = PROVENANCE_ALLOWED_SOURCES | ({"no_judge_fallback"} if not use_judge else set())
        if src not in allowed_sources or (rec.get("prior_used", False) and use_judge):
            raise ValueError(
                f"Wyciek prioru do testu! Rekord testowy prose ma label_source='{src}', prior_used={rec.get('prior_used')}"
            )
    elif q_t == "symbolic":
        if src not in {"symbolic_exact_match", "symbolic_structural_negative"}:
            raise ValueError(f"Nieautoryzowane źródło w teście symbolic: {src}")


def get_candidate_channel(c: Dict[str, Any]) -> str:
    """Klasyfikuje kandydata do jednego z 5 kanałów: kv, vector_dense, vector_sparse, graph, hot_buffer."""
    src = str(c.get("source") or c.get("origin") or "")
    if src == "vector":
        meta = c.get("metadata") or {}
        if meta.get("via_sparse") or c.get("via_sparse"):
            return "vector_sparse"
        return "vector_dense"
    return src


MIN_TARGET_IN_POOL_RATIO = 0.70
# V97 F1: minimalna liczba zbudowanych par (prose+symbolic), po której bramka puli
# odpala się PRZED pierwszym wywołaniem sędziego na normalnym biegu.
PRE_JUDGE_GATE_MIN_PAIRS = 10
HARNESS_CHANNELS: Tuple[str, ...] = ("kv", "vector_dense", "vector_sparse", "graph", "hot_buffer")


def harness_pool_presort(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """T4 (BUG-CAL23): deterministyczny pre-sort puli harnessu przed dedupem odcisku treści.

    Produkcyjny `score_candidates` (score.py:247) przy równej authority zachowuje PIERWSZY
    wpis w kolejności wejścia, więc dedup bliźniaków jest w nim zależny od kolejności źródeł.
    Harness stabilizuje to sortowaniem (authority DESC, subject ASC) — bez dotykania `src/`.
    """
    return sorted(
        cands,
        key=lambda c: (
            get_record_authority_score(str(c.get("subject", ""))),
            str(c.get("subject", "")),
        ),
        reverse=True,
    )


def build_harness_ranked_pool(q_text: str, cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Kanoniczna ścieżka rankingu puli harnessu: pre-sort (T4) + produkcyjny DBSF."""
    return score_candidates(q_text, harness_pool_presort(cands), fusion_strategy="dbsf")


NATURAL_QUESTIONS_CACHE_PATH = DOCS_CALIB_DIR / "natural_questions_cache.json"


def _load_natural_questions_cache() -> Dict[str, str]:
    if NATURAL_QUESTIONS_CACHE_PATH.exists():
        try:
            data = json.loads(NATURAL_QUESTIONS_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items() if v}
        except Exception:
            pass
    return {}


def _prose_variant_text(text: str, key: str = "") -> str:
    """V109.3 AGI Q&A Calibration: Synthesizes a natural human question instead of copying the literal 1st sentence.

    Eliminates self-querying data leakage (where copying first_sentence produced sparse=1.00, dense=0.95,
    score_top1=0.98 and an artificially high threshold t*=0.49 that rejected real user queries scoring ~0.45).
    Extracts only the core topic/entity anchor (3-5 salient terms, stripping paths, numbers, and answer details)
    and frames it as a natural Polish interrogative matching real human chat queries.
    """
    cache = _load_natural_questions_cache()
    if key and key in cache:
        return cache[key]

    # Clean key subject tokens (strip generic prefixes like hermes:memory_md:, fact:, rule:)
    clean_key = re.sub(r"^(hermes|fact|rule|preference|constraint|agent|user|system|config|memory_md|user_md|stated_memory|working_fact|episodic_event)[:_]+", " ", str(key), flags=re.IGNORECASE)
    clean_key = re.sub(r"[:_\-/]+", " ", clean_key)
    # Remove parenthetical details, file paths, hashes, and dates from the question stem
    base_clause = first_sentence(str(text))
    base_clause = re.sub(r"\([^)]*\)", " ", base_clause)
    base_clause = re.sub(r"~?/[\w./\-]+", " ", base_clause)
    base_clause = re.sub(r"\b(202[0-9][\-_][01][0-9][\-_][0-3][0-9]|[0-9a-f]{7,40})\b", " ", base_clause)
    # Take only the subject/topic part before colon, semicolon, equals, or arrow
    topic_part = re.split(r"[:;=→—]|\s+-\s+", base_clause)[0].strip()
    if len(topic_part.split()) < 2:
        topic_part = " ".join(base_clause.split()[:6])

    combined_words = []
    seen_low = set()
    stop_tokens = {
        "hermes", "memory", "fact", "stated", "working", "episodic", "event",
        "user", "jest", "oraz", "tylko", "przez", "jako", "jego", "tego",
        "który", "która", "które", "przy", "albo", "żeby", "musi", "może",
        "użytkownik", "korzysta", "system", "state", "rule", "preference",
        "constraint", "task", "pierwszy", "drugi", "trzeci"
    }
    # Take up to 2 tokens from clean_key and up to 2 distinctive tokens from base_clause to guarantee uniqueness (ANOMALIA-01)
    key_words = []
    for w in re.findall(r"[^\W_]+", clean_key):
        wl = w.lower()
        if len(w) < 3 or wl in seen_low or wl in stop_tokens or wl.isdigit():
            continue
        seen_low.add(wl)
        key_words.append(w)
        if len(key_words) >= 2:
            break

    body_words = []
    for w in re.findall(r"[^\W_]+", f"{topic_part} {base_clause}"):
        wl = w.lower()
        if len(w) < 4 or wl in seen_low or wl in stop_tokens or wl.isdigit():
            continue
        seen_low.add(wl)
        body_words.append(w)
        if len(body_words) >= 2:
            break

    # Build a natural 3-word topic phrase (2 from key + 1 disambiguator from body) with 3-word human interrogative
    # so query lexical overlap is ~3/6 (sparse ~0.38-0.50, dense ~0.38-0.48, score_top1 ~0.44-0.52)
    body_words = []
    for w in re.findall(r"[^\W_]+", f"{topic_part} {base_clause}"):
        wl = w.lower()
        if len(w) < 4 or wl in seen_low or wl in stop_tokens or wl.isdigit():
            continue
        seen_low.add(wl)
        body_words.append(w)
        if len(body_words) >= (1 if len(key_words) >= 2 else 2):
            break

    combined_words = (key_words + body_words)[:3]
    topic_phrase = " ".join(combined_words) if combined_words else " ".join(re.sub(r"[:_\-]", " ", first_sentence(text)).split()[:3])
    k_low = str(key).lower()
    if k_low.startswith(("rule:", "constraint:")) or "zasad" in topic_phrase.lower() or "reguł" in topic_phrase.lower():
        return f"Jakie reguły w projekcie określają {topic_phrase}?"
    if k_low.startswith("preference:") or "prefer" in topic_phrase.lower():
        return f"Jakie podejście na co dzień wybiera {topic_phrase}?"
    if "konfiguracj" in topic_phrase.lower() or "config" in k_low or "port" in topic_phrase.lower():
        return f"Gdzie w środowisku działa konfiguracja {topic_phrase}?"
    return f"Co mówią nasze dotychczasowe ustalenia odnośnie {topic_phrase}?"


def _query_variants(fact: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Dwie warianty zapytania per fakt: prose (naturalne pytanie człowieka) i symbolic (klucz)."""
    return [
        ("prose", normalize_stored_text(_prose_variant_text(fact["text"], str(fact.get("key", ""))))),
        ("symbolic", fact["key"]),
    ]


def _target_filtered_by_policy(tgt_key: str, text: str) -> bool:
    """M4: read-path policy filter (CAL17 & CAL15) — rekordy konwersacyjne i filtry."""
    return (
        any(tgt_key.lower().startswith(p) for p in PROBE_PREFIXES)
        or _is_filtered(tgt_key, text, None, None)
        or is_conversational_churn(tgt_key, text)
    )


def _account_pool_hit(
    q_type: str,
    ranked: List[Dict[str, Any]],
    by_key: Dict[str, float],
    fact: Dict[str, Any],
    in_pool: Dict[str, List[int]],
    target_deduped_away_count: Dict[str, int],
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Liczy target-in-pool / M3 dedup dla jednej pary (query, fakt) — wspólne dla biegu i dry-runu."""
    tgt_cand, was_deduped = _cand_for_fact(ranked, by_key, fact)
    if was_deduped:
        target_deduped_away_count[q_type] += 1
    pos_score = float(tgt_cand["score"]) if tgt_cand else _score_for_fact(ranked, by_key, fact)
    in_pool[q_type][1] += 1
    if pos_score > 0.0:
        in_pool[q_type][0] += 1
    return tgt_cand, was_deduped


def validate_pool_gates(
    coverage: Dict[str, int],
    edges_total: int,
    queries_with_edges: int,
    counters: Dict[str, Any],
) -> List[str]:
    """Bramki puli jako lista naruszeń (pusta lista = wszystkie bramki zdane).

    Wydzielone z asercji stojących PO pętli sędziego, żeby ten sam zestaw reguł
    dało się odpalić PRZED pierwszym wywołaniem sędziego LLM:
      * `--dry-run-pool` — pełna pula wszystkich zapytań bez sędziego (~2 s CPU),
      * normalny bieg — bramka przed wejściem w pętlę sędziego.
    `counters`: in_pool {q_type: [hit, tot]}, source, hot_buffer_enabled, graph_gate_applies.
    """
    violations: List[str] = []
    in_pool = counters.get("in_pool") or {}
    for qt in ("prose", "symbolic"):
        hit, tot = in_pool.get(qt, (0, 0))
        if not tot:
            violations.append(f"Wiring failure: brak par zapytań typu {qt} w puli (tot=0)")
            continue
        ratio = hit / tot
        if ratio < MIN_TARGET_IN_POOL_RATIO:
            violations.append(
                f"Wiring failure: target w realnej puli tylko {hit}/{tot} dla {qt} "
                f"({ratio * 100:.1f}% < {MIN_TARGET_IN_POOL_RATIO * 100:.0f}%) "
                f"— sprawdź ATLAS_QDRANT_PATH/kolekcję, nie ucz na takiej puli"
            )
    if counters.get("source") == "hybrid":
        channel_coverage_required = bool(counters.get("channel_coverage_required", True))
        if channel_coverage_required:
            for ch in HARNESS_CHANNELS:
                if ch == "hot_buffer" and not counters.get("hot_buffer_enabled", True):
                    continue
                # T1 (BUG-CAL22): kanał `graph` MA BYĆ bramkowany po krawędziach PRE-filter,
                # nie po liczbie kandydatów POST-filter (0-3/6613 — niedeterministyczne).
                if ch == "graph":
                    continue
                if int(coverage.get(ch, 0)) <= 0:
                    violations.append(f"Source Coverage Gate FAIL: 0 kandydatów z {ch}")
        if counters.get("graph_gate_applies", False) and int(edges_total) <= 0:
            violations.append(
                f"Source Coverage Gate FAIL: 0 krawędzi grafu PRE-filter "
                f"(queries_with_edges={queries_with_edges})"
            )
        # F6 (V97 BUG-CAL25): kanał hot_buffer nie może dokładać IDENTYCZNEGO zestawu
        # do każdej puli — wtedy jest szumem stałym, nie sygnałem per-zapytanie.
        if counters.get("hot_buffer_enabled", False):
            hb_keys = counters.get("hot_buffer_unique_keys")
            hb_pools = counters.get("hot_buffer_pools")
            if hb_keys is not None and hb_pools and int(hb_keys) <= int(hb_pools):
                violations.append(
                    f"Hot Buffer Gate FAIL: {hb_keys} unikalnych kluczy hot_buffer na "
                    f"{hb_pools} puli kandydatów (BUG-CAL25: stały syntetyczny bufor w każdej puli)"
                )
    return violations


HOT_BUFFER_KEY_PREFIX = "harness:calib_active_working_memory"


def hot_buffer_entry_key(q_text: str, idx: int) -> str:
    """Klucz wpisu bufora roboczego WŁAŚCIWY dla danego zapytania (F6 / BUG-CAL25).

    Stara wersja: `harness:calib_active_working_memory_{1,2,3}` — te same 3 klucze w KAŻDEJ
    puli, więc kanał hot_buffer dokładał identyczny zestaw do 100% zapytań (szum stały).
    Digest tekstu zapytania w kluczu daje bufor per-kontekst (tak jak w prodzie, gdzie bufor
    roboczy zmienia się z sesją i turą).
    """
    digest = hashlib.sha1(q_text.strip().lower().encode("utf-8")).hexdigest()[:10]
    return f"{HOT_BUFFER_KEY_PREFIX}_{digest}_{idx}"


def build_query_hot_buffer(
    base_buffer: Any,
    q_text: str,
    session_id: Optional[str] = None,
) -> Optional[Any]:
    """Buduje bufor roboczy DLA JEDNEGO zapytania (F6 / BUG-CAL25) albo None.

    Wpisy dziedziczą treść/confidence z bufora bazowego, ale klucze i wartość kontekstu są
    zależne od zapytania. Świadomie NIE wkładamy tu faktów z korpusu: bufor nie jest oceniany
    podobieństwem (score=0.0, wchodzi przez `get_active`), więc doklejenie klucza docelowego
    do jego własnej puli byłoby wyciekiem etykiety (target zawsze "w puli"). Różnica vs prod:
    produkcyjny bufor trzyma realne, świeże fakty sesji — harness trzyma syntetyczny
    scratchpad, ale już NIE identyczny dla wszystkich zapytań.
    """
    if base_buffer is None:
        return None
    digest = hashlib.sha1(q_text.strip().lower().encode("utf-8")).hexdigest()[:10]
    now = time.time()
    sess = "" if session_id is None else str(session_id)
    per_query = HotBufferPolicy(
        maxlen=int(getattr(base_buffer, "maxlen", 25) or 25),
        ttl_seconds=float(getattr(base_buffer, "ttl_seconds", 900.0) or 900.0),
    )
    for idx, entry in enumerate(list(base_buffer)):
        value = str(getattr(entry, "value", "") or "")
        per_query.update(
            key=hot_buffer_entry_key(q_text, idx),
            value=f"{value} [kontekst {digest}]",
            confidence=float(getattr(entry, "confidence", 1.0) or 1.0),
            metadata={"session_id": sess, "source_type": "working_fact", "harness_query_digest": digest},
            source_type="working_fact",
            timestamp=now,
        )
    return per_query


def hot_buffer_key_set(hb: Any) -> Set[str]:
    """Zbiór kluczy wpisów bufora (do licznika unikalności, F6)."""
    if hb is None:
        return set()
    return {str(getattr(e, "key", "")) for e in list(hb) if getattr(e, "key", "")}


async def build_query_pool(
    *,
    q_text: str,
    source: str,
    pool_limit: int,
    enc: Any,
    cached_store: Any,
    kv_store: Any,
    graph_store: Any,
    hot_buffer: Any,
    eval_session_id: Optional[str],
    facts: List[Dict[str, Any]],
    doc_matrix: np.ndarray,
    source_coverage: collections.Counter,
    graph_counters: Dict[str, int],
    hot_buffer_keys: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], List[str]]:
    """Buduje pulę kandydatów dla jednego zapytania (5-kanałowy harness albo macierz).

    Zwraca (ranked, by_key, current_sources_in_query). Aktualizuje `source_coverage`
    oraz `graph_counters` (PRE-filter liveness grafu, T1/BUG-CAL22).
    ``hot_buffer_keys`` (opcjonalny zbiór) zbiera klucze wpisów bufora per-zapytanie (F6).
    """
    q_vec = np.array(
        enc.encode_query(q_text) if hasattr(enc, "encode_query") else enc.encode(q_text),
        dtype=np.float32,
    )
    q_vec = q_vec / (float(np.linalg.norm(q_vec)) + 1e-9)

    if source == "hybrid":
        # F6 (BUG-CAL25): bufor roboczy WŁAŚCIWY temu zapytaniu, nie jeden globalny.
        hb_query = build_query_hot_buffer(hot_buffer, q_text, session_id=eval_session_id)
        if hot_buffer_keys is not None and hb_query is not None:
            hot_buffer_keys.update(hot_buffer_key_set(hb_query))
        # V96 (BUG-CAL21 M1 & M2): gather_candidates z pełnym 5-źródłowym harnessem:
        # 1. kv_store (ReadOnlyKV)
        # 2. vector_store (CachedHybridVectorStore: dense + sparse)
        # 3. graph_store (KuzuGraphStore)
        # 4. hot_buffer (HotBufferPolicy)
        # 5. session_id (eval_session_id)
        cands = await gather_candidates(
            query=q_text,
            session_id=eval_session_id,
            limit=max(pool_limit, 50),
            vector_store=cached_store,
            kv_store=kv_store,
            graph_store=graph_store,
            hot_buffer=hb_query,
            enable_calibrated_thresholds=False,
        )
        # T4 (BUG-CAL23): deterministyczny tie-break przeniesiony z prod score.py do harnessu.
        ranked = build_harness_ranked_pool(q_text, cands)
        by_key: Dict[str, float] = {}
        for c in ranked:
            k = str(c.get("key") or c.get("subject") or "")
            by_key[k] = max(by_key.get(k, float("-inf")), float(c.get("score", 0.0)))

        sources_in_query = set()
        for c in cands:
            ch = get_candidate_channel(c)
            sources_in_query.add(ch)
            source_coverage[ch] += 1
        current_sources_in_query = sorted(list(sources_in_query))

        # T1 (BUG-CAL22): Graph liveness PRE-filter — count raw edges
        # before session/policy filtering to verify channel is alive.
        # POST-filter gives 0-3/6613 (non-deterministic), PRE-filter gives 2122 edges.
        if graph_store is not None:
            words = q_text.split()[:3]
            query_edges = 0
            for w in words:
                try:
                    rels = graph_store.get_node_relations(w)
                    # BUG-CAL28 (V97 F1): CalibrationGraphStore.get_node_relations jest async —
                    # bez await `len(coroutine)` rzucał TypeError, wyjątek był łykany przez
                    # `except Exception: pass`, więc licznik PRE-filter był STRUKTURALNIE 0
                    # i bramka grafu nie mogła nigdy przejść (V96 umierał na niej po sędzim).
                    if inspect.iscoroutine(rels):
                        rels = await rels
                    query_edges += len(rels)
                except Exception:
                    pass
            graph_counters["edges_pre_filter"] += query_edges
            if query_edges > 0:
                graph_counters["queries_with_edges"] += 1
    else:
        current_sources_in_query = ["matrix"]
        # STARA pulą: wszystkie fakty, vector_score = cosine z macierzy.
        scores = doc_matrix @ q_vec
        cands = [
            {
                "key": facts[j]["key"],
                "subject": facts[j]["key"],
                "object": facts[j]["text"],
                "vector_score": float(scores[j]),
                "confidence": float(facts[j].get("confidence", 1.0) or 1.0),
                "source_type": "user_explicit",
            }
            for j in range(len(facts))
        ]
        ranked = score_candidates(q_text, cands, fusion_strategy="dbsf")
        by_key = {}
        for c in ranked:
            k = str(c.get("key") or c.get("subject") or "")
            by_key[k] = max(by_key.get(k, float("-inf")), float(c["score"]))

    return ranked, by_key, current_sources_in_query


async def scan_pool_gates(
    *,
    facts: List[Dict[str, Any]],
    source: str,
    pool_limit: int,
    enc: Any,
    cached_store: Any,
    kv_store: Any,
    graph_store: Any,
    hot_buffer: Any,
    eval_session_id: Optional[str],
    doc_matrix: np.ndarray,
    max_pairs: Optional[int] = None,
    channel_coverage_required: bool = True,
    graph_gate_applies: Optional[bool] = None,
    mode: str = "dry_run_pool",
) -> Dict[str, Any]:
    """Buduje pule dla zapytań (opcjonalnie do `max_pairs` par) i ZWRACA raport bramek.

    Zero wywołań sędziego, zero zapisu zbioru — jedyne wyjście to zwrócony dict.
    Wspólny rdzeń dla `--dry-run-pool` (pełny przekrój) i dla bramki PRZED pętlą
    sędziego na normalnym biegu (`max_pairs=PRE_JUDGE_GATE_MIN_PAIRS`).
    """
    in_pool = {"prose": [0, 0], "symbolic": [0, 0]}
    target_filtered_by_policy_count = {"prose": 0, "symbolic": 0}
    target_deduped_away_count = {"prose": 0, "symbolic": 0}
    source_coverage: collections.Counter = collections.Counter()
    graph_counters = {"edges_pre_filter": 0, "queries_with_edges": 0}
    # F6 (BUG-CAL25): unikalne klucze hot_buffer naprzeciw liczbie zbudowanych pul.
    hot_buffer_keys: Set[str] = set()
    hot_buffer_pools = 0
    pool_sizes: List[int] = []
    pairs_built = 0

    for fact in facts:
        tgt_key = fact["key"]
        for q_type, q_text in _query_variants(fact):
            if _target_filtered_by_policy(tgt_key, fact.get("text", "")):
                target_filtered_by_policy_count[q_type] += 1
                continue
            ranked, by_key, _sources = await build_query_pool(
                hot_buffer_keys=hot_buffer_keys,
                q_text=q_text,
                source=source,
                pool_limit=max(pool_limit, 50),
                enc=enc,
                cached_store=cached_store,
                kv_store=kv_store,
                graph_store=graph_store,
                hot_buffer=hot_buffer,
                eval_session_id=eval_session_id,
                facts=facts,
                doc_matrix=doc_matrix,
                source_coverage=source_coverage,
                graph_counters=graph_counters,
            )
            pool_sizes.append(len(ranked))
            _account_pool_hit(q_type, ranked, by_key, fact, in_pool, target_deduped_away_count)
            pairs_built += 1
            if hot_buffer is not None:
                hot_buffer_pools += 1
            if max_pairs is not None and pairs_built >= max_pairs:
                break
        if max_pairs is not None and pairs_built >= max_pairs:
            break

    if graph_gate_applies is None:
        graph_gate_applies = graph_store is not None

    violations = validate_pool_gates(
        source_coverage,
        graph_counters["edges_pre_filter"],
        graph_counters["queries_with_edges"],
        {
            "in_pool": in_pool,
            "source": source,
            "hot_buffer_enabled": hot_buffer is not None,
            "hot_buffer_unique_keys": len(hot_buffer_keys),
            "hot_buffer_pools": hot_buffer_pools,
            "graph_gate_applies": graph_gate_applies,
            "channel_coverage_required": channel_coverage_required,
        },
    )

    return {
        "mode": mode,
        "judge_called": False,
        "source": source,
        "pool_limit": pool_limit,
        "facts": len(facts),
        "pairs_built": pairs_built,
        "gate_violations": violations,
        "gates_passed": not violations,
        "pool_size": {
            "median": int(np.median(pool_sizes)) if pool_sizes else 0,
            "min": min(pool_sizes) if pool_sizes else 0,
            "max": max(pool_sizes) if pool_sizes else 0,
        },
        "source_coverage": dict(source_coverage),
        "graph_edges_pre_filter": graph_counters["edges_pre_filter"],
        "queries_with_graph_edges": graph_counters["queries_with_edges"],
        "target_in_pool": {qt: {"hit": in_pool[qt][0], "total": in_pool[qt][1]} for qt in ("prose", "symbolic")},
        "target_filtered_by_policy_M4": dict(target_filtered_by_policy_count),
        "target_deduped_away_M3": dict(target_deduped_away_count),
        "hot_buffer_unique_keys": len(hot_buffer_keys),
        "hot_buffer_pools": hot_buffer_pools,
    }


def print_pool_report(report: Dict[str, Any], title: str, report_path: Optional[Path] = None) -> None:
    """Surowy wydruk raportu bramek puli (wspólny dla dry-runu i bramki przed sędzią)."""
    cov = report["source_coverage"]
    tip = report["target_in_pool"]
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)
    print(f"Facts={report['facts']}, pary z pulą={report['pairs_built']}, "
          f"pool median={report['pool_size']['median']} (min={report['pool_size']['min']}, "
          f"max={report['pool_size']['max']})")
    for ch in HARNESS_CHANNELS:
        print(f"  - [{ch:<15}]: {cov.get(ch, 0)} kandydatów w pulach (POST-filter)")
    print(f"  - [{'graph_pre':<15}]: {report['graph_edges_pre_filter']} krawędzi PRE-filter, "
          f"{report['queries_with_graph_edges']} zapytań z krawędziami")
    for qt in ("prose", "symbolic"):
        print(f"  - [{qt:<8}] target w puli: {tip[qt]['hit']}/{tip[qt]['total']} "
              f"| M4 filtered: {report['target_filtered_by_policy_M4'][qt]} "
              f"| M3 deduped: {report['target_deduped_away_M3'][qt]}")
    if report_path is not None:
        print(f"Raport: {report_path}")


async def dry_run_pool_report(
    *,
    facts: List[Dict[str, Any]],
    source: str,
    pool_limit: int,
    enc: Any,
    cached_store: Any,
    kv_store: Any,
    graph_store: Any,
    hot_buffer: Any,
    eval_session_id: Optional[str],
    doc_matrix: np.ndarray,
    output_path: Path,
) -> Dict[str, Any]:
    """`--dry-run-pool`: pule WSZYSTKICH zapytań + bramki BEZ sędziego, raport JSON obok `--output`.

    Kod wyjścia: 0 gdy bramki zdane (raport w `<output>.pool.json`), 2 gdy naruszone.
    """
    report = await scan_pool_gates(
        facts=facts, source=source, pool_limit=max(pool_limit, 50), enc=enc, cached_store=cached_store,
        kv_store=kv_store, graph_store=graph_store, hot_buffer=hot_buffer,
        eval_session_id=eval_session_id, doc_matrix=doc_matrix, mode="dry_run_pool",
    )
    report_path = Path(str(output_path) + ".pool.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_pool_report(report, " 🧪 DRY-RUN POOL (bramki PRZED sędzią, zero wywołań LLM)", report_path)
    if report["gate_violations"]:
        print(f"[FAIL] Naruszenia bram puli ({len(report['gate_violations'])}):")
        for v in report["gate_violations"]:
            print(f"  - {v}")
        raise SystemExit(2)
    print("[OK] Wszystkie bramki puli zdane — sędzia nie był wywoływany.")
    return report


async def pre_judge_pool_gate(
    *,
    facts: List[Dict[str, Any]],
    source: str,
    pool_limit: int,
    enc: Any,
    cached_store: Any,
    kv_store: Any,
    graph_store: Any,
    hot_buffer: Any,
    eval_session_id: Optional[str],
    doc_matrix: np.ndarray,
    graph_gate_applies: bool,
) -> Dict[str, Any]:
    """Bramka puli PRZED pierwszym wywołaniem sędziego na normalnym biegu (V97 F1).

    Robi próbkę `PRE_JUDGE_GATE_MIN_PAIRS` par tym samym `scan_pool_gates` /
    `validate_pool_gates`, którego używa `--dry-run-pool`. Na próbce NIE żądamy
    obecności każdego kanału (patrz ADR-016) — to wymaga pełnego przekroju korpusa;
    żądamy wiringu target-in-pool i liveness grafu PRE-filter, czyli tego, co
    realnie każe płacić sędzią za martwą pulę. Naruszenie = SystemExit(2) bez
    ani jednego wywołania LLM.
    """
    report = await scan_pool_gates(
        facts=facts, source=source, pool_limit=max(pool_limit, 50), enc=enc, cached_store=cached_store,
        kv_store=kv_store, graph_store=graph_store, hot_buffer=hot_buffer,
        eval_session_id=eval_session_id, doc_matrix=doc_matrix,
        max_pairs=PRE_JUDGE_GATE_MIN_PAIRS, channel_coverage_required=False,
        graph_gate_applies=graph_gate_applies, mode="pre_judge_gate",
    )
    print_pool_report(report, f" ⚔️ BRAMKA PULI PRZED SĘDZIĄ (próbka {PRE_JUDGE_GATE_MIN_PAIRS} par)")
    if report["gate_violations"]:
        print("[GATE PRZED SĘDZIĄ] FAIL — przerywam PRZED pierwszym wywołaniem sędziego:")
        for v in report["gate_violations"]:
            print(f"  - {v}")
        raise SystemExit(2)
    print("[GATE PRZED SĘDZIĄ] OK — sędzia nie był jeszcze wywoływany.")
    return report


async def build_dataset(
    db_path: Path = DEFAULT_DB,
    output_path: Path = OUTPUT_JSONL,
    seed: int = 42,
    source: str = "hybrid",
    pool_limit: int = DEFAULT_POOL_LIMIT,
    qdrant_path: Optional[str] = None,
    allow_prod_store: bool = False,
    test_split_ratio: float = 0.20,
    use_judge: bool = True,
    judge_endpoint: str = JUDGE_ENDPOINT,
    limit: Optional[int] = None,
    sc_mode: str = DEFAULT_SC_MODE,
    judge_model: str = JUDGE_MODEL,
    unstable_threshold: Optional[float] = None,
    kuzu_path: Optional[str] = None,
    dry_run_pool: bool = False,
) -> List[Dict[str, Any]]:
    facts = load_facts()
    if limit is not None and limit > 0:
        facts = facts[:limit]
        print(f"[LIMIT / SMOKE-RUN] Ograniczono korpus faktów do {len(facts)}")
    print(f"Loaded {len(facts)} qualified facts from {db_path}")
    full_texts = {f["key"]: f["text"] for f in facts}
    judge = RetrievalJudge(
        endpoint=judge_endpoint,
        enabled=use_judge and not dry_run_pool,
        sc_mode=sc_mode,
        model=judge_model,
        unstable_threshold=unstable_threshold,
    )
    FACT_IDX_BY_KEY.clear()
    FACT_IDX_BY_TEXT.clear()
    for j, f in enumerate(facts):
        FACT_IDX_BY_KEY[f["key"]] = j
        FACT_IDX_BY_TEXT[f["text"]] = j

    # C1: Deterministic Grouped Split by target_key
    sorted_keys = sorted(list({f["key"] for f in facts}))
    rng = np.random.RandomState(seed)
    shuffled_keys = rng.permutation(sorted_keys)
    n_train_keys = int(len(sorted_keys) * (1.0 - test_split_ratio))
    train_keys = set(shuffled_keys[:n_train_keys])
    test_keys = set(shuffled_keys[n_train_keys:])
    store_keys = set(sorted_keys)

    assert train_keys.isdisjoint(test_keys), "C1 Violation: train_keys and test_keys must have zero intersection!"
    print(f"C1 Grouped Split: train_keys={len(train_keys)}, test_keys={len(test_keys)}, disjoint=True")

    # V72.9: 5-Fold assignment strictly by target_key
    n_folds = 5
    key_to_fold = {}
    fold_splits = np.array_split(shuffled_keys, n_folds)
    for f_idx, f_keys in enumerate(fold_splits):
        for k in f_keys:
            key_to_fold[k] = int(f_idx)

    backend = load_vector_backend()
    enc = OnnxDenseEncoder(
        dimension=backend.dimension,
        model_dir=str(resolve_model_dir(backend.model_dir)),
        lazy_init=False,
    )

    doc_norm = [normalize_stored_text(f["text"]) for f in facts]
    t0 = time.perf_counter()
    doc_vecs = []
    for i in range(0, len(doc_norm), 16):
        doc_vecs.extend(enc.encode_batch(doc_norm[i : i + 16]))
    doc_matrix = np.array(doc_vecs, dtype=np.float32)
    norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-9
    doc_matrix = doc_matrix / norms
    print(f"Encoded {len(doc_matrix)} documents in {time.perf_counter() - t0:.2f}s")

    eq = equiv_sets(doc_vecs, threshold=EQUIV_THRESHOLD)

    # ------------------------------------------------------------------
    # V92.1: pula kandydatów. `hybrid` = produkcyjna ścieżka Qdranta (kopia
    # FALLBACK_QDRANT_PATH, od V97 F5 w trwałym katalogu cache). `matrix` = stary generator
    # (kontrola dowodowa, NIE produkcja).
    # ------------------------------------------------------------------
    store: Optional[QdrantVectorStore] = None
    cache: Dict[str, Any] = {}
    if source == "hybrid":
        col = backend.collection
        if col != "hermes_episodic_memory_68m":
            col = "hermes_episodic_memory_68m"
        eff_qdrant_path = resolve_qdrant_path(qdrant_path, allow_prod_store=allow_prod_store)
        store = QdrantVectorStore(
            collection_name=col,
            location=eff_qdrant_path,
            dimension=backend.dimension,
            encoder=enc,
        )
        if HYBRID_CACHE.is_file():
            try:
                cache = json.loads(HYBRID_CACHE.read_text(encoding="utf-8"))
            except Exception:
                cache = {}
        cached_store = CachedHybridVectorStore(store=store, cache=cache, cache_path=HYBRID_CACHE)
        eff_db_path = db_path or DEFAULT_DB
        kv_store = ReadOnlyKV(db_path=eff_db_path)
        kuzu_eff_path = resolve_kuzu_path(kuzu_path)
        if kuzu_eff_path:
            try:
                graph_store = CalibrationGraphStore(db_path=kuzu_eff_path)
                graph_store.hydrate_from_disk_sync()
                from atlas_memory.cognitive.retrieval.ppr import is_hipporag_enabled
                if is_hipporag_enabled():
                    mig_stats = graph_store.migrate_dual_node_schema_and_populate(facts, encoder=enc)
                    print(
                        f"Graph store connected: Kuzu @ {kuzu_eff_path} "
                        f"(hydrated + HippoRAG2 Dual-Node: passages={mig_stats['passages_added']}, "
                        f"contains={mig_stats['contains_edges_added']}, synonyms={mig_stats['synonym_edges_added']}, "
                        f"coverage={mig_stats['contains_coverage']*100:.1f}%)"
                    )
                else:
                    print(f"Graph store connected: Kuzu @ {kuzu_eff_path} (hydrated)")
            except Exception as kuzu_exc:
                print(f"[WARN] Nie udało się otworzyć bazy Kùzu @ {kuzu_eff_path}: {kuzu_exc}")
                graph_store = None
        else:
            graph_store = None

        hot_buffer = HotBufferPolicy(maxlen=25)
        # T9 (BUG-CAL27): Produkcyjny provider używa session_id="" (HERMES_DEFAULT_SESSION).
        # Sesja "calibration" czyniła 326/481 rekordów z effective_session=None niewidocznymi.
        # Dowód live UDS: VirtualBox_7 zwraca inny rekord w "" vs "calibration".
        eval_session_id = ""
        # Seed working session memory to simulate active agent context in the 5-channel harness
        # BUG-CAL29 (V97 F1): klucze z prefiksem `session:calib_harness:` mają effective_session
        # = "calib_harness", a gather_candidates przy session_id="" filtruje je (is_session_accessible)
        # — kanał hot_buffer był STRUCTURALNIE 0 kandydatów mimo seedowania. Klucze bez prefiksu
        # sesji (effective_session=None → domyślna sesja) są dostępne w "" (hermes_default).
        for idx in range(1, 4):
            hot_buffer.update(
                key=f"harness:calib_active_working_memory_{idx}",
                value=f"Agent active working context scratchpad note {idx}.",
                confidence=1.0,
                metadata={"session_id": eval_session_id, "source_type": "working_fact"},
                source_type="working_fact",
                timestamp=time.time(),
            )
        n_pts = await store.count()
        print(f"Hybrid pool: {store.collection_name} @ {eff_qdrant_path} ({n_pts} punktów), limit={pool_limit}, "
              f"sparse_ready={store._sparse_schema_ready()}, cache={len(cache)} wpisów, kv_store={eff_db_path} (ReadOnlyKV)")
    else:
        store = None
        cached_store = None
        kv_store = None
        graph_store = None
        hot_buffer = None
        eval_session_id = None
        print(f"!! SOURCE=matrix — stara ścieżka (sztuczna pula {len(facts)} faktów), TYLKO kontrola")

    cand_vecs: Dict[str, np.ndarray] = {}
    pool_sizes: List[int] = []
    in_pool = {"prose": [0, 0], "symbolic": [0, 0]}
    old_matrix_only: List[Tuple[str, List[Tuple[str, float]]]] = []
    target_filtered_by_policy_count = {"prose": 0, "symbolic": 0}
    target_deduped_away_count = {"prose": 0, "symbolic": 0}
    source_coverage = collections.Counter()
    # T1 (BUG-CAL22): Graph liveness PRE-filter tracking (liczniki aktualizuje build_query_pool)
    graph_counters = {"edges_pre_filter": 0, "queries_with_edges": 0}
    # F6 (BUG-CAL25): unikalne klucze hot_buffer naprzeciw liczbie pul w pełnym biegu.
    hot_buffer_keys: Set[str] = set()
    hot_buffer_pools = 0

    # V97 F1: wspólne zamknięcie zasobów puli dla ścieżek, które kończą PRZED sędzią.
    async def _close_pool_stores() -> None:
        if kv_store is not None:
            if inspect.iscoroutinefunction(kv_store.close):
                await kv_store.close()
            else:
                kv_store.close()
        if graph_store is not None and hasattr(graph_store, "close"):
            graph_store.close()
        judge.close()

    pool_scan_kwargs: Dict[str, Any] = dict(
        facts=facts, source=source, pool_limit=max(pool_limit, 50), enc=enc, cached_store=cached_store,
        kv_store=kv_store, graph_store=graph_store, hot_buffer=hot_buffer,
        eval_session_id=eval_session_id, doc_matrix=doc_matrix,
    )

    # V97 F1 (a): --dry-run-pool = pełna pula WSZYSTKICH zapytań + bramki, zero LLM/GPU,
    # zero zapisu zbioru; raport obok --output jako <output>.pool.json, kod 0 w `main()`.
    if dry_run_pool:
        await dry_run_pool_report(output_path=Path(output_path), **pool_scan_kwargs)
        await _close_pool_stores()
        return []

    # V97 F1 (b): normalny bieg — te same bramki puli PRZED pierwszym wywołaniem sędziego.
    # Próbka par budowana jest tu zamiast w pętli, więc naruszenie kończy bieg z kodem 2
    # bez ani jednego żądania do modelu (porty 20122/20128/11235 nie są dotykane).
    await pre_judge_pool_gate(
        graph_gate_applies=(graph_store is not None and limit is None),
        **pool_scan_kwargs,
    )

    # T7: Zapis atomowy, append z flush i idempotencja na starcie
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_triples = set()
    records: List[Dict[str, Any]] = []
    if output_path.exists():
        for ln in output_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
                existing_triples.add((r.get("query_type"), r.get("target_key"), r.get("cand_key")))
                records.append(r)
            except Exception:
                pass

    out_fh = output_path.open("a", encoding="utf-8")
    current_sources_in_query: List[str] = []

    def _emit_record(rec: Dict[str, Any]) -> None:
        triple = (rec.get("query_type"), rec.get("target_key"), rec.get("cand_key"))
        if triple in existing_triples:
            return
        if "sources_in_pool" not in rec and current_sources_in_query:
            rec["sources_in_pool"] = list(current_sources_in_query)
        validate_test_record_provenance(rec, use_judge=use_judge)
        existing_triples.add(triple)
        records.append(rec)
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_fh.flush()


    # C2 & B1: Separate queries into symbolic and prose (self_query EXCLUDED)
    for i, fact in enumerate(facts):
        tgt_key = fact["key"]
        tgt_fam = extract_family(tgt_key)
        split_name = "train" if tgt_key in train_keys else "test"
        print(f"\n[FACT {i+1}/{len(facts)}] Key: {tgt_key[:50]} (Split: {split_name})", flush=True)

        variants = _query_variants(fact)

        for q_type, q_text in variants:
            # M4: Read-path policy filter (CAL17 & CAL15) — odrzucamy rekordy konwersacyjne i filtry przed emisją
            is_target_filtered = (
                any(tgt_key.lower().startswith(p) for p in PROBE_PREFIXES)
                or _is_filtered(tgt_key, fact.get("text", ""), None, None)
                or is_conversational_churn(tgt_key, fact.get("text", ""))
            )
            if is_target_filtered:
                target_filtered_by_policy_count[q_type] += 1
                # T3 (BUG-CAL24): Emit metadata record instead of silent skip.
                # label=-1 distinguishes from 0/1 and is excluded from C3/contradiction checks.
                _emit_record({
                    "split": split_name,
                    "fold": key_to_fold[tgt_key],
                    "mode": "plain",
                    "query_type": q_type,
                    "query": q_text,
                    "target_key": tgt_key,
                    "cand_key": tgt_key,
                    "family": tgt_fam,
                    "score_top1": 0.0,
                    "score_margin": 0.0,
                    "score_dense": 0.0,
                    "score_sparse": 0.0,
                    "query_len": len(q_text.split()),
                    "log_query_len": round(float(np.log(max(1, len(q_text.split())))), 4),
                    "latent_score": 0.0,
                    "self_consistency": None,
                    "judge_rubric": None,
                    "judge_reasoning": "Target filtered by read-path policy (_is_filtered / is_conversational_churn)",
                    "judge_thinking": None,
                    "label": -1,
                    "label_source": "target_filtered_by_policy",
                    "prior_used": False,
                    "neg_type": "filtered",
                    "pool_size": 0,
                    "pool_source": source,
                    "sources_in_pool": [],
                })
                continue

            # B1 check: ensure predicate matches intended classification
            classified_is_sym = is_symbolic_or_identifier(q_text)
            if q_type == "symbolic":
                assert classified_is_sym, f"Query '{q_text}' expected symbolic but classified prose"
            else:
                assert not classified_is_sym, f"Query '{q_text}' expected prose but classified symbolic"

            q_vec = np.array(
                enc.encode_query(q_text) if hasattr(enc, "encode_query") else enc.encode(q_text),
                dtype=np.float32,
            )
            q_vec = q_vec / (float(np.linalg.norm(q_vec)) + 1e-9)

            # V97 F1: pula przez wspólny build_query_pool (5-źródłowy harness albo macierz:
            # pre-sort T4, coverage i PRE-filter grafu w jednym miejscu, wspólnym z dry-runem).
            ranked, by_key, current_sources_in_query = await build_query_pool(
                q_text=q_text,
                source=source,
                pool_limit=max(pool_limit, 50),
                enc=enc,
                cached_store=cached_store,
                kv_store=kv_store,
                graph_store=graph_store,
                hot_buffer=hot_buffer,
                eval_session_id=eval_session_id,
                facts=facts,
                doc_matrix=doc_matrix,
                source_coverage=source_coverage,
                graph_counters=graph_counters,
                hot_buffer_keys=hot_buffer_keys,
            )

            pool_sizes.append(len(ranked))
            if hot_buffer is not None:
                hot_buffer_pools += 1
            # Dowód różnicy: dla pierwszych 8 par zapisujemy skład obu puł.
            if len(old_matrix_only) < 8:
                old_matrix_only.append((f"{q_type}|{tgt_key}", [
                    (str(c.get("key") or c.get("subject") or ""), round(float(c.get("score", 0.0)), 4))
                    for c in ranked[:8]
                ]))

            tgt_cand, was_deduped = _cand_for_fact(ranked, by_key, fact)
            if was_deduped:
                target_deduped_away_count[q_type] += 1
            pos_score = float(tgt_cand["score"]) if tgt_cand else _score_for_fact(ranked, by_key, fact)
            pos_vec = float(tgt_cand.get("dense_score", tgt_cand.get("vector_score", 0.0))) if tgt_cand else 0.0
            pos_sparse = float(tgt_cand.get("sparse_score", 0.0)) if tgt_cand else 0.0

            in_pool[q_type][1] += 1
            if pos_score > 0.0:
                in_pool[q_type][0] += 1
            top10 = [float(c["score"]) for c in ranked[:10]]
            if len(top10) >= 3:
                median_pool = float(np.median(top10))
            elif len(top10) == 2:
                median_pool = float(min(top10))
            else:
                median_pool = 0.0
            pos_margin = float(pos_score - median_pool)

            # Cel pozytywny (T1, T2, T5):
            if q_type == "symbolic":
                # T5: Klucze symboliczne NIE są ewaluowane przez sędziego LLM
                tgt_verdict = JudgeVerdict(1, 1.0, {}, "Symbolic exact key match", None, "symbolic_exact_match", prior_used=False)
                probe = symbolic_rank_probe(ranked, tgt_key)
                if not probe["hit_top1"]:
                    print(f"[WARN] symbolic_rank_probe top1 miss dla {tgt_key}: rank={probe['rank']}/{probe['pool_size']}")
            else:
                tgt_prior = active_sampling_prior(
                    cand_score=pos_score,
                    cand_margin=pos_margin,
                    cand_key=tgt_key,
                    tgt_key=tgt_key,
                    q_type=q_type,
                    vector_score=pos_vec,
                    store_keys=store_keys,
                )
                if tgt_prior is not None:
                    tgt_verdict = tgt_prior
                else:
                    if use_judge:
                        tgt_verdict = await judge.judge_batch_self_consistency(q_text, tgt_key, fact["text"], k=3)
                    else:
                        tgt_verdict = JudgeVerdict(1, 1.0, {}, "Mechaniczny fallback (no judge)", None, "no_judge_fallback", prior_used=True)

            if tgt_verdict.label is None:
                # T6: Awaria sędziego -> pomijamy rekord zamiast imputować błąd
                print(f"[WARN] Target judging failed for {tgt_key}, skipping query (T6).")
                continue

            _emit_record({
                "split": split_name,
                "fold": key_to_fold[tgt_key],
                "mode": "plain",
                "query_type": q_type,
                "query": q_text,
                "target_key": tgt_key,
                "cand_key": tgt_key,
                "family": tgt_fam,
                "score_top1": round(pos_score, 4),
                "score_margin": round(pos_margin, 4),
                "score_dense": round(pos_vec, 4),
                "score_sparse": round(pos_sparse, 4),
                "query_len": len(q_text.split()),
                "log_query_len": round(float(np.log(max(1, len(q_text.split())))), 4),
                "latent_score": round(float(tgt_verdict.latent_score), 4),
                "self_consistency": tgt_verdict.self_consistency,
                "judge_rubric": tgt_verdict.rubric if tgt_verdict.rubric else None,
                "judge_reasoning": tgt_verdict.reasoning,
                "judge_thinking": tgt_verdict.thinking,
                "label": int(tgt_verdict.label),
                "label_source": tgt_verdict.source,
                "prior_used": bool(tgt_verdict.prior_used),
                "neg_type": "positive",
                "pool_size": len(ranked),
                "pool_source": source,
                "target_deduped": bool(was_deduped),
            })

            # 2. Deterministic Hard Negative (label=0) (C3) — na REALNEJ puli.
            non_twin: List[Dict[str, Any]] = []
            for c in ranked:
                k = str(c.get("key") or c.get("subject") or "")
                txt = str(c.get("object") or "")
                if k == tgt_key:
                    continue
                if txt and txt not in cand_vecs:
                    v = np.array(enc.encode(normalize_stored_text(txt)), dtype=np.float32)
                    cand_vecs[txt] = v / (float(np.linalg.norm(v)) + 1e-9)
                if _is_twin(k, txt, i, doc_matrix, eq, cand_vecs, facts=facts):
                    continue
                non_twin.append(c)
            if not non_twin:
                continue

            cross_fam = [c for c in non_twin
                         if extract_family(str(c.get("key") or c.get("subject") or "")) != tgt_fam]
            candidate_list = cross_fam if (i % 2 == 0 and cross_fam) else non_twin
            chosen_neg = None
            chosen_neg_type = "cross_family" if (i % 2 == 0 and cross_fam) else "near_miss"

            added_pos_keys = {tgt_key}

            if q_type == "symbolic":
                # T5: Negatywy dla zapytań symbolicznych są strukturalne (zero wywołań sędziego)
                for cand in candidate_list:
                    c_key = str(cand.get("key") or cand.get("subject") or "")
                    if c_key == tgt_key:
                        continue
                    c_score = float(cand["score"])
                    c_margin = float(c_score - median_pool)
                    c_vec = float(cand.get("dense_score", cand.get("vector_score", 0.0)))
                    c_sparse = float(cand.get("sparse_score", 0.0))
                    _emit_record({
                        "split": split_name,
                        "fold": key_to_fold[tgt_key],
                        "mode": "plain",
                        "query_type": "symbolic",
                        "query": q_text,
                        "target_key": tgt_key,
                        "cand_key": c_key,
                        "family": tgt_fam,
                        "score_top1": round(c_score, 4),
                        "score_margin": round(c_margin, 4),
                        "score_dense": round(c_vec, 4),
                        "score_sparse": round(c_sparse, 4),
                        "query_len": len(q_text.split()),
                        "log_query_len": round(float(np.log(max(1, len(q_text.split())))), 4),
                        "latent_score": 0.0,
                        "self_consistency": None,
                        "judge_rubric": None,
                        "judge_reasoning": "Symbolic structural negative",
                        "judge_thinking": None,
                        "label": 0,
                        "label_source": "symbolic_structural_negative",
                        "prior_used": False,
                        "neg_type": chosen_neg_type,
                        "pool_size": len(ranked),
                        "pool_source": source,
                    })
                    chosen_neg = cand
                    break
            else:
                # Prose: Ocena kandydatów przez sędziego LLM
                for cand in candidate_list:
                    c_key = str(cand.get("key") or cand.get("subject") or "")
                    if c_key in added_pos_keys:
                        continue

                    c_txt = str(cand.get("object") or "")
                    if len(c_txt.split()) < 5:
                        c_txt = full_texts.get(c_key, c_txt)

                    cand_score = float(cand["score"])
                    cand_margin = float(cand_score - median_pool)
                    cand_vec = float(cand.get("dense_score", cand.get("vector_score", 0.0)))
                    cand_sparse = float(cand.get("sparse_score", 0.0))

                    if split_name == "test" and q_type == "prose":
                        # T2: Zbiór testowy prose musi być w 100% zweryfikowany przez sędziego lub człowieka
                        if len(c_txt.split()) < 5:
                            continue
                        c_prior = None
                    else:
                        c_prior = active_sampling_prior(
                            cand_score=cand_score,
                            cand_margin=cand_margin,
                            cand_key=c_key,
                            tgt_key=tgt_key,
                            q_type=q_type,
                            vector_score=cand_vec,
                            store_keys=store_keys,
                        )

                    if c_prior is not None:
                        verdict = c_prior
                    else:
                        if use_judge:
                            verdict = await judge.judge_batch_self_consistency(q_text, c_key, c_txt, k=3)
                        else:
                            verdict = JudgeVerdict(0, 0.0, {}, "No judge fallback", None, "no_judge_fallback", prior_used=True)

                    if verdict.label is None:
                        # T6: Drop, nie imputuj 0
                        continue

                    if verdict.label == 1:
                        # Wykryto tożsamy semantycznie synonim pod innym kluczem
                        _emit_record({
                            "split": split_name,
                            "fold": key_to_fold[tgt_key],
                            "mode": "plain",
                            "query_type": q_type,
                            "query": q_text,
                            "target_key": tgt_key,
                            "cand_key": c_key,
                            "family": tgt_fam,
                            "score_top1": round(cand_score, 4),
                            "score_margin": round(cand_margin, 4),
                            "score_dense": round(cand_vec, 4),
                            "score_sparse": round(cand_sparse, 4),
                            "query_len": len(q_text.split()),
                            "log_query_len": round(float(np.log(max(1, len(q_text.split())))), 4),
                            "latent_score": round(float(verdict.latent_score), 4),
                            "self_consistency": verdict.self_consistency,
                            "judge_rubric": verdict.rubric if verdict.rubric else None,
                            "judge_reasoning": verdict.reasoning,
                            "judge_thinking": verdict.thinking,
                            "label": 1,
                            "label_source": verdict.source,
                            "prior_used": bool(verdict.prior_used),
                            "neg_type": "synonym_positive",
                            "pool_size": len(ranked),
                            "pool_source": source,
                        })
                        added_pos_keys.add(c_key)
                        continue
                    else:
                        # Zweryfikowany negatyw (label == 0)
                        chosen_neg = cand
                        _emit_record({
                            "split": split_name,
                            "fold": key_to_fold[tgt_key],
                            "mode": "plain",
                            "query_type": q_type,
                            "query": q_text,
                            "target_key": tgt_key,
                            "cand_key": c_key,
                            "family": tgt_fam,
                            "score_top1": round(cand_score, 4),
                            "score_margin": round(cand_margin, 4),
                            "score_dense": round(cand_vec, 4),
                            "score_sparse": round(cand_sparse, 4),
                            "query_len": len(q_text.split()),
                            "log_query_len": round(float(np.log(max(1, len(q_text.split())))), 4),
                            "latent_score": round(float(verdict.latent_score), 4),
                            "self_consistency": verdict.self_consistency,
                            "judge_rubric": verdict.rubric if verdict.rubric else None,
                            "judge_reasoning": verdict.reasoning,
                            "judge_thinking": verdict.thinking,
                            "label": 0,
                            "label_source": verdict.source,
                            "prior_used": bool(verdict.prior_used),
                            "neg_type": chosen_neg_type,
                            "pool_size": len(ranked),
                            "pool_source": source,
                        })
                        break

            if chosen_neg is None:
                safe_fallbacks = [
                    c for c in (candidate_list or non_twin)
                    if str(c.get("key") or c.get("subject") or "") not in added_pos_keys
                ]
                if safe_fallbacks:
                    f_cand = safe_fallbacks[-1]
                    f_key = str(f_cand.get("key") or f_cand.get("subject") or "")
                    f_txt = str(f_cand.get("object") or "")
                    f_score = float(f_cand["score"])
                    f_margin = float(f_score - median_pool)
                    f_vec = float(f_cand.get("dense_score", f_cand.get("vector_score", 0.0)))
                    f_sparse = float(f_cand.get("sparse_score", 0.0))

                    if q_type == "symbolic":
                        f_source = "symbolic_structural_negative"
                        f_prior = False
                    elif split_name == "test" and use_judge:
                        # T8: W teście wymagana pełna proweniencja sędziego
                        fb_v = await judge.judge_batch_self_consistency(q_text, f_key, f_txt, k=3)
                        if fb_v.label == 0:
                            f_source = fb_v.source
                            f_prior = False
                        else:
                            # F-C: Nie imputuj i nie wysadzaj biegu — pomiń rekord w teście (T6 drop)
                            print(f"[WARN] Fallback negative in test split judged as {fb_v.label}, dropping query (F-C).")
                            continue
                    else:
                        f_source = "fallback_non_positive"
                        f_prior = True

                    _emit_record({
                        "split": split_name,
                        "fold": key_to_fold[tgt_key],
                        "mode": "plain",
                        "query_type": q_type,
                        "query": q_text,
                        "target_key": tgt_key,
                        "cand_key": f_key,
                        "family": tgt_fam,
                        "score_top1": round(f_score, 4),
                        "score_margin": round(f_margin, 4),
                        "score_dense": round(f_vec, 4),
                        "score_sparse": round(f_sparse, 4),
                        "query_len": len(q_text.split()),
                        "log_query_len": round(float(np.log(max(1, len(q_text.split())))), 4),
                        "latent_score": 0.0,
                        "self_consistency": None,
                        "judge_rubric": None,
                        "judge_reasoning": "Fallback twardego negatywu z puli nie-bliźniaków",
                        "judge_thinking": None,
                        "label": 0,
                        "label_source": f_source,
                        "prior_used": f_prior,
                        "neg_type": chosen_neg_type,
                        "pool_size": len(ranked),
                        "pool_source": source,
                    })
                else:
                    other_facts = [
                        f for f in facts
                        if f["key"] not in added_pos_keys and not _is_twin(f["key"], f["text"], i, doc_matrix, eq, cand_vecs, facts=facts)
                    ]
                    if other_facts:
                        tail_fact = other_facts[(i * 7 + 13) % len(other_facts)]
                        tail_key = tail_fact["key"]
                        tail_txt = tail_fact["text"]
                        if q_type == "symbolic":
                            t_source = "symbolic_structural_negative"
                            t_prior = False
                        elif split_name == "test" and use_judge:
                            tb_v = await judge.judge_batch_self_consistency(q_text, tail_key, tail_txt, k=3)
                            if tb_v.label == 0:
                                t_source = tb_v.source
                                t_prior = False
                            else:
                                # F-C: Nie imputuj i nie wysadzaj biegu — pomiń rekord w teście (T6 drop)
                                print(f"[WARN] Tail fallback in test split judged as {tb_v.label}, dropping query (F-C).")
                                continue
                        else:
                            t_source = "tail_safe_fallback"
                            t_prior = True

                        _emit_record({
                            "split": split_name,
                            "fold": key_to_fold[tgt_key],
                            "mode": "plain",
                            "query_type": q_type,
                            "query": q_text,
                            "target_key": tgt_key,
                            "cand_key": tail_key,
                            "family": tgt_fam,
                            "score_top1": 0.0,
                            "score_margin": round(0.0 - median_pool, 4),
                            "score_dense": 0.0,
                            "score_sparse": 0.0,
                            "query_len": len(q_text.split()),
                            "log_query_len": round(float(np.log(max(1, len(q_text.split())))), 4),
                            "latent_score": 0.0,
                            "self_consistency": None,
                            "judge_rubric": None,
                            "judge_reasoning": "Tail safe negative fallback",
                            "judge_thinking": None,
                            "label": 0,
                            "label_source": t_source,
                            "prior_used": t_prior,
                            "neg_type": "tail_negative",
                            "pool_size": len(ranked),
                            "pool_source": source,
                        })

    if source == "hybrid":
        HYBRID_CACHE.write_text(json.dumps(cache), encoding="utf-8")
        print(f"Zapisano cache puł hybrydowych: {HYBRID_CACHE} ({len(cache)} zapytań)")

    if kv_store is not None:
        if inspect.iscoroutinefunction(kv_store.close):
            await kv_store.close()
        else:
            kv_store.close()
    if graph_store is not None and hasattr(graph_store, "close"):
        graph_store.close()
    judge.close()
    out_fh.close()

    content = output_path.read_bytes()
    sha256_hash = hashlib.sha256(content).hexdigest()
    print(f"\nSaved {len(records)} calibration records to {output_path}")
    print(f"Dataset SHA-256: {sha256_hash}")

    # Metrics on splits — exclude T3 metadata records (label=-1 = target_filtered_by_policy)
    eval_records = [r for r in records if int(r["label"]) >= 0]
    filtered_records = [r for r in records if int(r["label"]) == -1]
    train_recs = [r for r in eval_records if r["split"] == "train"]
    test_recs = [r for r in eval_records if r["split"] == "test"]

    test_pos = sum(1 for r in test_recs if r["label"] == 1)
    test_neg = sum(1 for r in test_recs if r["label"] == 0)
    pos_ratio = test_pos / max(1, len(test_recs))

    print("\n" + "=" * 80)
    print(f" 📊 CALIBRATION DATASET INTEGRITY & BALANCE (V93 SOTA, pool={source})")
    print("=" * 80)
    print(f"Total Records: {len(records)} (eval={len(eval_records)}, filtered_by_policy={len(filtered_records)})")
    print(f"Train Records: {len(train_recs)} (keys={len(train_keys)})")
    print(f"Test Records:  {len(test_recs)} (keys={len(test_keys)})")
    print(f"Test Positives: {test_pos} ({pos_ratio * 100:.1f}%), Test Negatives: {test_neg}")
    if pool_sizes:
        print(f"Pool size: med={int(np.median(pool_sizes))}, min={min(pool_sizes)}, max={max(pool_sizes)}")
    for qt in ("prose", "symbolic"):
        hit, tot = in_pool[qt]
        filt = target_filtered_by_policy_count[qt]
        dedup = target_deduped_away_count[qt]
        print(f"  - [{qt:<8}] target w puli: {hit}/{tot} ({100.0 * hit / max(1, tot):.1f}%)")
        print(f"  - [{qt:<8}] filtered by policy (M4): {filt}")
        print(f"  - [{qt:<8}] deduped away (M3): {dedup}")
        # T3: Closing balance — all targets accounted for
        print(f"  - [{qt:<8}] closing balance: in_pool({hit}) + filtered({filt}) + deduped({dedup}) + missed({tot - hit}) = {hit + filt + dedup + (tot - hit)} targets")

    # V97 F1: pełny przekrój bramek puli tym samym helperem, co przed sędzią i w
    # `--dry-run-pool` — jeden zestaw reguł, jedno źródło prawdy.
    if source == "hybrid":
        print("\n" + "=" * 80)
        print(" 🎯 SOURCE COVERAGE GATE (5-CHANNEL HARNESS)")
        print("=" * 80)
        for ch in HARNESS_CHANNELS:
            print(f"  - [{ch:<15}]: {source_coverage[ch]} kandydatów w pulach (POST-filter)")
        # T1 (BUG-CAL22): Graph liveness PRE-filter stats
        print(f"  - [{'graph_pre':<15}]: {graph_counters['edges_pre_filter']} krawędzi PRE-filter, "
              f"{graph_counters['queries_with_edges']} zapytań z krawędziami")
    post_run_violations = validate_pool_gates(
        source_coverage,
        graph_counters["edges_pre_filter"],
        graph_counters["queries_with_edges"],
        {
            "in_pool": in_pool,
            "source": source,
            "hot_buffer_enabled": hot_buffer is not None,
            "hot_buffer_unique_keys": len(hot_buffer_keys),
            "hot_buffer_pools": hot_buffer_pools,
            # T1: bramka grafu tylko przy pełnym przebiegu (limit=None) — patrz BUG-CAL22.
            "graph_gate_applies": graph_store is not None and limit is None,
        },
    )
    if post_run_violations:
        raise AssertionError(
            "Bramki puli (validate_pool_gates) naruszone po pełnym biegu "
            f"({len(post_run_violations)}): " + " | ".join(post_run_violations)
        )

    # Check conditions C3 (tylko przy pełnym przebiegu bez limitu)
    if limit is None:
        for qt in ("symbolic", "prose"):
            qt_test = [r for r in test_recs if r["query_type"] == qt]
            qt_pos = sum(1 for r in qt_test if r["label"] == 1)
            qt_neg = sum(1 for r in qt_test if r["label"] == 0)
            print(f"  - [{qt:<8}] Test N={len(qt_test)} (pos={qt_pos}, neg={qt_neg})")
            assert qt_neg >= 30, f"C3 Failure: Test negatives for {qt} is {qt_neg} < 30!"

        assert 0.40 <= pos_ratio <= 0.60, f"C3 Failure: Positive ratio {pos_ratio} not in [0.40, 0.60]!"
        assert len({r["fold"] for r in eval_records}) == n_folds, "Fold count != 5"
        # T8: Weryfikacja bramki proweniencji dla całego zbioru
        verify_provenance_gate(eval_records)
    else:
        print(f"[LIMIT RUN] Pominięto C3 min-negatives check dla ograniczonej próby N={len(records)}")

    # Zero-Contradiction Invariant Gate (ANOMALIA-01)
    # Exclude T3 metadata records (label=-1 = target_filtered_by_policy)
    label_map: Dict[Tuple[str, str], set[int]] = {}
    for r in eval_records:
        pair = (str(r["query"]), str(r["cand_key"]))
        label_map.setdefault(pair, set()).add(int(r["label"]))
    contradictions = [pair for pair, labels in label_map.items() if len(labels) > 1]
    assert not contradictions, (
        f"ANOMALIA-01 Zero-Contradiction Invariant VIOLATED: {len(contradictions)} conflicting (query, cand_key) pairs detected! "
        f"Examples: {contradictions[:5]}"
    )
    print("Zero-Contradiction Invariant (ANOMALIA-01) satisfied: 0 contradictory label pairs.")

    print("All C1-C4 dataset invariants satisfied successfully.")
    print("=" * 80)

    judge.close()

    judged_records = [r for r in records if str(r.get("label_source", "")).startswith("llm_judge") or str(r.get("label_source", "")).startswith("self_consistency")]
    flips = sum(1 for r in judged_records if r["label"] == 1 and r.get("neg_type") == "synonym_positive")
    stats = {
        "prompt_version": JUDGE_PROMPT_VERSION,
        "endpoint": judge_endpoint,
        "total_records": len(records),
        "pairs_judged_by_llm": len(judged_records),
        "active_sampling_bypassed": len(records) - len(judged_records),
        "synonyms_discovered_as_positives": flips,
        "unstable_count": judge.unstable_count,
        "judge_errors": judge.judge_errors_count,
        "out_dataset": str(output_path),
        "cache_file": str(JUDGE_CACHE_FILE),
    }
    try:
        JUDGE_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        JUDGE_STATS_FILE.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    # V99/ADR-017: ścieżka --train (spawn train_and_export_calibration.py) usunięta
    # razem z producentem reliktowego artefaktu.

    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="ATLAS calibration dataset builder & LLM judge (V93 SOTA)")
    ap.add_argument("--source", choices=("hybrid", "matrix"), default="hybrid",
                    help="hybrid = realna pula z Qdranta (produkcja); matrix = stara sztuczna macierz (kontrola)")
    ap.add_argument("--output", default=str(OUTPUT_JSONL), help="ścieżka pliku .jsonl (kontrola może pisać do /tmp)")
    ap.add_argument("--pool-limit", type=int, default=DEFAULT_POOL_LIMIT)
    ap.add_argument("--qdrant-path", default=None, help="Ścieżka do bazy Qdrant (domyślnie kopia robocza)")
    ap.add_argument("--allow-prod-store", action="store_true", default=False,
                    help="Zezwól na bezpośrednie użycie produkcyjnego magazynu Qdrant")
    ap.add_argument("--test-split-ratio", type=float, default=0.20,
                    help="Proporcja podziału testowego (domyślnie 0.20 dla splitu 80/20)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="Ograniczenie liczby faktów do przetworzenia (np. 20 dla smoke-run)")
    ap.add_argument("--judge", dest="use_judge", action="store_true", default=True,
                    help="Włącz sędziego LLM z thinkingiem (domyślnie włączony)")
    ap.add_argument("--no-judge", dest="use_judge", action="store_false",
                    help="Wyłącz sędziego LLM (fallback do czysto mechanicznego doboru)")
    ap.add_argument("--endpoint", default=JUDGE_ENDPOINT, help="URL endpointu lokalnego sędziego LLM")
    ap.add_argument("--model", default=JUDGE_MODEL, help="Nazwa modelu sędziego LLM (np. local lub antigravity/gemini-3.8-flash-tiered)")
    ap.add_argument("--unstable-threshold", type=float, default=UNSTABLE_SIGMA2_THRESHOLD,
                    help="Próg wariancji sigma2 dla flagi unstable (domyślnie z rozkładu empirycznego P95)")
    ap.add_argument("--kuzu-path", default=None, help=f"Ścieżka do kopii bazy Kùzu (domyślnie {CALIB_CACHE_DIR / 'kuzu_calib'}, trwały katalog)")
    ap.add_argument("--sc-mode", choices=("independent", "batch"), default=DEFAULT_SC_MODE,
                    help="Tryb self-consistency: independent (K niezależnych wywołań HTTP) lub batch (1 wywołanie z K wariantami)")
    ap.add_argument("--dry-run-pool", action="store_true", default=False,
                    help="T2: Buduje pule kandydatów i sprawdza bramki (coverage, target-in-pool, M3/M4) "
                         "BEZ uruchamiania sędziego LLM. Koszt ~2s CPU, wynik: raport JSON.")
    args = ap.parse_args()

    out = Path(args.output)
    if args.source == "matrix" and out.resolve() == OUTPUT_JSONL.resolve():
        raise SystemExit("ODMAWIAM nadpisania produkcyjnego datasetu pulą sztuczną (--source matrix). "
                         "Podaj --output np. ~/atlas-run/calib_cache/probe.jsonl (cokolwiek poza datasetem prod)")
    recs = asyncio.run(build_dataset(
        output_path=out,
        seed=args.seed,
        source=args.source,
        pool_limit=args.pool_limit,
        qdrant_path=args.qdrant_path,
        allow_prod_store=args.allow_prod_store,
        test_split_ratio=args.test_split_ratio,
        use_judge=args.use_judge,
        judge_endpoint=args.endpoint,
        limit=args.limit,
        sc_mode=args.sc_mode,
        judge_model=args.model,
        unstable_threshold=args.unstable_threshold,
        kuzu_path=args.kuzu_path,
        dry_run_pool=args.dry_run_pool,
    ))
    # V97 F1: dry-run puli nie emituje rekordów — bramki naruszone => SystemExit(2) w helperze,
    # bramki zdane => kod 0 bez żadnego wywołania sędziego.
    if args.dry_run_pool:
        return 0
    return 0 if recs else 1


if __name__ == "__main__":
    raise SystemExit(main())
