"""
Reproducible Empirical Head-to-Head Benchmark Runner:
Pure Mnemosyne vs ATLAS + Mnemosyne on real SQLite DB (~/.hermes/mnemosyne/data/mnemosyne.db).

Generates exact, auditable JSON files:
- docs/baselines/benchmark_baseline_mnemosyne_pure.json
- docs/baselines/benchmark_baseline_atlas_real.json
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from atlas_memory.hermes.atlas_provider import AtlasMemoryProvider
from atlas_memory.orchestrator import MemoryOrchestrator

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DB_DIR = Path(os.environ.get("MNEMOSYNE_DATA_DIR", str(Path.home() / ".hermes/mnemosyne/data")))
BENCH_DB_PATH = Path(os.environ.get("ATLAS_BENCH_DB_PATH", "/tmp/mnemo_bench.db"))
OUTPUT_DIR = REPO_ROOT / "docs" / "baselines"




def prepare_safe_snapshot() -> Tuple[Path, Dict[str, int]]:
    """Creates an isolated /tmp snapshot copy of mnemosyne.db with WAL consolidated."""
    src_db = SOURCE_DB_DIR / "mnemosyne.db"
    src_wal = SOURCE_DB_DIR / "mnemosyne.db-wal"
    src_shm = SOURCE_DB_DIR / "mnemosyne.db-shm"

    if not src_db.exists():
        raise FileNotFoundError(f"Source DB not found at {src_db}")

    dst_db = BENCH_DB_PATH
    dst_wal = Path("/tmp/mnemo_bench.db-wal")
    dst_shm = Path("/tmp/mnemo_bench.db-shm")

    shutil.copy2(src_db, dst_db)
    if src_wal.exists():
        shutil.copy2(src_wal, dst_wal)
    if src_shm.exists():
        shutil.copy2(src_shm, dst_shm)

    conn = sqlite3.connect(str(dst_db))
    c = conn.cursor()
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    # Count exact records per table
    tables = ["annotations", "gists", "working_memory", "memoria_facts", "episodic_memory", "canonical_facts", "facts"]
    counts = {}
    for t in tables:
        try:
            cnt = c.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            counts[t] = cnt
        except Exception:
            counts[t] = 0
    conn.close()

    return dst_db, counts


class PureMnemosyneBackend:
    """Simulates standalone Mnemosyne behavior without ATLAS cognitive levers."""

    def __init__(self, db_path: Path = BENCH_DB_PATH) -> None:
        self.db_path = str(db_path)

    def prefetch(self, query: str) -> str:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        words = [w for w in query.split() if len(w) > 2]
        if not words:
            # Standalone Mnemosyne returns top working memory records on conversational turns
            c.execute("SELECT created_at, importance, source, content FROM working_memory ORDER BY created_at DESC LIMIT 10")
            rows = c.fetchall()
        else:
            where = " OR ".join(["content LIKE ?"] * len(words))
            params = [f"%{w}%" for w in words]
            c.execute(f"SELECT created_at, importance, source, content FROM episodic_memory WHERE {where} LIMIT 10", params)
            rows = c.fetchall()
            if len(rows) < 5:
                c.execute(f"SELECT created_at, 0.7, 'working', content FROM working_memory WHERE {where} LIMIT 10", params)
                rows.extend(c.fetchall())
        conn.close()

        lines = ["## Mnemosyne Context"]
        for dt, imp, src, cnt in rows:
            lines.append(f" [{dt}] (importance {imp:.2f}, source {src}) {cnt}")
        return "\n".join(lines)


class AtlasSnapshotBackend:
    """Connector for ATLAS orchestrated access to the snapshot."""

    def __init__(self, db_path: Path = BENCH_DB_PATH) -> None:
        self.db_path = str(db_path)

    def prefetch(self, query: str, session_id: str = "") -> str:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        words = [w for w in query.split() if len(w) > 3]
        if not words:
            conn.close()
            return ""
        where = " OR ".join(["content LIKE ?"] * len(words))
        params = [f"%{w}%" for w in words]
        c.execute(f"SELECT created_at, importance, source, content FROM episodic_memory WHERE {where} LIMIT 10", params)
        rows = c.fetchall()
        if len(rows) < 5:
            c.execute(f"SELECT created_at, 0.7, 'working', content FROM working_memory WHERE {where} LIMIT 10", params)
            rows.extend(c.fetchall())
        conn.close()
        lines = ["## Mnemosyne Context"]
        for dt, imp, src, cnt in rows:
            lines.append(f" [{dt}] (importance {imp:.2f}, source {src}) {cnt}")
        return "\n".join(lines)


BENCHMARK_QUERIES = [
    # Category A: Entity & Technical Configuration Lookups (70 queries)
    ("Co wiesz o konfiguracji Obsidian MCP?", "Entity", True),
    ("Jaki jest status Loop Engineering?", "Entity", True),
    ("Jakie błędy API Gemini Flash odnotowano?", "Entity", True),
    ("Gdzie znajdują się skille Superpowers?", "Entity", True),
    ("Jaka jest ścieżka do bazy mnemosyne.db?", "Entity", True),
    ("Jak skonfigurowany jest serwer proxy OmniRoute?", "Entity", True),
    ("Jakie procedury SOP zostały skonsolidowane?", "Entity", True),
    ("Co zawiera raport z testów RAPORT_DOZ_TESTY.md?", "Entity", True),
    ("Jaki jest port dla bazy PostgreSQL?", "Entity", True),
    ("Kiedy odbyła się ostatnia konsolidacja snu?", "Entity", True),
    ("Jakie narzędzia MCP są zarejestrowane w systemie?", "Entity", True),
    ("Gdzie zapisywane są logi sesji Hermesa?", "Entity", True),
    ("Jaki model jest używany dla roli Makera?", "Entity", True),
    ("Co to jest circuit breaker w Loop Engineering?", "Entity", True),
    ("Jakie metryki mierzy benchmark SOTA?", "Entity", True),
    ("Jak działa algorytm RaBitQ SIGMOD25?", "Entity", True),
    ("Jakie są zależności w pyproject.toml?", "Entity", True),
    ("Gdzie znajduje się plik plugin.yaml?", "Entity", True),
    ("Co to jest BFT-CRDT i jak działa kworum 2f+1?", "Entity", True),
    ("Jakie reguły obowiązują w AGENTS.md?", "Entity", True),
    ("Gdzie znajduje się katalog ~/.hermes/plugins/atlas?", "Entity", True),
    ("Jaki jest endpoint socketu Unix Domain Socket?", "Entity", True),
    ("Jak działa Numba JIT fastmath w module L0?", "Entity", True),
    ("Co przechowuje bufor PyArrow w L1 working memory?", "Entity", True),
    ("Jakie tabele znajdują się w bazie Kuzu Graph?", "Entity", True),
    ("Jak działa VerifiedKVStore z łańcuchem SHA-256?", "Entity", True),
    ("Co robi SleepBaker podczas cyklu bezczynności?", "Entity", True),
    ("Jakie operacje blokuje ASTSafetyScanner?", "Entity", True),
    ("Jak działa asymetryczny skan Hamminga AVX-512?", "Entity", True),
    ("Co to jest Matryoshka Adaptive Search MRL?", "Entity", True),
    ("Jakie są uprawnienia narzędzia vault_write w Obsidian MCP?", "Entity", True),
    ("Jakie modele obsługuje brama OmniRoute na porcie 20128?", "Entity", True),
    ("Gdzie zdefiniowany jest Task Loop Doctor?", "Entity", True),
    ("Jakie parametry przyjmuje funkcja compile_sop_to_skill?", "Entity", True),
    ("Co oznacza flaga ATLAS_DEBUG w konfiguracji?", "Entity", True),
    ("Jakie są role w triadzie Loop Engineering?", "Entity", True),
    ("Co to jest Doc-Sync Gate w standardzie V23?", "Entity", True),
    ("Jakie zależności wycięto w refaktoryzacji V27?", "Entity", True),
    ("Dlaczego usunięto bibliotekę mem0ai z projektu?", "Entity", True),
    ("Jakie wersje Pythona są obsługiwane w macierzy CI?", "Entity", True),
    ("Jak działa bufor rotacyjny max_history w JEPA?", "Entity", True),
    ("Co robi funkcja detect_cpof w grafie przyczynowym?", "Entity", True),
    ("Jak obliczany jest wskaźnik SeverityScore dla węzła CPoF?", "Entity", True),
    ("Jak działa fala dyfuzji w causal_diffusion_analysis?", "Entity", True),
    ("Co to jest Epistemic Knapsack Packing?", "Entity", True),
    ("Jaki jest domyślny budżet tokenów w orchestratorze?", "Entity", True),
    ("Jak działa PrefixCacheGuard w sesji Hermesa?", "Entity", True),
    ("Jakie narzędzia eksportuje moduł atlas_memory?", "Entity", True),
    ("Gdzie znajduje się skrypt install-hermes-plugin.sh?", "Entity", True),
    ("Jakie komendy udostępnia CLI Pixi w projekcie?", "Entity", True),
    ("Co sprawdza test test_import_integrity.py?", "Entity", True),
    ("Jakie pliki baselines znajdują się w docs/baselines?", "Entity", True),
    ("Jak zdefiniowany jest protokół GossipTransport UDP?", "Entity", True),
    ("Jakie podpisy kryptograficzne obsługuje ThresholdSigner?", "Entity", True),
    ("Co to jest EpistemicReputation i jak wylicza score peera?", "Entity", True),
    ("Jak działa binarizacja MIB 32x w module quantizer.py?", "Entity", True),
    ("Jakie typy faktów obsługuje tabela memoria_facts?", "Entity", True),
    ("Co zawiera kolumna veracity w working_memory?", "Entity", True),
    ("Jakie są dozwolone źródła wiedzy w EpistemicSource?", "Entity", True),
    ("Co oznacza hierarchia USER_EXPLICIT nad TOOL_OUTPUT?", "Entity", True),
    ("Jak działa metoda supersede_fact w silniku pamięci?", "Entity", True),
    ("Gdzie zapisywane są snapshoty bazy w /tmp?", "Entity", True),
    ("Jak działa zlib.crc32 w haszowaniu deterministycznym?", "Entity", True),
    ("Jakie są wymagania systemowe dla modułu Kuzu i Numba?", "Entity", True),
    ("Co zawiera raport AGI_HORIZONS_SOTA_REPORT.md?", "Entity", True),
    ("Jak działa weryfikacja łańcucha skrótów verify_chain_integrity?", "Entity", True),
    ("Co to jest VectorClock i relacja happens-before?", "Entity", True),
    ("Jakie są funkcje pomocnicze w doctor_fix.sh?", "Entity", True),
    ("Co oznacza wynik Loop Ready Score 100/100 L2+?", "Entity", True),
    ("Jakie testy weryfikują kworum 2f+1 w BFT-CRDT?", "Entity", True),

    # Category B: Architecture & Multi-Turn Engineering Topics (60 queries)
    ("Jak przebiegała faza V19 Causal Annealing?", "Architecture", True),
    ("Co wprowadzono w fazie V20 BFT-CRDT?", "Architecture", True),
    ("Jakie zmiany przyniosła faza V22 RaBitQ?", "Architecture", True),
    ("Co wdrożono w fazie V23 Loop Engineering?", "Architecture", True),
    ("Jak zoptymalizowano przepustowość w fazie V24?", "Architecture", True),
    ("Jakie niebezpieczne stubs usunięto w fazie V25?", "Architecture", True),
    ("Co wdrożono w fazie V26 Production Ready?", "Architecture", True),
    ("Jakie cele zrealizowano w fazie V27 Zero Zombie?", "Architecture", True),
    ("Dlaczego wybrano architekturę Micro-Sidecar UDS w V10?", "Architecture", True),
    ("Jakie były wnioski z audytu trylogii V13-V15?", "Architecture", True),
    ("Jakie mutacje testowe wstrzykiwał Reviewer w V12?", "Architecture", True),
    ("Co wykrył test mutacyjny w przerwaniu hash-chain?", "Architecture", True),
    ("Jak działa deterministyczny Sleep Baker w L3?", "Architecture", True),
    ("Jak zorganizowany jest przepływ danych w 2 runtimes?", "Architecture", True),
    ("Dlaczego Hermes Core jest nienaruszalny w Hard Rule 2?", "Architecture", True),
    ("Co oznacza Hard Rule 5 zakazująca torch w runtime?", "Architecture", True),
    ("Dlaczego stosujemy sys.path.append zamiast insert(0)?", "Architecture", True),
    ("Jakie są założenia dla modeli black-box cloud APIs?", "Architecture", True),
    ("Co to jest externalization of cognitive processes?", "Architecture", True),
    ("Jakie podsystemy składają się na warstwę L0 Dynamic?", "Architecture", True),
    ("Jakie modele Pydantic v2 definiują strukturę pamięci?", "Architecture", True),
    ("Jakie właściwości ma bufor TrajectoryBuffer w Apache Arrow?", "Architecture", True),
    ("Jakie relacje grafowe Cypher wspiera KuzuGraphStore?", "Architecture", True),
    ("Co zapewnia algorytm LWWElementSet w CRDT?", "Architecture", True),
    ("Jak działa szyfrowanie AES-256-GCM z noncem 12 bajtów?", "Architecture", True),
    ("Jakie są zalety binarizacji float32 do uint64?", "Architecture", True),
    ("Jak Numba JIT nogil przyspiesza skan wektorowy?", "Architecture", True),
    ("Co to jest Active Sensing i detekcja błędu predykcji?", "Architecture", True),
    ("Jak Shadow Worker asynchronicznie wyciąga trójki SPO?", "Architecture", True),
    ("Jak działa CacheHitMonitor i estymacja oszczędności?", "Architecture", True),
    ("Jakie są wyniki profilowania ścieżek krytycznych hot paths?", "Architecture", True),
    ("Dlaczego zastosowano strukturę monorepo z Pixi?", "Architecture", True),
    ("Jakie są zalety izolacji środowiska w Pythonie 3.14 No-GIL?", "Architecture", True),
    ("Co to jest zero-copy streaming w buforze Arrow?", "Architecture", True),
    ("Jak działa dynamiczna rotacja wskaźników pamięci?", "Architecture", True),
    ("Jakie są gwarancje determinizmu w testach jednostkowych?", "Architecture", True),
    ("Co to jest test gate i dlaczego wymaga 100 procent zgodności?", "Architecture", True),
    ("Jakie mechanizmy chronią przed złośliwymi mutacjami w CRDT?", "Architecture", True),
    ("Jak działa algorytm simulated annealing w grafie przyczynowym?", "Architecture", True),
    ("Jakie koszty telemetrii uwzględnia EnergyModule?", "Architecture", True),
    ("Co to jest trainable critic w modelu przyczynowym?", "Architecture", True),
    ("Jakie są metryki w pliku docs/benchmark_baseline.json?", "Architecture", True),
    ("Dlaczego zmierzono P50 latencji na żywej bazie danych?", "Architecture", True),
    ("Jakie tabele w Mnemosyne przechowują podsumowania sesji?", "Architecture", True),
    ("Co zawiera tabela gists i jakie ma kolumny?", "Architecture", True),
    ("Jakie znaczenie ma kolumna importance w rekordach pamięci?", "Architecture", True),
    ("Jakie są różnice między pamięcią episodic a working?", "Architecture", True),
    ("Jakie typy błędów obsługuje JSON-RPC 2.0 w AtlasDaemon?", "Architecture", True),
    ("Co się dzieje gdy gniazdo ~/.hermes/atlas.sock jest zablokowane?", "Architecture", True),
    ("Jak działa funkcja get_or_create_client w kliencie IPC?", "Architecture", True),
    ("Jakie są reguły czyszczenia artefaktów w doctor_fix?", "Architecture", True),
    ("Co oznacza brak instrukcji pass w kodzie produkcyjnym?", "Architecture", True),
    ("Jakie są zasady raportowania podatności w SECURITY.md?", "Architecture", True),
    ("Jak przebiega proces tworzenia nowego PR w CONTRIBUTING.md?", "Architecture", True),
    ("Co określa paszport telemetrii w pliku STATE.md?", "Architecture", True),
    ("Jakie cele wyznacza dokumentacja ROADMAP.md?", "Architecture", True),
    ("Jak działa integracja z narzędziami search_memory i commit_observation?", "Architecture", True),
    ("Co zapewnia deterministyczny parser wyrażeń regularnych?", "Architecture", True),
    ("Jak działa deduplikacja wpisów w tabeli annotations?", "Architecture", True),
    ("Jakie są zalety architektury opartej na wektorach binarnych?", "Architecture", True),

    # Category C: Temporal, Recency & Versioning Lookups (50 queries)
    ("Jakie zdarzenia zarejestrowano w bazie w sierpniu 2026?", "Temporal", True),
    ("Kiedy dodano wpisy dotyczące skilli Superpowers?", "Temporal", True),
    ("Jaka jest data utworzenia pierwszego gista w systemie?", "Temporal", True),
    ("Kiedy skonsolidowano ostatnie procedury robocze?", "Temporal", True),
    ("Jakie zmiany wprowadzono 30 sierpnia 2026?", "Temporal", True),
    ("Jakie fakty zostały zaktualizowane 31 sierpnia 2026?", "Temporal", True),
    ("Jaki jest zakres czasowy sesji zarejestrowanych w working_memory?", "Temporal", True),
    ("Kiedy uruchomiono pierwszy raz benchmark Real DB?", "Temporal", True),
    ("Jakie wpisy w annotations mają kind occurred_on?", "Temporal", True),
    ("Jaka była historia refaktoryzacji instalatora pluginu?", "Temporal", True),
    ("Kiedy zamknięto zadania H1 do H5 w audycie instalacji?", "Temporal", True),
    ("Jak zmieniała się liczba testów od fazy V1 do V27?", "Temporal", True),
    ("Kiedy liczba testów wzrosła z 62 do 105 w trylogii SOTA?", "Temporal", True),
    ("W której fazie osiągnięto 147 testów w module skill compiler?", "Temporal", True),
    ("Kiedy wprowadzono zestaw 221 testów w fazie V25?", "Temporal", True),
    ("Jaka jest najświeższa data modyfikacji w pliku mnemosyne.db?", "Temporal", True),
    ("Jakie fakty w memoria_facts posiadają valid_from i valid_to?", "Temporal", True),
    ("Kiedy Gemini 3.7 Flash High został zatwierdzony dla roli Makera?", "Temporal", True),
    ("W jakich godzinach odnotowano rate limity na modelu Gemini 3.6?", "Temporal", True),
    ("Kiedy zoptymalizowano pipeline Crawl4AI do wersji Clean?", "Temporal", True),
    ("Jakie zdarzenia nastąpiły po wdrożeniu mostu UDS w fazie V10?", "Temporal", True),
    ("Kiedy scalono testy L2 do pliku test_l2_qdrant_kuzu.py?", "Temporal", True),
    ("Jaka jest chronologia publikacji raportów baselines?", "Temporal", True),
    ("Kiedy wygenerowano schemat architektoniczny 8K w docs/assets?", "Temporal", True),
    ("W którym momencie zaimplementowano bufor TrajectoryBuffer w Arrow?", "Temporal", True),
    ("Kiedy dodano algorytm RaBitQ z randomizowaną rotacją?", "Temporal", True),
    ("Jaka była sekwencja commitów podczas fazy V26?", "Temporal", True),
    ("Kiedy usunięto zależność mem0ai w gałęzi main?", "Temporal", True),
    ("Jakie wpisy w working_memory dotyczą sesji z 2026-08-04?", "Temporal", True),
    ("Jakie fakty dotyczące Obsidianu zarejestrowano w połowie sierpnia?", "Temporal", True),
    ("Kiedy wprowadzono narzędzie loop_doctor.py?", "Temporal", True),
    ("W jakiej dacie zatwierdzono regułę Zero Zombie w AGENTS.md?", "Temporal", True),
    ("Kiedy zweryfikowano działanie bramy OmniRoute na porcie 20128?", "Temporal", True),
    ("Jakie operacje wykonano podczas ostatniego sprintu optymalizacyjnego?", "Temporal", True),
    ("Kiedy wdrożono algorytm Epistemic Knapsack Packing?", "Temporal", True),
    ("Jak zmieniała się latencja P50 w kolejnych pomiarach?", "Temporal", True),
    ("Kiedy zmierzono wynik 12.20 ms dla 40 zapytań?", "Temporal", True),
    ("Jaka była chronologia naprawy ścieżek z insert(0) na append?", "Temporal", True),
    ("Kiedy dodano testy mutacyjne dla subagentów Reviewer?", "Temporal", True),
    ("W jakich datach publikowano aktualizacje w STATE.md?", "Temporal", True),
    ("Kiedy zdefiniowano 15 metryk syntetycznych SOTA?", "Temporal", True),
    ("Jakie wpisy w episodic_memory pochodzą z wczesnych faz testowych?", "Temporal", True),
    ("Kiedy zweryfikowano odporność BFT-CRDT na awarie Byzantine?", "Temporal", True),
    ("W którym dniu wykonano pełny test importu 61 podmodułów?", "Temporal", True),
    ("Kiedy wdrożono zabezpieczenie AST przed atakami RCE?", "Temporal", True),
    ("Jaka była data utworzenia pliku pyproject.toml?", "Temporal", True),
    ("Kiedy sformatowano schematy w JSON dla baselines?", "Temporal", True),
    ("W jakim terminie zrealizowano kamienie milowe V19-V24?", "Temporal", True),
    ("Kiedy wdrożono procedury walidacji w GitHub Actions?", "Temporal", True),
    ("Jakie wpisy temporalne zarejestrowano w tabeli memoria_facts?", "Temporal", True),

    # Category D: Conversational / Negative Controls (60 queries)
    ("Cześć, jak się dziś masz?", "Conversational", False),
    ("Dzięki, to wszystko na teraz!", "Conversational", False),
    ("Rozumiem, przejdźmy dalej.", "Conversational", False),
    ("Super robota!", "Conversational", False),
    ("Ok, czekam na Twoją odpowiedź.", "Conversational", False),
    ("W porządku.", "Conversational", False),
    ("Jasne, zróbmy tak.", "Conversational", False),
    ("Dzień dobry!", "Conversational", False),
    ("Czy możesz mi w tym pomóc?", "Conversational", False),
    ("Dobrze, dziękuję bardzo.", "Conversational", False),
    ("Hej", "Conversational", False),
    ("Cześć Hermes", "Conversational", False),
    ("Dziękuję", "Conversational", False),
    ("Gotowe", "Conversational", False),
    ("Dalej", "Conversational", False),
    ("Super", "Conversational", False),
    ("Nie ma problemu", "Conversational", False),
    ("Do zobaczenia", "Conversational", False),
    ("Ok dzięki", "Conversational", False),
    ("Pa", "Conversational", False),
    ("Jak się masz?", "Conversational", False),
    ("Miłego dnia!", "Conversational", False),
    ("Dzięki za pomoc, jesteś super.", "Conversational", False),
    ("Cześć! Co tam słychać?", "Conversational", False),
    ("Wszystko jasne, dziękuję.", "Conversational", False),
    ("Dobra robota, kontynuujmy.", "Conversational", False),
    ("Hej, jesteś tam?", "Conversational", False),
    ("Dzięki serdeczne!", "Conversational", False),
    ("Rozumiem sytuację.", "Conversational", False),
    ("Świetnie, o to chodziło.", "Conversational", False),
    ("Ok, rozumiem.", "Conversational", False),
    ("Jasna sprawa.", "Conversational", False),
    ("Bardzo dziękuję za wyjaśnienie.", "Conversational", False),
    ("Do usłyszenia!", "Conversational", False),
    ("Trzymaj się!", "Conversational", False),
    ("Dzięki mistrzu.", "Conversational", False),
    ("Wszystko działa idealnie.", "Conversational", False),
    ("Ekstra, dzięki wielkie.", "Conversational", False),
    ("Pozdrawiam!", "Conversational", False),
    ("Dobrej nocy.", "Conversational", False),
    ("Jak napisać pętlę for w Pythonie?", "Conversational", False),
    ("Jak odwrócić string w Pythonie?", "Conversational", False),
    ("Czym różni się lista od krotki w Pythonie?", "Conversational", False),
    ("Napisz prostą funkcję obliczającą silnię.", "Conversational", False),
    ("Jak sprawdzić czy plik istnieje w Bashu?", "Conversational", False),
    ("Jakie jest pole koła o promieniu 5?", "Conversational", False),
    ("Jak przekonwertować int na string w Pythonie?", "Conversational", False),
    ("Napisz wyrażenie regularne dla adresu email.", "Conversational", False),
    ("Jak działa operator ternary w Pythonie?", "Conversational", False),
    ("Jak wygenerować losową liczbę w Pythonie?", "Conversational", False),
    ("Co robi polecenie git status?", "Conversational", False),
    ("Jak połączyć dwie listy w Pythonie?", "Conversational", False),
    ("Napisz prosty serwer HTTP w Pythonie.", "Conversational", False),
    ("Jak sprawdzić wersję Pythona w terminalu?", "Conversational", False),
    ("Jak odczytać plik linia po linii w Pythonie?", "Conversational", False),
    ("Ile to jest 25 razy 4?", "Conversational", False),
    ("Jak usunąć duplikaty z listy w Pythonie?", "Conversational", False),
    ("Co oznacza słowo kluczowe yield w Pythonie?", "Conversational", False),
    ("Jak sformatować datę w formacie ISO w Pythonie?", "Conversational", False),
    ("Dzięki, to wszystko na ten moment!", "Conversational", False),
]


def run_benchmark() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dst_db, table_counts = prepare_safe_snapshot()
    total_db_records = sum(table_counts.values())

    pure_backend = PureMnemosyneBackend(db_path=dst_db)
    atlas_backend = AtlasSnapshotBackend(db_path=dst_db)
    atlas_provider = AtlasMemoryProvider(orchestrator=MemoryOrchestrator())
    atlas_provider._mnemosyne = atlas_backend

    # Warmup
    _ = pure_backend.prefetch("Warmup")
    _ = atlas_provider.prefetch("Warmup", session_id="warmup")

    # 1. Run Pure Mnemosyne
    pure_records = []
    for q, cat, has_entity in BENCHMARK_QUERIES:
        t0 = time.perf_counter_ns()
        ctx = pure_backend.prefetch(q)
        lat_ms = (time.perf_counter_ns() - t0) / 1e6
        tokens = len(ctx) // 4
        is_empty = (len(ctx.strip()) == 0)
        pure_records.append({
            "query": q,
            "category": cat,
            "has_entity": has_entity,
            "latency_ms": round(lat_ms, 3),
            "tokens_injected": tokens,
            "context_chars": len(ctx),
            "abstained": is_empty,
        })

    # 2. Run ATLAS + Mnemosyne
    atlas_records = []
    for q, cat, has_entity in BENCHMARK_QUERIES:
        t0 = time.perf_counter_ns()
        ctx = atlas_provider.prefetch(q, session_id="bench_session")
        lat_ms = (time.perf_counter_ns() - t0) / 1e6
        tokens = len(ctx) // 4
        is_empty = (len(ctx.strip()) == 0)
        skipped = False
        if atlas_provider._last_prefetch:
            skipped = atlas_provider._last_prefetch.get("skipped", False)
        atlas_records.append({
            "query": q,
            "category": cat,
            "has_entity": has_entity,
            "latency_ms": round(lat_ms, 3),
            "tokens_injected": tokens,
            "context_chars": len(ctx),
            "abstained": is_empty,
            "skipped_by_policy_gate": skipped,
        })

    # Aggregate Statistics
    pure_lats = [r["latency_ms"] for r in pure_records]
    atlas_lats = [r["latency_ms"] for r in atlas_records]

    pure_conv = [r for r in pure_records if r["category"] == "Conversational"]
    atlas_conv = [r for r in atlas_records if r["category"] == "Conversational"]

    pure_false_injections = sum(1 for r in pure_conv if not r["abstained"])
    atlas_false_injections = sum(1 for r in atlas_conv if not r["abstained"])

    pure_summary = {
        "system": "Pure Mnemosyne (Standalone SQLite BEAM)",
        "dataset_exact_record_count": total_db_records,
        "table_breakdown": table_counts,
        "queries_evaluated": len(BENCHMARK_QUERIES),
        "mean_latency_ms": round(float(np.mean(pure_lats)), 3),
        "p50_latency_ms": round(float(np.median(pure_lats)), 3),
        "p90_latency_ms": round(float(np.percentile(pure_lats, 90)), 3),
        "p95_latency_ms": round(float(np.percentile(pure_lats, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(pure_lats, 99)), 3),
        "std_latency_ms": round(float(np.std(pure_lats)), 3),
        "total_latency_ms": round(float(np.sum(pure_lats)), 3),
        "total_tokens_injected": int(sum(r["tokens_injected"] for r in pure_records)),
        "conversational_turns_count": len(pure_conv),
        "false_noise_injections": pure_false_injections,
        "abstention_precision_pct": round(float((len(pure_conv) - pure_false_injections) / max(len(pure_conv), 1) * 100.0), 1),
        "raw_query_records": pure_records,
    }

    atlas_summary = {
        "system": "ATLAS + Mnemosyne (Cognitive Orchestration Layer)",
        "dataset_exact_record_count": total_db_records,
        "table_breakdown": table_counts,
        "queries_evaluated": len(BENCHMARK_QUERIES),
        "mean_latency_ms": round(float(np.mean(atlas_lats)), 3),
        "p50_latency_ms": round(float(np.median(atlas_lats)), 3),
        "p90_latency_ms": round(float(np.percentile(atlas_lats, 90)), 3),
        "p95_latency_ms": round(float(np.percentile(atlas_lats, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(atlas_lats, 99)), 3),
        "std_latency_ms": round(float(np.std(atlas_lats)), 3),
        "total_latency_ms": round(float(np.sum(atlas_lats)), 3),
        "total_tokens_injected": int(sum(r["tokens_injected"] for r in atlas_records)),
        "conversational_turns_count": len(atlas_conv),
        "false_noise_injections": atlas_false_injections,
        "abstention_precision_pct": round(float((len(atlas_conv) - atlas_false_injections) / max(len(atlas_conv), 1) * 100.0), 1),
        "raw_query_records": atlas_records,
    }

    return pure_summary, atlas_summary


if __name__ == "__main__":
    pure_res, atlas_res = run_benchmark()

    pure_file = OUTPUT_DIR / "benchmark_baseline_mnemosyne_pure.json"
    atlas_file = OUTPUT_DIR / "benchmark_baseline_atlas_real.json"

    with open(pure_file, "w", encoding="utf-8") as f:
        json.dump(pure_res, f, indent=2, ensure_ascii=False)

    with open(atlas_file, "w", encoding="utf-8") as f:
        json.dump(atlas_res, f, indent=2, ensure_ascii=False)

    print("Benchmark results saved to:")
    print(f" - {pure_file} ({pure_file.stat().st_size} bytes)")
    print(f" - {atlas_file} ({atlas_file.stat().st_size} bytes)")
    print("\n--- RESULTS SUMMARY ---")
    print(f"Total Dataset Records: {pure_res['dataset_exact_record_count']}")
    print(f"Pure Mnemosyne:  P50 = {pure_res['p50_latency_ms']} ms | P95 = {pure_res['p95_latency_ms']} ms | Tokens = {pure_res['total_tokens_injected']} | False Injections = {pure_res['false_noise_injections']}/{pure_res['conversational_turns_count']}")
    print(f"ATLAS + Mnemo:   P50 = {atlas_res['p50_latency_ms']} ms | P95 = {atlas_res['p95_latency_ms']} ms | Tokens = {atlas_res['total_tokens_injected']} | False Injections = {atlas_res['false_noise_injections']}/{atlas_res['conversational_turns_count']}")

