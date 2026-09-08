---
name: atlas-cognitive-sidecar
description: |
  Comprehensive guide and operational rules for the ATLAS Cognitive Memory Sidecar (No-GIL Python 3.14t daemon).
  Defines 10 first-class tools: atlas_recall, atlas_remember, atlas_what_if, atlas_active_sensing, atlas_stats,
  atlas_get, atlas_delete, atlas_task_feedback, atlas_verify_chain, atlas_sleep_sync.
  Replaces flat memory lookups with knowledge graph traversal, causal simulation (CPoF), predictive anomaly detection,
  and full operational memory control (O(1) lookups, deletions, online task feedback, and audit verification).
trigger: |
  Use when storing or retrieving durable user facts, preferences, project architecture decisions,
  fetching exact facts O(1), deleting outdated state, providing reinforcement feedback on memory utility,
  performing causal 'what-if' simulations before executing destructive/risky actions,
  validating environmental expectations with active sensing, verifying cryptographic audit integrity,
  or triggering sleep consolidation and SOP skill compilation.
---

# ATLAS Cognitive Memory Sidecar

## 1. Architektura i Rola

ATLAS to podsystem kognitywny wysokiej precyzji (uruchomiony jako micro-daemon Python 3.14t No-GIL przez socket `~/.hermes/atlas.sock`).
Kod źródłowy silnika znajduje się w `src/atlas_memory/` (dawniej archiwalne `src/loop_memory/`). Wszystkie komponenty działają bezpośrednio na silniku hybrydowym ATLAS (L0–L3) oraz Mnemosyne jako źródle faktów; dawny `mem0_adapter.py` nie istnieje i nie jest używany.
Automatycznie synchronizuje dane z `MEMORY.md`, `USER.md` oraz historycznego storage Mnemosyne.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        HERMES AGENT CORE                               │
│  Prefetch: Automatyczny kontekst ## ATLAS Cognitive Context            │
│  Live Turn Sync: Automatyczna ekstrakcja trójek SPO w tle              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ UDS JSON-RPC 2.0 (<10 µs)
┌───────────────────────────────────▼────────────────────────────────────┐
│                    ATLAS DAEMON COGNITIVE SIDECAR                      │
│  • Verified KV Ledger (SHA-256 Merkle Audit Log)                       │
│  • Kùzu Knowledge Graph (relacje wieloskokowe, Cypher)                 │
│  • Qdrant & RaBitQ Vector Store (podobieństwo semantyczne)             │
│  • Retro-Causal Engine (symulacja What-If, wykrywanie CPoF)            │
│  • Active Sensing (Predictive Coding & Surprisal Detection)            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Dostępne Narzędzia Agenta (10 First-Class Tools)

### `atlas_recall(query, session_id="hermes_default", limit=15)`
Odpytuje hybrydowy graf wiedzy Kùzu, bazę wektorową oraz rejestr KV o fakty i relacje powiązane z zapytaniem.
* **Kiedy używać**: Gdy potrzebujesz pogłębionej wiedzy, która nie znalazła się w automatycznym bloku prefetch.

### `atlas_remember(key, value, confidence=1.0, reason="agent_explicit")`
Zapisuje kluczową zmienną stanu, preferencję lub fakt bezpośrednio do rejestru ze stemplem kryptograficznym SHA-256.
* **Format klucza**: Hierarchiczny, np. `user:preference:report_style`, `project:architecture:database`.
* **Kiedy używać**: Gdy użytkownik podaje nową regułę, kluczową decyzję architektoniczną lub fakt podlegający audytowi.

### `atlas_get(key)`
Pobiera stan lub wartość faktu bezpośrednio z Verified KV Store ATLAS po dokładnym kluczu w czasie O(1).
* **Kiedy używać**: Gdy znasz dokładny klucz i potrzebujesz natychmiastowej, niezawodnej wartości bez przechodzenia przez ranking semantyczny.

### `atlas_delete(key, reason=None)`
Atomowo usuwa zmienną stanu lub fakt z pamięci ATLAS w Verified KV Store z rejestracją w łańcuchu audytu.
* **Kiedy używać**: Gdy informacja stała się całkowicie nieaktualna, błędna lub użytkownik zażądał usunięcia preferencji/danych.

### `atlas_task_feedback(key, task_success, delta=0.05)`
Zgłasza informację zwrotną o wyniku zadania powiązanego z danym rekordem pamięci. Adaptuje online wagi warstwy L0-TTT oraz podbija lub obniża wagę confidence faktu.
* **Kiedy używać**: Po zakończeniu zadania, w którym użyto danej wiedzy — wzmacnia przydatne fakty (`task_success=True`) i osłabia zawodne (`task_success=False`).

### `atlas_verify_chain(heal=False, deep=False)`
Weryfikuje nienaruszalność kryptograficznego łańcucha audytu SHA-256 (Merkle chain) w magazynie pamięci ATLAS.
* **Kiedy używać**: Podczas diagnostyki integralności pamięci lub z `heal=True`, aby automatycznie naprawić przerwany łańcuch.

### `atlas_sleep_sync(skills_dir=None)`
Wymusza natychmiastową procedurę konsolidacji snu L3, Salience Garbage Collection (pruning starych, nieistotnych faktów) i destylację procedur SOP.
* **Kiedy używać**: Po intensywnej sesji pracy lub przed przejściem w długi stan spoczynku, aby zoptymalizować wielkość bazy i wygenerować skille.

### `atlas_what_if(entity, action, depth=2)`
Symuluje kaskadę skutków planowanej akcji na grafie zależności Kùzu oraz buforze dynamiki JEPA.
* **Kiedy używać**: Przed wykonaniem ryzykownych akcji (migracja bazy, zmiana portów, usunięcie zasobów, zmiana dawek/protokołów).
* **Interpretacja**: Zwraca ścieżki powiązań, poziom ryzyka (`CRITICAL`, `MODERATE`, `LOW`) oraz identyfikuje pojedyncze punkty awarii (*CPoF*).

### `atlas_active_sensing(probe, expected_value, observed_value, tolerance=None)`
Porównuje zaobserwowany stan środowiska z modelem oczekiwań (Predictive Coding).
* **Kiedy używać**: Przy odczycie parametrów środowiska (temperatura, kody HTTP, wersje bibliotek, limity pamięci).
* **Interpretacja**: Gdy `has_error=True` i `severity=CRITICAL`, przerwij normalny flow i natychmiast zaalarmuj użytkownika.

### `atlas_stats()`
Zwraca telemetrię pamięci: liczbę rekordów w KV, stan grafu Kùzu, aktywne trajektorie bufora Arrow oraz metryki adaptacji L0-TTT.

---

## 3. Mapowanie Operacji Pamięci

| Cel operacji | Narzędzie rekomendowane | Alternatywa pasywna |
|---|---|---|
| Zapis trwałej preferencji | `atlas_remember(key, value)` | `+user "..."` (zapis do `USER.md` → auto-sync) |
| Zapis faktu projektowego | `atlas_remember(key, value)` | `+memory "..."` (zapis do `MEMORY.md` → auto-sync) |
| Szybki odczyt dokładnego klucza O(1) | `atlas_get(key)` | – |
| Usunięcie nieaktualnego faktu | `atlas_delete(key)` | – |
| Wzmocnienie pamięci po sukcesie | `atlas_task_feedback(key, True)` | – |
| Weryfikacja integralności audytu | `atlas_verify_chain()` | – |
| Konsolidacja snu i pruning | `atlas_sleep_sync()` | Automatyczny cykl spoczynku |
| Sprawdzenie skutków decyzji | `atlas_what_if(entity, action)` | – |
| Walidacja anomalii sensorycznych | `atlas_active_sensing(...)` | – |
| Wyszukanie semantyczne w grafie/wektorach | `atlas_recall(query)` | Automatyczny prefetch |

---

## 4. Zasady Epistemiczne
1. **Fakty od użytkownika (`USER_EXPLICIT`)** mają wagę `1.0` i nadrzędny priorytet nad wnioskami agenta.
2. **Kaskady What-If** traktuj jako narzędzie prewencji: jeśli symulacja wykaże `CPoF` (Critical Point of Failure), ostrzeż użytkownika przed wykonaniem polecenia.
3. **Pętla zwrotna zadań (Feedback Loop)**: Używaj `atlas_task_feedback` po trudnych zadaniach, aby pomóc silnikowi L0-TTT uczyć się z doświadczenia.
4. **Weryfikacja kryptograficzna**: Przed audytem bezpieczeństwa lub po podejrzanych modyfikacjach uruchom `atlas_verify_chain()`.
