---
name: atlas-cognitive-sidecar
description: |
  Comprehensive guide and operational rules for the ATLAS Cognitive Memory Sidecar (No-GIL Python 3.14t daemon).
  Defines first-class tools: atlas_recall, atlas_remember, atlas_what_if, atlas_active_sensing, atlas_stats.
  Replaces flat memory lookups with knowledge graph traversal, causal simulation (CPoF), and predictive anomaly detection.
trigger: |
  Use when storing or retrieving durable user facts, preferences, project architecture decisions,
  performing causal 'what-if' simulations before executing destructive/risky actions,
  or validating environmental expectations with active sensing.
---

# ATLAS Cognitive Memory Sidecar

## 1. Architektura i Rola

ATLAS to podsystem kognitywny wysokiej precyzji (uruchomiony jako micro-daemon Python 3.14t No-GIL przez socket `~/.hermes/atlas.sock`).
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

## 2. Dostępne Narzędzia Agenta (First-Class Tools)

### `atlas_recall(query, session_id="hermes_default")`
Odpytuje hybrydowy graf wiedzy Kùzu, bazę wektorową oraz rejestr KV o fakty i relacje powiązane z zapytaniem.
* **Kiedy używać**: Gdy potrzebujesz pogłębionej wiedzy, która nie znalazła się w automatycznym bloku prefetch.

### `atlas_remember(key, value, confidence=1.0, reason="agent_explicit")`
Zapisuje kluczową zmienną stanu, preferencję lub fakt bezpośrednio do rejestru ze stemplem kryptograficznym SHA-256.
* **Format klucza**: Hierarchiczny, np. `user:preference:report_style`, `project:architecture:database`.
* **Kiedy używać**: Gdy użytkownik podaje nową regułę, kluczową decyzję architektoniczną lub fakt podlegający audytowi.

### `atlas_what_if(entity, action, depth=2)`
Symuluje kaskadę skutków planowanej akcji na grafie zależności Kùzu oraz buforze dynamiki JEPA.
* **Kiedy używać**: Przed wykonaniem ryzykownych akcji (migracja bazy, zmiana portów, usunięcie zasobów, zmiana dawek/protokołów).
* **Interpretacja**: Zwraca ścieżki powiązań, poziom ryzyka (`CRITICAL`, `MODERATE`, `LOW`) oraz identyfikuje pojedyncze punkty awarii (*CPoF*).

### `atlas_active_sensing(probe, expected_value, observed_value)`
Porównuje zaobserwowany stan środowiska z modelem oczekiwań (Predictive Coding).
* **Kiedy używać**: Przy odczycie parametrów środowiska (temperatura, kody HTTP, wersje bibliotek, limity pamięci).
* **Interpretacja**: Gdy `has_error=True` i `severity=CRITICAL`, przerwij normalny flow i natychmiast zaalarmuj użytkownika.

### `atlas_stats()`
Zwraca telemetrię pamięci: liczbę rekordów w KV, stan grafu Kùzu i oszczędności tokenów.

---

## 3. Mapowanie Operacji Pamięci

| Cel operacji | Narzędzie rekomendowane | Alternatywa pasywna |
|---|---|---|
| Zapis trwałej preferencji | `atlas_remember(key, value)` | `+user "..."` (zapis do `USER.md` → auto-sync) |
| Zapis faktu projektowego | `atlas_remember(key, value)` | `+memory "..."` (zapis do `MEMORY.md` → auto-sync) |
| Sprawdzenie skutków decyzji | `atlas_what_if(entity, action)` | – |
| Walidacja anomalii | `atlas_active_sensing(...)` | – |
| Wyszukanie w grafie/wektorach | `atlas_recall(query)` | Automatyczny prefetch |

---

## 4. Zasady Epistemiczne
1. **Fakty od użytkownika (`USER_EXPLICIT`)** mają wagę `1.0` i nadrzędny priorytet nad wnioskami agenta.
2. **Kaskady What-If** traktuj jako narzędzie prewencji: jeśli symulacja wykaże `CPoF` (Critical Point of Failure), ostrzeż użytkownika przed wykonaniem polecenia.
3. **Pamiętaj o podpisie**: Jeśli w pamięci znajduje się aktywna reguła weryfikacyjna, stosuj wymagany format podsumowań.
