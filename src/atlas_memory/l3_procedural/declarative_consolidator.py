
"""V66 LUKA 1 (audyt §3B): deklaratywna kaskada konsolidacji snu.

Offline'owa (cykl snu) kaskada porządkująca PAMIĘĆ DEKLARATYWNĄ:

  L0  pary kandydatów o podobieństwie kosinusowym >= progu (domyślnie 0.86),
      liczonym w TEJ SAMEJ przestrzeni wektorowej co fakty (encoder istniejącego
      indeksu wektorowego) — tylko rekordy aktywne, nie-superseeded;
  L1  grupa klaster = składowe spójne (union-find) na przefiltrowanych parach;
  L2  Entity-Root Guard — supersession DOZWOLONY wyłącznie w obrębie tego samego
      korzenia encji (semantyka BUG-004 z ``cognitive.conflict_arbiter``); obcy
      korzeń = BEZ łączenia (krawędź odrzucana, rekord nietknięty).

Kanonizacja: nowszy / z wyższym confidence = kanonik. Rekord starszy NIE JEST
usuwany — dostaje ``metadata.is_superseded = True`` przez chained ``kv.set_state``
(ten sam łańcuch audytu SHA-256, reason='declarative_supersession') oraz krawędź
``(old)-[:SUPERSEDED_BY]->(new)`` w Kùzu (wzorzec ``cognitive.write_path``).

Salience GC: kasowane są WYŁĄCZNIE rekordy tymczasowe bez provenance o
confidence < progu GC (domyślnie 0.2). NIGDY reguły, preferencje, ograniczenia,
fakty użytkownika ani chronione prefiksy (``security.delete_guard``).

Próg podobieństwa jest odseparowany od bramy overlap-density 0.72 z
ConflictArbitera (rewizja przekonań online) — zmiana jednego nie może
przypadkiem przesunąć drugiego.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from pydantic import BaseModel, Field

from atlas_memory.cognitive.conflict_arbiter import (
    ConflictArbiter,
    EpistemicCoordinate,
    EpistemicDecisionKernel,
    entity_root,
)
from atlas_memory.cognitive.lexicon import FUNCTIONAL_SLOTS
from atlas_memory.core.linguistics import is_discourse_garbage_subject
from atlas_memory.l2_semantic.audit_ledger import record_swallow
from atlas_memory.l2_semantic.qdrant_store import normalize_upsert_key
from atlas_memory.security.delete_guard import DELETE_PROTECTED_PREFIXES

logger = logging.getLogger("atlas.l2.declarative_consolidator")

# ── Bramy prógowe V66 (oddzielne od online-owych bram ConflictArbitera) ──────
DEFAULT_COSINE_THRESHOLD: float = 0.86
DEFAULT_GC_CONF_FLOOR: float = 0.2
DEFAULT_FACT_BATCH_CAP: int = 500

SUPERSESSION_REASON: str = "declarative_supersession"
SALIENCE_GC_REASON: str = "declarative_salience_gc"

# Klucze sterowania/systemu — nigdy nie uczestniczą w kaskadzie deklaratywnej.
CONTROL_PREFIXES: Tuple[str, ...] = (
    "atlas:", "sop:", "session_", "txn:", "transaction", "intent:", "hot:",
    "probe:", "watermark", "checkpoint",
)

# Salience GC: prefiksy absolutnie nietykalne (deletion).
GC_PROTECTED_PREFIXES: Tuple[str, ...] = tuple(DELETE_PROTECTED_PREFIXES) + (
    "rule:", "preference:", "constraint:", "user:", "sop:", "atlas:",
    "hot:", "probe:", "watermark", "checkpoint",
)

# Typy źródła, które oznaczają "rekord ma provenance" (nie jest śmieciem inferencyjnym).
PROVENANCE_SOURCE_TYPES: Tuple[str, ...] = (
    "user_explicit", "tool_output", "tool_observation", "external_doc",
)

# V151 (Wektor I, H3): reason tombstone'owania śmieci dyskursywnych w stanie.
GARBAGE_TOMBSTONE_REASON: str = "discourse_garbage_tombstone"


def are_orthogonal_slots(coord_a: EpistemicCoordinate, coord_b: EpistemicCoordinate) -> bool:
    """Sprawdza czy dwa koordynaty w tej samej encji należą do ortogonalnych slotów funkcjonalnych EAV."""
    slot_a = coord_a.slot.lower()
    slot_b = coord_b.slot.lower()
    if slot_a == slot_b:
        return False
    known_functional = FUNCTIONAL_SLOTS | {
        "credentials", "auth", "token", "password", "username", "port",
        "timeout", "url", "host", "ip", "path", "config", "database", "db",
    }
    if slot_a in known_functional or slot_b in known_functional:
        return True
    if coord_a.is_functional or coord_b.is_functional:
        return True
    return False


class DeclarativeConsolidationStats(BaseModel):
    """Statystyki jednego przejścia kaskady deklaratywnej (raportowalne w RPC snu)."""

    candidates_scanned: int = 0
    pairs_found: int = 0
    clusters: int = 0
    facts_superseded: int = 0
    gc_pruned: int = 0
    # V151 (Wektor I, H3): liczba śmieci dyskursywnych oznaczonych tombstone
    # w cyklu snu (atomiczna metoda superseded z inkrementowanym revision_count).
    garbage_tombstoned: int = 0
    roots_blocked: int = 0
    orthogonal_slots_blocked: int = 0
    edges_wired: int = 0
    skipped_no_vectors: int = 0
    truncated_by_batch_cap: bool = False
    duration_ms: float = 0.0
    notes: List[str] = Field(default_factory=list)


class CandidateFact:
    """Jeden kandydat deklaratywny wyciągnięty ze zweryfikowanego stanu KV."""

    __slots__ = ("key", "value", "text", "confidence", "timestamp", "metadata")

    def __init__(
        self,
        key: str,
        value: Any,
        text: str,
        confidence: float,
        timestamp: float,
        metadata: Dict[str, Any],
    ) -> None:
        self.key = key
        self.value = value
        self.text = text
        self.confidence = confidence
        self.timestamp = timestamp
        self.metadata = metadata

    @property
    def root(self) -> str:
        return entity_root(self.key)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Kosinus dwóch wektorów (0.0 gdy którykolwiek jest zerowy / niezgodny wymiar)."""
    va = [float(x) for x in a]
    vb = [float(x) for x in b]
    if not va or not vb or len(va) != len(vb):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(va, vb, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class DeclarativeConsolidator:
    """
    Kaskada L0 → L1 → L2 + Salience GC dla cyklu snu (V66, audyt §3B).

    Zależności zewnętrzne (KV, Kùzu, indeks wektorowy) są wstrzykiwane; bez
    dostępnej Bramy wektorowej kaskada L0-L2 jest pomijana (Salience GC działa
    nadal), nigdy nie budujemy drugiego indeksu.
    """

    def __init__(
        self,
        kv_store: Any,
        graph_store: Any = None,
        vector_store: Any = None,
        embedder: Optional[Callable[[str], Sequence[float]]] = None,
        cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
        gc_conf_floor: float = DEFAULT_GC_CONF_FLOOR,
        fact_batch_cap: int = DEFAULT_FACT_BATCH_CAP,
    ) -> None:
        self.kv_store = kv_store
        self.graph_store = graph_store
        self.vector_store = vector_store
        self._injected_embedder = embedder
        self.cosine_threshold = float(cosine_threshold)
        self.gc_conf_floor = float(gc_conf_floor)
        self.fact_batch_cap = max(2, int(fact_batch_cap))
        # Licznik krawędzi odrzuconych przez Entity-Root Guard w bieżącym przejściu.
        self._blocked_roots = 0
        self._blocked_orthogonal_slots = 0
        # V56 choke point: WSZYSTKIE zapisy do grafu (SUPERSEDED_BY) idą przez
        # EpistemicWritePath — leniwy import, by nie wiązać pętli importowych.
        from atlas_memory.cognitive.write_path import EpistemicWritePath

        self._write_path = EpistemicWritePath(kv_store=kv_store, graph_store=graph_store)

    # ── Brama wektorowa ──────────────────────────────────────────────────────
    def _resolve_embedder(self) -> Optional[Any]:
        """Wybiera enkoder ISTNIEJĄCEGO indeksu wektorowego (bez drugiej encji)."""
        if self._injected_embedder is not None:
            return self._injected_embedder
        encoder = getattr(self.vector_store, "encoder", None)
        encode = getattr(encoder, "encode", None)
        if callable(encode):
            return encode
        direct = getattr(self.vector_store, "encode", None)
        if callable(direct):
            return direct
        return None

    def _embed(self, text: str, embedder: Any) -> Sequence[float]:
        vec = embedder(text)
        return [float(x) for x in vec]

    # ── Selekcja kandydatów ──────────────────────────────────────────────────
    @staticmethod
    def _is_control_key(key: str) -> bool:
        return key.startswith(CONTROL_PREFIXES)

    @staticmethod
    def _is_superseded(data: Dict[str, Any], meta: Dict[str, Any]) -> bool:
        return bool(data.get("is_superseded") or meta.get("is_superseded") or meta.get("superseded"))

    @classmethod
    def _is_active_candidate(cls, key: str, data: Dict[str, Any], meta: Dict[str, Any]) -> bool:
        """Aktywny, nie-superseeded, deklaratywny rekord z treścią do porównania."""
        if not key or cls._is_control_key(key):
            return False
        if cls._is_superseded(data, meta):
            return False
        text = cls._value_text(data.get("value"))
        return text is not None and len(text) >= 3

    @staticmethod
    def _value_text(value: Any) -> Optional[str]:
        """Normalizuje wartość deklaratywną do tekstu porównawczego.

        V71.1 F2: jawne rozgałęzienie typów (żadnego `str(value)` na ślepo):
          None                 → None (odrzucone)
          dict / list          → fakt strukturalny JSON (json.dumps, nie repr)
          str                  → czysty string (semantyka bez zmian)
          bool/int/float/bytes → None (nie są treścią deklaratywną)
          obiekt obcy (Pydantic itp.) → None, bo str(obj) daje "<X at 0x…>"
        """
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, str):
            return value.strip()
        return None

    def collect_candidates(self, all_states: Dict[str, Any]) -> List[CandidateFact]:
        """Wyciąga kandydatów (aktywnych, nie-superseeded) z limitem partii."""
        candidates: List[CandidateFact] = []
        for key, data in all_states.items():
            if not isinstance(data, dict):
                continue
            meta = data.get("metadata")
            meta = meta if isinstance(meta, dict) else {}
            if not self._is_active_candidate(str(key), data, meta):
                continue
            try:
                confidence = float(data.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            try:
                timestamp = float(data.get("timestamp", 0.0))
            except (TypeError, ValueError):
                timestamp = 0.0
            candidates.append(
                CandidateFact(
                    key=str(key),
                    value=data.get("value"),
                    # V71.1 F2: ten sam kanał normalizacji co brama kandydatów —
                    # fakt strukturalny (dict/list) porównywany jako JSON, nie repr.
                    text=self._value_text(data.get("value")) or "",
                    confidence=confidence,
                    timestamp=timestamp,
                    metadata=meta,
                )
            )
        # Deterministyczna kolejność (po kluczu) + twardy limit partii.
        candidates.sort(key=lambda c: c.key)
        return candidates

    # ── L0: pary kandydatów ──────────────────────────────────────────────────
    def find_pairs(
        self,
        candidates: List[CandidateFact],
        embedder: Any,
    ) -> List[Tuple[str, str, float]]:
        """L0: pary (a, b, cosine) o podobieństwie >= progu, z poszanowaniem Entity-Root Guard i Fiber Bundle EAV.

        V67 (F2, audyt SOTA): pełna macierz cosinusów liczona WEKTORYZOWANIE przez
        numpy (M_norm @ M_norm.T + maska triu) zamiast ręcznej pętli O(n²) z
        czystego Pythona.
        DYREKTYWA 3: Rzutowanie na EpistemicCoordinate — jeśli fakty należą do ortogonalnych
        slotów funkcjonalnych (np. port vs timeout), krawędź jest supresowana (sim=0.0).
        """
        if len(candidates) < 2:
            return []
        raw_vectors = [self._embed(c.text, embedder) for c in candidates]
        try:
            hits = numpy_cosine_pairs(raw_vectors, self.cosine_threshold)
        except (TypeError, ValueError):
            hits = manual_cosine_pairs(raw_vectors, self.cosine_threshold)
        pairs: List[Tuple[str, str, float]] = []
        for i, j, sim in hits:
            a, b = candidates[i], candidates[j]
            # L2 (EpistemicDecisionKernel & Entity-Root Guard, F4 / BUG-004): obcy korzeń lub slot wielowartościowy = BEZ łączenia.
            if not EpistemicDecisionKernel.can_supersede(a.key, b.key):
                self._blocked_roots += 1
                continue
            coord_a = EpistemicCoordinate.from_key(a.key)
            coord_b = EpistemicCoordinate.from_key(b.key)
            if are_orthogonal_slots(coord_a, coord_b):
                self._blocked_orthogonal_slots += 1
                continue
            pairs.append((a.key, b.key, sim))
        return pairs

    # ── L1: składowe spójne ──────────────────────────────────────────────────
    @staticmethod
    def connected_components(
        nodes: Sequence[str],
        pairs: Sequence[Tuple[str, str, float]],
    ) -> List[List[str]]:
        """Fiber Bundle Clustering z weryfikacją spójności geometrycznej (Pairwise Clique / Star Topology).

        Rozbija naiwną przechodniość Union-Find (A ~ B oraz B ~ C nie implikuje A ~ C).
        1. Klastrowanie wyłącznie w obrębie tego samego włókna (EntityRoot, Slot).
        2. Spójność geometryczna: w klastrze każdy węzeł musi posiadać bezpośrednią
           krawędź o wysokim podobieństwie do kandydata kanonicznego (Star Topology)
           oraz spełniać warunek pairwise connectivity (brak długich łańcuchów OOD).
        """
        if not nodes or not pairs:
            return []

        edge_sim: Dict[Tuple[str, str], float] = {}
        adj: Dict[str, Set[str]] = {n: set() for n in nodes}
        for a, b, sim in pairs:
            if a in adj and b in adj:
                k = (min(a, b), max(a, b))
                edge_sim[k] = sim
                adj[a].add(b)
                adj[b].add(a)

        # 1. Rozbicie przestrzeni na włókna (Fiber Bundle) per (entity_root, functional_slot)
        fibers: Dict[Tuple[str, str], List[str]] = {}
        for n in nodes:
            coord = EpistemicCoordinate.from_key(n)
            fiber_slot = coord.slot.lower() if coord.is_functional else "__default_entity_fiber__"
            fiber_key = (coord.entity_root, fiber_slot)
            fibers.setdefault(fiber_key, []).append(n)

        clusters: List[List[str]] = []

        # 2. Klastrowanie geometrycznie spójne wewnątrz każdego włókna
        for _f_key, fiber_nodes in fibers.items():
            if len(fiber_nodes) < 2:
                continue

            fiber_set = set(fiber_nodes)
            visited: Set[str] = set()

            for node in fiber_nodes:
                if node in visited:
                    continue

                comp: List[str] = []
                queue = [node]
                visited.add(node)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for neighbor in adj.get(curr, set()):
                        if neighbor in fiber_set and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                if len(comp) < 2:
                    continue

                if len(comp) == 2:
                    a, b = comp[0], comp[1]
                    if (min(a, b), max(a, b)) in edge_sim:
                        clusters.append(sorted([a, b]))
                    continue

                # >= 3 węzłów: Pairwise Clique Check wokół węzła centralnego
                sorted_by_centrality = sorted(
                    comp,
                    key=lambda u: (
                        len(adj[u] & set(comp)),
                        sum(edge_sim.get((min(u, v), max(u, v)), 0.0) for v in comp if v != u),
                    ),
                    reverse=True,
                )

                assigned_in_comp: Set[str] = set()
                for seed in sorted_by_centrality:
                    if seed in assigned_in_comp:
                        continue
                    current_cluster = [seed]
                    candidates = sorted(
                        list((adj[seed] & set(comp)) - assigned_in_comp),
                        key=lambda v: edge_sim.get((min(seed, v), max(seed, v)), 0.0),
                        reverse=True,
                    )
                    for cand in candidates:
                        if all((min(m, cand), max(m, cand)) in edge_sim for m in current_cluster):
                            current_cluster.append(cand)

                    if len(current_cluster) > 1:
                        assigned_in_comp.update(current_cluster)
                        clusters.append(sorted(current_cluster))

        return clusters

    # ── Kanonizacja + supersession ────────────────────────────────────────────
    @staticmethod
    def pick_canonical(members: List[CandidateFact]) -> CandidateFact:
        """Kanonik = wyższy confidence, a przy remisie — nowszy timestamp."""
        return max(members, key=lambda c: (c.confidence, c.timestamp, c.key))

    async def _tombstone_views(self, key: str, subject: str, ts: float) -> None:
        """V151.1 (F-031/II-1): po zapisie tombstone w KV zsynchronizuj widoki.

        Kontrakt (test_tombstone_spread_marks_views):
          (a) punkt Qdrant usunięty pod kanonicznymi wariantami klucza
              (normalize_upsert_key = funkcja niezależna od prefiksów transportowych),
          (b) węzeł Kùzu (Entity) dostaje właściwość is_superseded
              (NIE DELETE — audyt + krawędzie zostają; read-path filtruje).
        """
        # (a) Qdrant widok: bezwłoczne usunięcie punktu pod każdym kanałem transportowym
        if self.vector_store is not None:
            canon = normalize_upsert_key(key)
            for tgt in (canon, key, f"kv:{canon}"):
                if not tgt:
                    continue
                try:
                    await self.vector_store.delete_by_upsert_key(tgt)
                except Exception as exc:  # widok jest best-effort, nie wywraca cyklu snu
                    logger.debug("[V151.1] qdrant tombstone delete failed for %s: %s", tgt, exc)
                    break
        # (b) Kùzu widok: flaga is_superseded na węźle (właściwość, NIE DELETE — FILAR II)
        if self.graph_store is not None:
            try:
                self.graph_store.mark_entity_superseded(subject, float(ts))
            except Exception as exc:  # widok jest best-effort
                logger.debug("[V151.1] graph tombstone mark failed for %s: %s", subject, exc)

    async def _tombstone_discourse_garbage(self, all_states: Dict[str, Any], stats: "DeclarativeConsolidationStats") -> None:
        """V151 (Wektor I / H3): atomowo oznacza aktywne śmieci dyskursywne jako
        tombstone (is_superseded=True) z inkrementowanym revision_count.

        Reguła: klucz aktywny, nie-control, nie-chroniony przez GC_PROTECTED_PREFIXES,
        którego entity_root lub sam klucz trafia w barierę dyskursywną (head-token
        w QUERY_META_STOPSTEMS/QUERY_META_STOPWORDS albo stem pomocniczy śmieci).
        NIE usuwa rekordu — zapisuje nową wersję przez `kv.set_state` (łańcuch audytu
        SHA-256, reason='discourse_garbage_tombstone') i usuwa z kopii all_states (RAM),
        by śmieć nie stał się kandydatem kaskady L0→L1→L2.
        """
        if self.kv_store is None:
            return
        # V151 (H3): wąski wyjątek — namespace 'fact:' (fakty użytkownika) jest
        # chroniony przed DELETE w GC, ale MUSI podlegać tombstone'owaniu
        # śmieci dyskursywnych (model 'fact:<śmieć>:<slot>'). Pozostałe
        # chronione prefiksy (rule/preference/constraint/user/sop/atlas/...)
        # NIE są nigdy ruszane przez tę ścieżkę.
        blocked = tuple(p for p in GC_PROTECTED_PREFIXES if p != "fact:")
        for key in sorted(all_states):
            k = str(key)
            data = all_states[k]
            if not isinstance(data, dict):
                continue
            meta = data.get("metadata")
            meta = meta if isinstance(meta, dict) else {}
            if self._is_superseded(data, meta) or self._is_control_key(k):
                continue
            if k.startswith(blocked):
                continue
            # Entity-segment (środkowa część klucza) jest kanonicznym tematem —
            # tam właśnie żyje 'jak_obecnie' w 'fact:jak_obecnie:state'.
            try:
                coord = EpistemicCoordinate.from_key(k)
                subject_seg = coord.entity_id or k
            except Exception:
                subject_seg = k
            if not (is_discourse_garbage_subject(subject_seg) or is_discourse_garbage_subject(k)):
                continue
            try:
                rev = int(meta.get("revision_count", 0)) if meta.get("revision_count") is not None else 0
            except (TypeError, ValueError):
                rev = 0
            new_meta = dict(meta)
            new_meta.update({
                "is_superseded": True,
                "superseded_at": time.time(),
                "supersession_reason": GARBAGE_TOMBSTONE_REASON,
                "revision_count": rev + 1,
            })
            try:
                confidence = float(data.get("confidence", meta.get("confidence", 0.0)) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            try:
                await self.kv_store.set_state(
                    key=k,
                    value=data.get("value"),
                    confidence=confidence,
                    metadata=new_meta,
                    reason=GARBAGE_TOMBSTONE_REASON,
                )
            except Exception as exc:  # tombstone nie może wywrócić cyklu snu
                logger.warning("[DeclarativeCascade] garbage tombstone failed for %s: %s", k, exc)
                continue
            # V151.1 (F-031): spread — widoki muszą odzwierciedlić tombstone KV
            try:
                await self._tombstone_views(k, subject_seg, new_meta["superseded_at"])
            except Exception as exc:  # spread nie wywraca cyklu snu
                logger.warning("[V151.1] tombstone views sync failed for %s: %s", k, exc)
            all_states.pop(k, None)
            stats.garbage_tombstoned += 1
            logger.info("[V151] Garbage tombstone: key=%s subject=%s revision_count=%d", k, subject_seg, rev + 1)

    async def _mark_superseded(self, old: CandidateFact, canon: CandidateFact) -> bool:
        """Zapisuje is_superseded w KV przez chained set_state (bez usuwania rekordu)."""
        if self.kv_store is None:
            return False
        meta = dict(old.metadata)
        meta.update({
            "is_superseded": True,
            "superseded_by": canon.key,
            "superseded_at": time.time(),
            "supersession_reason": SUPERSESSION_REASON,
        })
        await self.kv_store.set_state(
            key=old.key,
            value=old.value,
            confidence=old.confidence,
            metadata=meta,
            reason=SUPERSESSION_REASON,
        )
        # V109 Pillar 1: Evict superseded entry from Qdrant vector index & Kùzu dual-node passages
        if self.vector_store is not None:
            try:
                del_async = getattr(self.vector_store, "delete_by_upsert_key", None)
                del_sync = getattr(self.vector_store, "delete_by_upsert_key_sync", None)
                if callable(del_async):
                    await del_async(old.key)
                elif callable(del_sync):
                    del_sync(old.key)
            except Exception as exc:
                logger.debug("[DeclarativeCascade] vector eviction failed for %s: %s", old.key, exc)
        if self.graph_store is not None and hasattr(self.graph_store, "rollback_dual_node_record"):
            try:
                self.graph_store.rollback_dual_node_record(old.key)
            except Exception as exc:
                logger.debug("[DeclarativeCascade] kuzu dual-node rollback failed for %s: %s", old.key, exc)
        return True

    def _wire_supersession_edge(self, old_key: str, canon_key: str, confidence: float) -> bool:
        """Krawędź (old)-[:SUPERSEDED_BY]->(new) w Kùzu.

        Przez EpistemicWritePath.commit_relation — doktryna V56 (test_v56_grep_
        add_relation_choke_point): WSZYSTKIE zapisy do grafu idą przez single
        write-path choke point, nigdy bezpośrednio do graph_store.add_relation.
        """
        try:
            return self._write_path.commit_relation(
                str(old_key), "SUPERSEDED_BY", str(canon_key),
                confidence=float(confidence),
            )
        except Exception as exc:  # graf nie może wywrócić cyklu snu
            logger.warning("[DeclarativeCascade] graph edge wiring failed for %s: %s", old_key, exc)
            return False

    # ── Salience GC ───────────────────────────────────────────────────────────
    @classmethod
    def _has_provenance(cls, meta: Dict[str, Any]) -> bool:
        if meta.get("provenance") or meta.get("source_episode") or meta.get("episode_id"):
            return True
        source = str(meta.get("source_type") or meta.get("source") or "")
        return source in PROVENANCE_SOURCE_TYPES

    @staticmethod
    def _is_temporary(key: str, meta: Dict[str, Any]) -> bool:
        if meta.get("is_temporary") or meta.get("temporary") or meta.get("ttl") is not None:
            return True
        return key.startswith("temp:")

    @classmethod
    def is_gc_eligible(cls, key: str, data: Dict[str, Any], meta: Dict[str, Any], gc_floor: float) -> bool:
        """Tylko tymczasowe, bez provenance, conf < floor; nigdy reguły/fakty/chronione prefiksy."""
        if not key or cls._is_superseded(data, meta):
            return False
        if key.startswith(GC_PROTECTED_PREFIXES) or cls._is_control_key(key):
            return False
        if not cls._is_temporary(key, meta):
            return False
        if cls._has_provenance(meta):
            return False
        try:
            confidence = float(data.get("confidence", 1.0))
        except (TypeError, ValueError):
            return False
        return confidence < gc_floor

    async def _delete_state(self, key: str) -> bool:
        """
        V72.3 (BUG-060): propaguje wynik kasowania z KV.

        Do V72.2 wynik ``delete_state`` (``deleted = cursor.rowcount > 0``) był
        ignorowany, a metoda zwracała bezwarunkowe ``True`` — więc gdy klucza nie
        było w bazie, kaskada i tak podbijała ``stats.gc_pruned`` i wyrzucała klucz
        z lokalnego ``all_states``. Teraz sukces = faktyczne usunięcie wiersza.
        """
        if self.kv_store is None:
            return False
        delete = getattr(self.kv_store, "delete_state", None)
        if callable(delete):
            res = await delete(key)
            return bool(res)
        delete_sync = getattr(self.kv_store, "delete_sync", None)
        if callable(delete_sync):
            res = delete_sync(key)
            return bool(res)
        return False

    # ── Główna kaskada ────────────────────────────────────────────────────────
    async def consolidate(self, candidate_keys: Optional[Set[str]] = None) -> DeclarativeConsolidationStats:
        """Uruchamia L0→L1→L2 + Salience GC na zweryfikowanym stanie KV."""
        start = time.perf_counter()
        stats = DeclarativeConsolidationStats()
        self._blocked_roots = 0

        if self.kv_store is None:
            stats.notes.append("no_kv_store")
            stats.duration_ms = (time.perf_counter() - start) * 1000.0
            return stats

        if candidate_keys is not None and len(candidate_keys) == 0:
            stats.duration_ms = (time.perf_counter() - start) * 1000.0
            return stats

        all_states: Dict[str, Any] = {}
        if candidate_keys is not None:
            get_states: Any = getattr(self.kv_store, "get_states", None)
            if callable(get_states):
                fetched = await get_states(list(candidate_keys))
                all_states = dict(fetched) if fetched else {}
            elif hasattr(self.kv_store, "get_sync"):
                for k in candidate_keys:
                    val = self.kv_store.get_sync(k)
                    if val is not None:
                        all_states[k] = val if isinstance(val, dict) else {"value": val}
            # V109 Pillar 1: Expand candidate_keys with same-entity_root siblings from KV
            get_arb: Any = getattr(self.kv_store, "get_arbitration_candidates_sync", None)
            if callable(get_arb):
                for k in list(candidate_keys):
                    try:
                        val_str = str((all_states.get(k) or {}).get("value", ""))
                        sibs = get_arb(key=k, value=val_str, limit=50)
                        if isinstance(sibs, dict):
                            for sk, sv in sibs.items():
                                if sk not in all_states and entity_root(sk) == entity_root(k) and entity_root(k):
                                    all_states[sk] = sv
                    except Exception as exc:
                        # V112 F-028+ (fail-loud-or-evidence): sibling-fetch beepowodzie
                        # wczesniej ginial bez sladu; teraz dowod w swallow ledger.
                        record_swallow("l3_procedural.declarative_consolidator.sibling_fetch", str(k), exc)
        else:
            get_all: Any = getattr(self.kv_store, "get_all_states", None)
            get_all_sync: Any = getattr(self.kv_store, "get_all_sync", None)
            if callable(get_all):
                fetched = await get_all(paged=True, limit=5000)
                all_states = dict(fetched) if fetched else {}
            elif callable(get_all_sync):
                fetched_sync = get_all_sync(paged=True, limit=5000)
                all_states = dict(fetched_sync) if fetched_sync else {}

        # 1) Salience GC (przed kaskadą — sprząta śmieci, których kaskada nie dotyczy).
        # V71.1 F1: iteracja po KOPII (sorted(...)) — zbiór mutujemy bezpiecznie poza
        # iteracją. Po udanym DELETE rekord MUSI zniknąć także ze słownika w RAM;
        # inaczej collect_candidates() niżej wczyta go ponownie jako kandydata,
        # a przegrana kanonizacja wywoła _mark_superseded → set_state, co fizycznie
        # wskrzesi skasowany wiersz w SQLite (zombie resurrectio).
        for key, data in sorted(all_states.items()):
            if not isinstance(data, dict):
                continue
            meta = data.get("metadata")
            meta = meta if isinstance(meta, dict) else {}
            if not self.is_gc_eligible(str(key), data, meta, self.gc_conf_floor):
                continue
            if await self._delete_state(str(key)):
                all_states.pop(str(key), None)
                stats.gc_pruned += 1

        # 1b) Garbage tombstone pass (V151, Wektor I / H3): śmieci dyskursywne
        # (zaimek/wyraz pytający wcielony w encję — 'jak_obecnie', 'czy_warto'),
        # które dawniej przeciekały do stanu i ZOSTAWAŁY aktywnymi pseudo-subjektami,
        # są teraz tombstone'owane w cyklu snu (is_superseded + revision_count+1) —
        # odcięte od kaskady deklaratywnej i od online'owego ConflictArbitera.
        await self._tombstone_discourse_garbage(all_states, stats)

        # 2) L0 → L1 → L2 kaskada deklaratywna.
        candidates = self.collect_candidates(all_states)
        stats.candidates_scanned = len(candidates)
        if len(candidates) > self.fact_batch_cap:
            candidates = candidates[: self.fact_batch_cap]
            stats.truncated_by_batch_cap = True
            stats.notes.append(f"batch_cap={self.fact_batch_cap}")

        # V109 Pillar 1: Symbolic ConflictArbiter pairing across same-entity_root candidates
        arbiter_pairs: List[Tuple[str, str, float]] = []
        by_root: Dict[str, List[CandidateFact]] = {}
        for c in candidates:
            if c.root:
                by_root.setdefault(c.root, []).append(c)
        arbiter = ConflictArbiter()
        for _root, r_members in by_root.items():
            if len(r_members) < 2:
                continue
            for idx_a in range(len(r_members)):
                for idx_b in range(idx_a + 1, len(r_members)):
                    ca, cb = r_members[idx_a], r_members[idx_b]
                    coord_ca = EpistemicCoordinate.from_key(ca.key)
                    coord_cb = EpistemicCoordinate.from_key(cb.key)
                    if are_orthogonal_slots(coord_ca, coord_cb):
                        self._blocked_orthogonal_slots += 1
                        continue
                    try:
                        arb_res = arbiter.arbitrate(
                            key=ca.key,
                            val=str(ca.value),
                            confidence=ca.confidence,
                            existing_items={cb.key: {"value": str(cb.value), "confidence": cb.confidence}},
                        )
                        if arb_res.is_conflict or arb_res.is_belief_revision or arb_res.action == "supersede":
                            arbiter_pairs.append((ca.key, cb.key, max(0.88, float(arb_res.best_score or 0.90))))
                    except Exception as exc:
                        # V112 F-028+ (fail-loud-or-evidence): arbiter pairwise probe
                        # bez dowodu — teraz zapisany w swallow ledger.
                        record_swallow("l3_procedural.declarative_consolidator.arbiter_pairwise", f"{ca.key}|{cb.key}", exc)

        embedder = self._resolve_embedder()
        if embedder is None and not arbiter_pairs:
            stats.skipped_no_vectors = len(candidates)
            stats.notes.append("no_vector_embedder: L0-L2 skipped")
        else:
            by_key = {c.key: c for c in candidates}
            pairs = list(self.find_pairs(candidates, embedder)) if embedder is not None else []
            existing_pair_keys = {(a, b) for a, b, _ in pairs} | {(b, a) for a, b, _ in pairs}
            for ap_a, ap_b, ap_s in arbiter_pairs:
                if (ap_a, ap_b) not in existing_pair_keys:
                    pairs.append((ap_a, ap_b, ap_s))
            stats.pairs_found = len(pairs)
            stats.roots_blocked = self._blocked_roots
            stats.orthogonal_slots_blocked = self._blocked_orthogonal_slots

            clusters = self.connected_components([c.key for c in candidates], pairs)
            stats.clusters = len(clusters)

            superseded_pairs: List[str] = []
            for cluster in clusters:
                members = [by_key[k] for k in cluster if k in by_key]
                if len(members) < 2:
                    continue
                canon = self.pick_canonical(members)
                for member in members:
                    if member.key == canon.key:
                        continue
                    if not await self._mark_superseded(member, canon):
                        continue
                    stats.facts_superseded += 1
                    superseded_pairs.append(f"{member.key}->{canon.key}")
                    if self._wire_supersession_edge(member.key, canon.key, member.confidence):
                        stats.edges_wired += 1

        stats.duration_ms = (time.perf_counter() - start) * 1000.0
        if stats.facts_superseded or stats.gc_pruned:
            logger.info(
                "[DeclarativeCascade] pairs=%d clusters=%d superseded=%d gc_pruned=%d roots_blocked=%d orthogonal_slots_blocked=%d superseded_pairs=%s",
                stats.pairs_found, stats.clusters, stats.facts_superseded,
                stats.gc_pruned, stats.roots_blocked, stats.orthogonal_slots_blocked,
                superseded_pairs[:10],
            )
        return stats


# ── V67 (F2): ciała liczące pary kandydatów ──────────────────────────────────
# MODULOWE helpery (koniec pliku, nigdy w środku klasy). Właściwości:
#  * manual_cosine_pairs = dosłowne odzwierciedlenie pętli przed-V67 (referencja
#    do testu ekwiwalencji + ścieżka awaryjna dla macierzy ragged),
#  * numpy_cosine_pairs = ścieżka wektoryzowana (M_norm @ M_norm.T + triu maska).
# Oba zwracają TĘ SAMĄ listę (i, j, cosine) w tej samej kolejności (i rosnąco,
# j rosnąco, i < j) i przy tym samym progu — guard Entity-Root oraz licznik
# zablokowanych korzeni zostają w wołającym find_pairs().


def manual_cosine_pairs(
    vectors: Sequence[Sequence[float]],
    threshold: float,
) -> List[Tuple[int, int, float]]:
    """Referencyjna pętla O(n²): pary (i, j, cosine) o cosine >= threshold (i < j)."""
    hits: List[Tuple[int, int, float]] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim = cosine_similarity(vectors[i], vectors[j])
            if sim >= threshold:
                hits.append((i, j, sim))
    return hits


def numpy_cosine_pairs(
    vectors: Sequence[Sequence[float]],
    threshold: float,
) -> List[Tuple[int, int, float]]:
    """Wektoryzowane pary (i, j, cosine) >= threshold — ekwiwalent manual_cosine_pairs.

    Rzuca ValueError/TypeError, gdy wektory nie tworzą regularnej macierzy
    (ragged, różne wymiary, puste) — wtedy find_pairs() świadomie wraca do
    referencyjnej pętli ręcznej zamiast zgłaszać błąd konsolidacji snu.
    """
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(vectors) or matrix.shape[1] == 0:
        raise ValueError("wektory kandydatów nie tworzą regularnej macierzy")
    norms = np.linalg.norm(matrix, axis=1)
    # Wektor zerowy => jednostkowy zerowy (cosine 0.0), dokładnie jak w referencji.
    safe = np.where(norms > 0.0, norms, 1.0)
    unit = matrix / safe[:, None]
    sim_matrix = unit @ unit.T
    iu, ju = np.triu_indices(len(vectors), k=1)
    vals = sim_matrix[iu, ju]
    keep = vals >= threshold
    return [(int(i), int(j), float(v)) for i, j, v in zip(iu[keep], ju[keep], vals[keep], strict=True)]
