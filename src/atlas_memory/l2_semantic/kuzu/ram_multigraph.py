from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Any, Dict, Optional

import networkx as nx

from atlas_memory.l2_semantic.audit_ledger import record_swallow
from atlas_memory.l2_semantic.kuzu.schema_driver import _as_float

logger = logging.getLogger(__name__)


@lru_cache(maxsize=10000)
def _fast_lemma(w: str) -> str:
    """O(1) cached Polish lemmatization via spaCy with fallback to bilingual stem."""
    clean = w.strip().lower()
    if not clean or len(clean) < 3 or any(ch.isdigit() for ch in clean):
        return clean
    try:
        from atlas_memory.core.linguistics import _bilingual_stem, get_spacy_nlp
        nlp = get_spacy_nlp("pl", fast_lemma_only=True)
        if nlp is not None:
            doc = nlp(clean)
            if len(doc) > 0 and doc[0].lemma_:
                return doc[0].lemma_.lower().strip()
        return _bilingual_stem(clean) or clean
    except Exception:
        return clean


class MultiEdgeData(dict):
    """Słownik atrybutów krawędzi w multigrafie zachowujący zgodność wsteczną z DiGraph.

    W nx.MultiDiGraph get_edge_data(u, v) zwraca słownik {key: edge_dict}.
    Klasa MultiEdgeData pozwala na odczyt zarówno po kluczu krawędzi (np. d['causes']),
    jak i bezpośrednio po nazwie atrybutu (np. d['confidence'], d.get('predicate')),
    delegując do krawędzi o najwyższym wskaźniku confidence.
    """

    def __getitem__(self, item: Any) -> Any:
        if item in self:
            return super().__getitem__(item)
        if len(self) > 0:
            best = max(
                self.values(),
                key=lambda d: _as_float(d.get("confidence"), 0.0) if isinstance(d, dict) else 0.0,
            )
            if isinstance(best, dict) and item in best:
                return best[item]
        return super().__getitem__(item)

    def get(self, item: Any, default: Any = None) -> Any:
        try:
            return self[item]
        except KeyError:
            return default


class AtlasMultiDiGraph(nx.MultiDiGraph):
    """Natywny MultiDiGraph ATLAS-a z przezroczystym fallbackiem atrybutów w get_edge_data, O(1) number_of_edges() oraz O(1) indeksami kanonicznymi/lematycznymi."""

    def __init__(self, incoming_graph_data: Any = None, multigraph_input: Optional[bool] = None, **attr: Any) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._cached_edge_count: Optional[int] = 0
        self._revision: int = 0
        super().__init__(incoming_graph_data=incoming_graph_data, multigraph_input=multigraph_input, **attr)
        if incoming_graph_data is not None:
            self._cached_edge_count = None
        self._node_index_lower: Dict[str, Any] = {}
        self._canonical_lower: Dict[str, str] = {}
        self._lemma_index: Dict[str, set[str]] = {}
        self._adj_out = self._adj
        self._adj_in = self._pred
        if incoming_graph_data is not None:
            self._rebuild_indices()

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def revision(self) -> int:
        return self._revision

    def bump_revision(self) -> int:
        """Explicit revision increment on edge attribute updates or re-weighting."""
        with self._lock:
            self._revision += 1
            return self._revision

    def _index_node_entry(self, n: Any) -> None:
        """V122: Indeksuje węzeł w O(1) canonical_lower oraz lemma_index."""
        s = str(n).strip()
        if not s:
            return
        s_low = s.lower()
        self._canonical_lower[s_low] = s
        self._node_index_lower[s_low] = s

        # Indeksowanie lematów słów składowych
        parts = [
            w for w in s_low.replace(":", " ").replace("_", " ").replace("-", " ").split()
            if len(w) >= 3 and not any(ch.isdigit() for ch in w)
        ]
        for p in parts:
            lemma = _fast_lemma(p)
            if lemma:
                self._lemma_index.setdefault(lemma, set()).add(s)
            self._lemma_index.setdefault(p, set()).add(s)

    def add_node(self, node_for_adding: Any, **attr: Any) -> None:
        with self._lock:
            self._revision += 1
            res = super().add_node(node_for_adding, **attr)
            self._index_node_entry(node_for_adding)
            return res

    def add_nodes_from(self, nodes_for_adding: Any, **attr: Any) -> None:
        with self._lock:
            self._revision += 1
            res = super().add_nodes_from(nodes_for_adding, **attr)
            for n in nodes_for_adding:
                node = n[0] if isinstance(n, tuple) else n
                self._index_node_entry(node)
            return res

    def add_edge(self, u_for_edge: Any, v_for_edge: Any, key: Any = None, **attr: Any) -> Any:
        with self._lock:
            self._revision += 1
            self._cached_edge_count = None
            return super().add_edge(u_for_edge, v_for_edge, key=key, **attr)

    def add_edges_from(self, ebunch_to_add: Any, **attr: Any) -> Any:
        with self._lock:
            self._revision += 1
            self._cached_edge_count = None
            return super().add_edges_from(ebunch_to_add, **attr)

    def remove_edge(self, u: Any, v: Any, key: Any = None) -> None:
        with self._lock:
            self._revision += 1
            self._cached_edge_count = None
            super().remove_edge(u, v, key=key)

    def remove_edges_from(self, ebunch: Any) -> None:
        with self._lock:
            self._revision += 1
            self._cached_edge_count = None
            super().remove_edges_from(ebunch)

    def remove_node(self, n: Any) -> None:
        with self._lock:
            self._revision += 1
            self._cached_edge_count = None
            s = str(n).strip()
            s_low = s.lower()
            self._canonical_lower.pop(s_low, None)
            self._node_index_lower.pop(s_low, None)
            super().remove_node(n)

    def remove_nodes_from(self, nodes: Any) -> None:
        with self._lock:
            self._revision += 1
            self._cached_edge_count = None
            for n in nodes:
                s_low = str(n).strip().lower()
                self._canonical_lower.pop(s_low, None)
                self._node_index_lower.pop(s_low, None)
            super().remove_nodes_from(nodes)

    def clear(self) -> None:
        with self._lock:
            self._revision += 1
            self._cached_edge_count = 0
            self._canonical_lower.clear()
            self._node_index_lower.clear()
            self._lemma_index.clear()
            super().clear()

    def _rebuild_indices(self) -> None:
        """Odbudowuje indeksy canonical_lower oraz lemma_index dla wszystkich węzłów."""
        self._canonical_lower.clear()
        self._node_index_lower.clear()
        self._lemma_index.clear()
        for node in self.nodes:
            self._index_node_entry(node)

    def resolve_canonical_node(self, term: str) -> Optional[str]:
        """O(1) wyszukiwanie kanonicznej nazwy węzła (exact lub case-insensitive)."""
        if self.has_node(term):
            return term
        t_low = str(term).lower().strip()
        return self._canonical_lower.get(t_low)

    def resolve_nodes_by_lemma(self, lemma: str) -> set[str]:
        """O(1) wyszukiwanie kanonicznych węzłów powiązanych z danym lematem."""
        l_low = str(lemma).lower().strip()
        matched = self._lemma_index.get(l_low, set())
        return {n for n in matched if self.has_node(n)}

    def resolve_query_entities(self, words: list[str], query: str = "") -> list[str]:
        """W122: Błyskawiczna rezolucja encji z zapytania (O(1) canonical + lemma + n-gram)."""
        resolved: list[str] = []
        seen: set[str] = set()

        def _add(cand: Optional[str]) -> None:
            if cand and self.has_node(cand) and cand not in seen:
                seen.add(cand)
                resolved.append(cand)

        # 1. Multi-word n-gramy z query (np. 'Obsidian MCP', 'LOOP Engineering')
        q_clean = str(query or "").strip()
        if q_clean:
            q_tokens = [tok.strip(".,!?:;\"'()[]{}") for tok in q_clean.split()]
            q_tokens = [tok for tok in q_tokens if tok]
            for n_gram in (3, 2):
                for i in range(len(q_tokens) - n_gram + 1):
                    phrase = " ".join(q_tokens[i:i+n_gram]).lower()
                    _add(self._canonical_lower.get(phrase))

        # 2. Tokeny ze słów zapytania (exact i case-insensitive)
        for w in words:
            w_str = str(w).strip(".,!?:;\"'()[]{}")
            if not w_str:
                continue
            if self.has_node(w_str):
                _add(w_str)
            _add(self._canonical_lower.get(w_str.lower()))

        # 3. Lematy z zapytania (dla fleksji: 'narzędzia' -> 'narzędzie', 'wazach' -> 'waza')
        for w in words:
            w_clean = str(w).lower().strip(".,!?:;\"'()[]{}")
            if not w_clean or len(w_clean) < 3:
                continue
            lemma = _fast_lemma(w_clean)
            if lemma:
                for n in self.resolve_nodes_by_lemma(lemma):
                    _add(n)

        return resolved

    def clear_edges(self) -> None:
        self._cached_edge_count = 0
        super().clear_edges()

    def number_of_edges(self, u: Any = None, v: Any = None) -> int:
        if u is None and v is None:
            if self._cached_edge_count is None:
                self._cached_edge_count = super().number_of_edges()
            return self._cached_edge_count
        return super().number_of_edges(u=u, v=v)

    def get_edge_data(self, u: Any, v: Any, key: Any = None, default: Any = None) -> Any:
        d = super().get_edge_data(u, v, key=key, default=default)
        if key is None and isinstance(d, dict) and not isinstance(d, MultiEdgeData):
            return MultiEdgeData(d)
        return d


class KuzuRamHydrationMixin:
    """In-memory NetworkX hydration and telemetry stats from Kùzu disk storage."""

    def hydrate_from_disk_sync(self) -> Dict[str, int]:
        """
        W1 (V72.2): wczytuje CAŁY graf z pliku Kùzu (węzły ``Entity`` + krawędzie
        ``RELATION`` z atrybutami predicate/confidence/timestamp) do ``self.nx_graph``.

        Właściwości:
        * **addytywna i idempotentna** — nic nie usuwa; dwukrotne wywołanie daje ten sam
          stan, bo ``add_node``/``add_edge`` nadpisują istniejące wpisy, a licznik
          zwracany jako ``loaded_*`` dotyczy nowo dodanych elementów;
        * **semantyka MAX jak w ``add_relation``** — przy kolizji pary (DiGraph trzyma
          jedną krawędź na parę) confidence/timestamp idą w górę, a predykat bierze się
          z relacji o wyższej pewności (remis = pierwszy w kolejności leksykograficznej,
          więc wynik jest deterministyczny niezależnie od porządku skanu Kùzu);
        * węzły już obecne w RAM nie są nadpisywane (lokalny zapis jest nowszy niż dysk).

        Zwraca ``{"loaded_nodes", "loaded_edges", "kuzu_nodes", "kuzu_edges"}``.
        Gdy Kùzu jest niedostępne — zera i zero wyjątków (NetworkX działa dalej).
        """
        with self._graph_lock:
            if self.conn is None:
                logger.debug("hydrate_from_disk: Kùzu niedostępne — RAM bez zmian")
                return {
                    "loaded_nodes": 0, "loaded_edges": 0,
                    "kuzu_nodes": 0, "kuzu_edges": 0, "kuzu_pairs": 0,
                    "partial": False,
                }

            loaded_nodes = 0
            loaded_edges = 0
            disk_nodes = 0
            disk_edges = 0
            disk_pairs = 0
            partial = False

            # V72.2 (W1a): skany per etap + stronicowanie. Do 2026-09-10 oba skany
            # dzieliły jeden `try`, więc awaria bufra przy węzłach (Kùzu 0.11.3
            # wyczerpuje pulę na skanie ~2k wierszy dzielonej z resztą procesu)
            # przerywała CAŁĄ hydrację — razem ze skanem krawędzi. Licznik pokazywał
            # wtedy „na dysku: 0 RELATION", więc kryterium parytetu (ram_edges ==
            # kuzu_edges) świeciło zielono przez 0 == 0. Pomiar na produkcji
            # (2026-09-10): 2127/2483 węzłów i 0/2161 krawędzi w RAM.
            from atlas_memory.core.linguistics import is_valid_graph_entity
            invalid_disk_entities: set[str] = set()
            try:
                node_rows = self._hydrate_paged(
                    "MATCH (e:Entity) RETURN e.name, e.entity_type ORDER BY e.name"
                )
                for row in node_rows:
                    name = row[0] if row else None
                    if name is None or str(name) == "":
                        continue
                    name_s = str(name)
                    if not is_valid_graph_entity(name_s):
                        invalid_disk_entities.add(name_s)
                        continue
                    disk_nodes += 1
                    if name_s in self.nx_graph:
                        continue
                    etype = row[1] if len(row) > 1 and row[1] is not None else "entity"
                    self.nx_graph.add_node(name_s, type=str(etype))
                    loaded_nodes += 1
            except Exception as exc:  # pragma: no cover - awaria silnika nie może wywalić startu
                partial = True
                logger.warning("hydrate_from_disk: skan WĘZŁÓW zawiódł (%s) — krawędzie czytam mimo to", exc)

            # Etap krawędzi we własnym `try`: awaria węzłów nie wolno mu już pomijać skanu.
            try:
                rel_rows = self._hydrate_paged(
                    "MATCH (a:Entity)-[r:RELATION]->(b:Entity) "
                    "RETURN a.name, b.name, r.predicate, r.confidence, r.timestamp "
                    "ORDER BY a.name, b.name, r.predicate"
                )
                rel_rows.sort(key=lambda r: (str(r[0]), str(r[1]), str(r[2] if r[2] is not None else "")))
                valid_rel_rows = []
                for r in rel_rows:
                    if r[0] is None or r[1] is None:
                        continue
                    s_n, o_n = str(r[0]), str(r[1])
                    if not is_valid_graph_entity(s_n):
                        invalid_disk_entities.add(s_n)
                        continue
                    if not is_valid_graph_entity(o_n):
                        invalid_disk_entities.add(o_n)
                        continue
                    valid_rel_rows.append(r)
                disk_pairs = len({(str(r[0]), str(r[1])) for r in valid_rel_rows})
                for subj, obj, pred, conf, ts in (tuple(r) for r in valid_rel_rows):
                    subj, obj = str(subj), str(obj)
                    disk_edges += 1
                    predicate = str(pred) if pred not in (None, "") else "related_to"
                    confidence = _as_float(conf, 1.0)
                    stamp = _as_float(ts, 0.0)

                    if subj not in self.nx_graph:
                        self.nx_graph.add_node(subj, type="entity")
                        loaded_nodes += 1
                    if obj not in self.nx_graph:
                        self.nx_graph.add_node(obj, type="entity_or_value")
                        loaded_nodes += 1

                    prev = self.nx_graph.get_edge_data(subj, obj, key=predicate)
                    if prev is None:
                        self.nx_graph.add_edge(
                            subj, obj,
                            key=predicate,
                            predicate=predicate,
                            confidence=confidence,
                            timestamp=stamp,
                            session_id=None,
                        )
                        loaded_edges += 1
                        continue

                    # MultiDiGraph: ta sama trójka relacyjna już istniała — scal w górę (confidence/timestamp).
                    prev_conf = _as_float(prev.get("confidence"), 0.0)
                    prev_ts = _as_float(prev.get("timestamp"), 0.0)
                    merged_conf = max(prev_conf, confidence)
                    merged_ts = max(prev_ts, stamp)
                    self.nx_graph.add_edge(
                        subj, obj,
                        key=predicate,
                        predicate=predicate,
                        confidence=merged_conf,
                        timestamp=merged_ts,
                        session_id=prev.get("session_id"),
                    )
            except Exception as exc:  # pragma: no cover - patrz wyżej (etap węzłów jest osobno)
                partial = True
                logger.warning("hydrate_from_disk: skan KRAWĘDZI zawiódł (%s) — hydracja częściowa", exc)

            # V104 (W1): Hydracja węzłów Passage oraz krawędzi CONTAINS i SYNONYM (opcjonalne, bez wywracania startu)
            try:
                passage_rows = self._hydrate_paged(
                    "MATCH (p:Passage) RETURN p.passage_id, p.source_key, p.text ORDER BY p.passage_id"
                )
                for pid, skey, txt in (tuple(r) for r in passage_rows if len(r) >= 3):
                    if pid is None or str(pid) == "":
                        continue
                    pid_s = str(pid)
                    was_in_graph = pid_s in self.nx_graph
                    self.nx_graph.add_node(
                        pid_s,
                        type="passage",
                        source_key=str(skey or pid_s),
                        text=str(txt or ""),
                    )
                    if not was_in_graph:
                        loaded_nodes += 1
            except Exception as exc:
                logger.debug("hydrate_from_disk: Passage scan skipped (%s)", exc)

            try:
                contains_rows = self._hydrate_paged(
                    "MATCH (p:Passage)-[c:CONTAINS]->(e:Entity) RETURN p.passage_id, e.name, c.weight ORDER BY p.passage_id, e.name"
                )
                for pid, ename, weight in (tuple(r) for r in contains_rows if len(r) >= 3):
                    if pid is None or ename is None:
                        continue
                    pid_s, ename_s = str(pid), str(ename)
                    if not is_valid_graph_entity(ename_s):
                        invalid_disk_entities.add(ename_s)
                        continue
                    w_val = _as_float(weight, 1.0)
                    if pid_s not in self.nx_graph:
                        self.nx_graph.add_node(pid_s, type="passage", source_key=pid_s, text="")
                    if ename_s not in self.nx_graph:
                        self.nx_graph.add_node(ename_s, type="entity")
                    self.nx_graph.add_edge(pid_s, ename_s, key="CONTAINS", predicate="CONTAINS", confidence=w_val, weight=w_val, timestamp=0.0)
                    loaded_edges += 1
            except Exception as exc:
                logger.debug("hydrate_from_disk: CONTAINS scan skipped (%s)", exc)

            try:
                syn_rows = self._hydrate_paged(
                    "MATCH (a:Entity)-[s:SYNONYM]->(b:Entity) RETURN a.name, b.name, s.sim ORDER BY a.name, b.name"
                )
                for ea, eb, sim_v in (tuple(r) for r in syn_rows if len(r) >= 3):
                    if ea is None or eb is None:
                        continue
                    ea_s, eb_s = str(ea), str(eb)
                    if not is_valid_graph_entity(ea_s) or not is_valid_graph_entity(eb_s):
                        continue
                    s_val = _as_float(sim_v, 0.85)
                    if ea_s not in self.nx_graph:
                        self.nx_graph.add_node(ea_s, type="entity")
                    if eb_s not in self.nx_graph:
                        self.nx_graph.add_node(eb_s, type="entity")
                    self.nx_graph.add_edge(ea_s, eb_s, key="SYNONYM", predicate="SYNONYM", confidence=s_val, sim=s_val, timestamp=0.0)
                    loaded_edges += 1
            except Exception as exc:
                logger.debug("hydrate_from_disk: SYNONYM scan skipped (%s)", exc)

            if invalid_disk_entities and self.conn is not None:
                purged_cnt = 0
                for bad_ent in list(invalid_disk_entities):
                    try:
                        self.conn.execute("MATCH (e:Entity {name: $name}) DETACH DELETE e", {"name": bad_ent})
                        purged_cnt += 1
                    except Exception as exc:
                        # V112 F-028+: purge invalid/hub/prose entity z Kùzu disk nie udal sie
                        # — wczesniej pass (cisza). Przeplyw bez zmian (entity zostaje),
                        # ale awaria ma dowod: warning + swallow ledger.
                        record_swallow("l2_semantic.kuzu.ram_multigraph.purge_invalid_entity", str(bad_ent), exc)
                if purged_cnt > 0:
                    logger.info("[GRAPH:PURGE] Purged %d invalid/hub/prose Entity nodes from Kùzu disk", purged_cnt)
            disk_passages = sum(1 for _, d in self.nx_graph.nodes(data=True) if isinstance(d, dict) and d.get("type") == "passage")
            disk_contains = sum(1 for _, _, d in self.nx_graph.edges(data=True) if isinstance(d, dict) and d.get("predicate") == "CONTAINS")
            disk_synonyms = sum(1 for _, _, d in self.nx_graph.edges(data=True) if isinstance(d, dict) and d.get("predicate") == "SYNONYM")
            logger.info(
                "[GRAPH:HYDRATE] RAM Multigraph ready: %d total_nodes (Entity=%d, Passage=%d) | "
                "%d total_edges (RELATION=%d [%d unique pairs], CONTAINS=%d, SYNONYM=%d)%s",
                loaded_nodes, disk_nodes, disk_passages,
                loaded_edges, disk_edges, disk_pairs, disk_contains, disk_synonyms,
                " [PARTIAL!]" if partial else " [OK]",
            )
            self._hydration_partial = bool(partial)
            return {
                "loaded_nodes": loaded_nodes,
                "loaded_edges": loaded_edges,
                "kuzu_nodes": disk_nodes,
                "kuzu_passages": disk_passages,
                "kuzu_edges": disk_edges,
                "kuzu_contains": disk_contains,
                "kuzu_synonyms": disk_synonyms,
                "kuzu_pairs": disk_pairs,
                "partial": partial,
            }

    def _hydrate_paged(self, base_stmt: str, page: int = 512) -> list:
        """
        W1a (V72.2): stronicowany skan Kùzu — chroni buffer pool silnika.

        Zmierzone 2026-09-10 na produkcji (2483 węzły / 2161 krawędzi): pojedynczy
        ``MATCH ... RETURN`` na całej tabeli wyczerpuje pulę buforów Kùzu 0.11.3
        (``Buffer manager exception: Unable to allocate memory``) po ok. 2,1k wierszy.
        Stronicowanie przez SKIP/LIMIT z ORDER BY (stabilna kolejność = brak dziur
        i duplikatów) trzyma rozmiar wyniku w ryzach; przy awarii strony limit maleje
        o połowę (min. 64) zamiast poddawać się od razu.
        """
        if self.conn is None:
            return []
        rows: list = []
        offset = 0
        limit = max(64, int(page))
        while True:
            try:
                got = self._drain(self.conn.execute(f"{base_stmt} SKIP {offset} LIMIT {limit}"))
            except Exception:
                if limit <= 64:
                    raise
                limit = max(64, limit // 2)
                logger.warning("hydrate_from_disk: strona skanu zawiodła — zmniejszam limit do %d", limit)
                continue
            rows.extend(got)
            if len(got) < limit:
                break
            offset += len(got)
        return rows

    async def hydrate_from_disk(self) -> Dict:
        """W1 (V72.2): asynchroniczny wrapper hydracji (kolejność blokad: _lock -> _graph_lock)."""
        async with self._lock:
            return self.hydrate_from_disk_sync()

    def get_stats_sync(self) -> Dict[str, Any]:
        """Returns node and edge counts adhering strictly to _graph_lock."""
        stats: Dict[str, Any] = {
            "ram_nodes": None,
            "ram_edges": None,
            "kuzu_nodes": None,
            "kuzu_edges": None,
            "kuzu_available": False,
        }
        if self.nx_graph is not None:
            try:
                stats["ram_nodes"] = int(self.nx_graph.number_of_nodes())
                stats["ram_edges"] = int(self.nx_graph.number_of_edges())
            except Exception as exc:
                logger.debug("graph stats (RAM) error: %s", exc)

        if self.conn is None:
            return stats

        with self._graph_lock:
            try:
                nodes_res = self.conn.execute("MATCH (e:Entity) RETURN count(e)")
                stats["kuzu_nodes"] = int(nodes_res.get_next()[0]) if nodes_res.has_next() else 0
                edges_res = self.conn.execute("MATCH ()-[r:RELATION]->() RETURN count(r)")
                stats["kuzu_edges"] = int(edges_res.get_next()[0]) if edges_res.has_next() else 0
                stats["kuzu_available"] = True
            except Exception as exc:
                logger.debug("graph stats (Kùzu COUNT) error: %s", exc)
                stats["kuzu_nodes"] = None
                stats["kuzu_edges"] = None
                stats["kuzu_available"] = False
        return stats

    async def get_stats(self) -> Dict[str, Any]:
        """Asynchronously returns graph statistics preserving lock hierarchy (_lock -> _graph_lock)."""
        async with self._lock:
            return self.get_stats_sync()
