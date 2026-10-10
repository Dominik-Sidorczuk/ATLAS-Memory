from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from atlas_memory.l2_semantic.kuzu.schema_driver import _as_float

logger = logging.getLogger(__name__)

_MAX_HOPS_LIMIT = 20


class KuzuGraphQueriesMixin:
    """Cypher multi-hop queries, Personalized PageRank, and subgraph extraction."""

    async def get_subgraph_relations(
        self,
        active_entities: List[str],
        max_depth: int = 2,
    ) -> Dict[str, Any]:
        async with self._lock:
            with self._graph_lock:
                subgraph_nodes: Set[str] = set()
                matched_roots: Set[str] = set()

                for entity in active_entities:
                    if not entity:
                        continue
                    # 1. Exact match
                    if entity in self.nx_graph:
                        matched_roots.add(entity)

                    # 2. Case-insensitive & token/substring matching
                    ent_clean = str(entity).strip().lower().replace("_", " ")
                    ent_tokens = set(w for w in ent_clean.split() if len(w) >= 2)

                    for node in self.nx_graph.nodes:
                        node_str = str(node).strip().lower()
                        if node_str == ent_clean or (len(ent_clean) >= 5 and ent_clean in node_str):
                            matched_roots.add(node)
                        elif ent_tokens:
                            node_tokens = set(w for w in re.findall(r'\w+', node_str) if len(w) >= 3)
                            common = ent_tokens & node_tokens
                            if len(common) >= 2 or (len(ent_tokens) == 1 and common):
                                matched_roots.add(node)

                for root in matched_roots:
                    subgraph_nodes.add(root)
                    curr_layer = {root}
                    for _ in range(max(1, max_depth)):
                        next_layer = set()
                        for node in curr_layer:
                            succ = set(self.nx_graph.successors(node))
                            pred = set(self.nx_graph.predecessors(node))
                            next_layer.update(succ | pred)
                        next_layer -= subgraph_nodes
                        if not next_layer:
                            break
                        subgraph_nodes.update(next_layer)
                        curr_layer = next_layer

                sub_g = self.nx_graph.subgraph(subgraph_nodes)
                edges_list = []
                for u, v, data in sub_g.edges(data=True):
                    edges_list.append({
                        "subject": u,
                        "predicate": data.get("predicate", "related_to"),
                        "object": v,
                        "confidence": data.get("confidence", 1.0),
                        "timestamp": data.get("timestamp", 0.0),
                        "session_id": data.get("session_id"),
                    })

                return {
                    "active_roots": list(matched_roots) if matched_roots else active_entities,
                    "matched_nodes_count": len(subgraph_nodes),
                    "nodes": list(subgraph_nodes),
                    "relations": edges_list,
                }

    async def find_causal_path(self, source: str, target: str) -> Optional[List[Dict[str, Any]]]:
        async with self._lock:
            with self._graph_lock:
                if not (self.nx_graph.has_node(source) and self.nx_graph.has_node(target)):
                    return None
                try:
                    path = nx.shortest_path(self.nx_graph, source, target)
                    chain = []
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        edge_data = self.nx_graph.get_edge_data(u, v) or {}
                        chain.append({
                            "from": u,
                            "predicate": edge_data.get("predicate", "leads_to"),
                            "to": v,
                            "confidence": edge_data.get("confidence", 1.0),
                        })
                    return chain
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    return None

    @staticmethod
    def _iter_path_structs(row: List[Any]) -> List[Dict[str, Any]]:
        """
        Normalizuje wiersz Kùzu dla ``RETURN path`` (typ RECURSIVE_REL).

        W kuzu 0.11.x kolumna ścieżki to dict ``{"_nodes": [...], "_rels": [...]}``;
        defensywnie obsługujemy też wariant, w którym wartość jest listą takich dictów.
        """
        if not row:
            return []
        value = row[0]
        candidates = value if isinstance(value, list) else [value]
        return [c for c in candidates if isinstance(c, dict) and "_nodes" in c]

    def _kuzu_multi_hop_paths(
        self,
        source: str,
        target: str,
        max_hops: int,
    ) -> List[Dict[str, Any]]:
        """
        Wykonuje zapytanie Cypher w Kùzu i mapuje wynik na format ścieżek NetworkX
        (``{"source", "target", "nodes", "edges"}``).

        Wołane z trzymanym ``self._graph_lock`` (pojedyncze połączenie Kùzu nie jest
        thread-safe). Rzuca wyjątek, gdy Kùzu zawiedzie — wołający wchodzi wtedy
        w jawny fallback NetworkX. Zero ścieżek => pusta lista (bez fallbacku).
        """
        if self.conn is None:
            # Wołane tylko gdy połączenie istnieje; defensywnie => jawny fallback u wołającego.
            raise RuntimeError("Kuzu connection unavailable")
        hops = max(1, min(int(max_hops), _MAX_HOPS_LIMIT))
        cypher_query = f"""
            MATCH path = (s:Entity)-[:RELATION*1..{hops}]->(t:Entity)
            WHERE s.name = $source AND t.name = $target
            RETURN path
        """
        result = self.conn.execute(cypher_query, {"source": source, "target": target})
        mapped: List[Dict[str, Any]] = []
        while result.has_next():
            row = result.get_next()
            for path_struct in self._iter_path_structs(row):
                nodes_raw = path_struct.get("_nodes") or []
                names = [str(node.get("name")) for node in nodes_raw if isinstance(node, dict) and "name" in node]
                # Kùzu na grafach z cyklami zwraca także ścieżki z powtórzonymi węzłami;
                # zachowujemy semantykę acykliczną (jak nx.all_simple_paths).
                if len(names) < 2 or len(set(names)) != len(names):
                    continue
                rels_raw = path_struct.get("_rels") or []
                edges_list = []
                for i, rel in enumerate(rels_raw):
                    if i + 1 >= len(names):
                        break
                    edges_list.append({
                        "source": names[i],
                        "predicate": rel.get("predicate", "leads_to"),
                        "target": names[i + 1],
                        "confidence": rel.get("confidence", 1.0),
                    })
                if len(edges_list) != len(names) - 1:
                    continue
                mapped.append({
                    "source": source,
                    "target": target,
                    "nodes": names,
                    "edges": edges_list,
                })
        return mapped

    async def execute_multi_hop_cypher(
        self,
        source: str,
        target: str,
        max_hops: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Wyszukuje wszystkie wieloskokowe ścieżki acykliczne (do max_hops) między source a target.
        Używa Kùzu Cypher jeśli dostępne, w przeciwnym razie grafu NetworkX w pamięci.
        Zwraca listę ścieżek zawierających węzły i krawędzie z wagami pewności.

        V71.2 (G1): wynik zapytania Kùzu jest ZWRACANY (wcześniej trafiał w próżnię).
        NetworkX jest wyłącznie jawnym fallbackiem — gdy ``self.conn is None``
        albo gdy ``execute`` rzuciło wyjątek. Pusty wynik Kùzu => ``[]`` bez fallbacku.
        """
        async with self._lock:
            with self._graph_lock:
                if not (self.nx_graph.has_node(source) and self.nx_graph.has_node(target)):
                    return []

                # 1. Kùzu Cypher — wynik jest mapowany i zwracany (nie wyrzucany)
                if self.conn is not None:
                    try:
                        return self._kuzu_multi_hop_paths(source, target, max_hops)
                    except Exception as exc:
                        logger.debug("Kùzu multi-hop cypher fallback to NetworkX: %s", exc)

                # 2. Jawny fallback NetworkX — tylko gdy Kùzu niedostępne lub zawiodło
                results = []
                try:
                    paths = list(nx.all_simple_paths(self.nx_graph, source=source, target=target, cutoff=max_hops))
                    for path in paths:
                        edges_list = []
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i + 1]
                            edge_data = self.nx_graph.get_edge_data(u, v) or {}
                            edges_list.append({
                                "source": u,
                                "predicate": edge_data.get("predicate", "leads_to"),
                                "target": v,
                                "confidence": edge_data.get("confidence", 1.0),
                            })
                        results.append({
                            "source": source,
                            "target": target,
                            "nodes": path,
                            "edges": edges_list,
                        })
                except Exception as exc:
                    logger.warning("Error computing multi-hop paths: %s", exc)
                    return []

                return results

    async def get_node_relations(self, subject: str, predicate: Optional[str] = None) -> List[Dict[str, Any]]:
        async with self._lock:
            with self._graph_lock:
                if not self.nx_graph.has_node(subject):
                    return []
                results = []
                for _, v, data in self.nx_graph.out_edges(subject, data=True):
                    if predicate is None or data.get("predicate") == predicate:
                        results.append({
                            "subject": subject,
                            "predicate": data.get("predicate"),
                            "object": v,
                            "confidence": data.get("confidence", 1.0),
                            "timestamp": data.get("timestamp", 0.0),
                            "session_id": data.get("session_id"),
                        })
                return results

    def get_all_relations(self) -> List[Dict[str, Any]]:
        """
        V72.1 (G3) + V72.5: eksport krawędzi z pamięci RAM (MultiDiGraph) jako lista dyktów.

        Zwraca DOKŁADNIE 5 pól na wpis — ``subject``, ``predicate``, ``object``,
        ``confidence``, ``timestamp``.

        V72.5: ``nx_graph`` to ``AtlasMultiDiGraph`` (MultiDiGraph), więc zachowuje
        wszystkie predykaty między parą węzłów i w pełni odzwierciedla stan Kùzu.

        Metoda jest sync (jak ``add_relation``), bo czyta wyłącznie NetworkX.
        """
        with self._graph_lock:
            return [
                {
                    "subject": u,
                    "predicate": data.get("predicate", "related_to"),
                    "object": v,
                    "confidence": _as_float(data.get("confidence"), 1.0),
                    "timestamp": _as_float(data.get("timestamp"), 0.0),
                }
                for u, v, data in self.nx_graph.edges(data=True)
            ]

    @staticmethod
    def _fuzzy_needle(name: str) -> str:
        """Normalizacja pod fuzzy-match — TA SAM reguła co w gałęzi RAM (``what_if``)."""
        return str(name).strip().lower().replace("_", " ")

    @staticmethod
    def _fuzzy_needle_regex_body(needle: str) -> str:
        """
        V72.4 (BUG-065): needle -> treść regexu dla Kùzu, równoważna ``_fuzzy_needle``.

        Kùzu 0.11.3 **nie ma funkcji ``replace()``** (sonda na kopii produkcji:
        ``Catalog exception: function REPLACE does not exist``), więc normalizacji
        ``_`` -> spacja nie da się wyrazić w Cypherze funkcją. Zamiast tego każdy znak
        ze stocku (spacja lub podkreślenie) dopuszcza obie postacie: ``[_ ]``, a flagę
        ``(?i)`` dokłada wołający. Dzięki temu porównujemy nadal **znormalizowane**
        nazwy, ale filtracja zostaje na dysku — do Pythona wracają wyłącznie trafienia
        (wcześniej: cały słownik encji, tysiące stringów do sterty interpretera).
        """
        meta = set(".^$*+?()[]{}|\\")
        parts: List[str] = []
        for ch in str(needle):
            if ch in " _":
                parts.append("[_ ]")
            elif ch in meta:
                parts.append("\\" + ch)
            else:
                parts.append(ch)
        return "".join(parts)

    def count_entities(self, name: str) -> int:
        """
        W2 (V72.2): LICZBA węzłów ``Entity`` o danej nazwie w Kùzu (na dysku).

        Najpierw dokładnie parametrzowane dopasowanie nazwy (bez wstrzykiwania do
        stringa), a gdy da zero — fuzzy ``== needle`` / ``needle in nazwa`` liczony
        **w silniku** regexem (V72.4 / BUG-065), z zachowaniem progu ``len(needle) >= 5``
        dla gałęzi podciągowej. Fallback zachowuje semantykę fuzzy-matched z RAM, żeby
        ``what_if`` nie zaczął odrzucać zapytań, które wcześniej przechodziły.

        Zwraca 0, gdy Kùzu jest niedostępne (wtedy wołający opiera się wyłącznie na RAM).
        """
        with self._graph_lock:
            if self.conn is None:
                return 0
            try:
                res = self.conn.execute(
                    "MATCH (e:Entity) WHERE e.name = $name RETURN count(e)", {"name": str(name)}
                )
                rows = self._drain(res)
                exact = int(rows[0][0]) if rows else 0
                if exact:
                    return exact

                needle = self._fuzzy_needle(name)
                if len(needle) < 3:
                    return 0
                body = self._fuzzy_needle_regex_body(needle)
                # Próg 5 znaków jest częścią kontraktu (RAM robi tak samo): krótki
                # stock liczy się WYŁĄCZNIE jako równość, nie jako podciąg.
                rx = ("(?i)" + body) if len(needle) >= 5 else ("(?i)^" + body + "$")
                rows = self._drain(self.conn.execute(
                    "MATCH (e:Entity) WHERE e.name =~ $rx RETURN count(e)", {"rx": rx}
                ))
                return int(rows[0][0]) if rows else 0
            except Exception as exc:
                logger.debug("Kuzu count_entities error (%s): %s", name, exc)
                return 0

    def entity_exists(self, name: str) -> bool:
        """W2 (V72.2): encja istnieje NA DYSKU (Kùzu). RAM to osobna gałąź bramki."""
        return self.count_entities(name) > 0

    def resolve_entity_names_on_disk(self, name: str, limit: int = 10) -> List[str]:
        """Nazwy węzłów z Kùzu pasujące do ``name`` (dokładnie, inaczej fuzzy) — do odczytu ścieżek."""
        with self._graph_lock:
            if self.conn is None:
                return []
            try:
                rows = self._drain(self.conn.execute(
                    "MATCH (e:Entity) WHERE e.name = $name RETURN e.name LIMIT 1", {"name": str(name)}
                ))
                if rows:
                    return [str(r[0]) for r in rows]
                needle = self._fuzzy_needle(name)
                if len(needle) < 3:
                    return []
                body = self._fuzzy_needle_regex_body(needle)
                rx = ("(?i)" + body) if len(needle) >= 5 else ("(?i)^" + body + "$")
                # LIMIT wstawiany jako int() — parametr w LIMIT nie jest wspierany
                # w każdym wariancie Cyphera Kùzu; int() czyni wstrzyknięcie niemożliwym.
                lim = max(1, int(limit))
                rows = self._drain(self.conn.execute(
                    "MATCH (e:Entity) WHERE e.name =~ $rx RETURN e.name LIMIT " + str(lim),
                    {"rx": rx},
                ))
                return [str(r[0]) for r in rows]
            except Exception as exc:
                logger.debug("Kuzu resolve_entity_names error (%s): %s", name, exc)
                return []

    def get_relations_on_disk(self, name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        W2 (V72.2): RZECZYWISTE relacje encji z Kùzu (obejścia i wejścia), bez udziału RAM.

        Format spójny z ``get_all_relations`` (subject/predicate/object/confidence/timestamp).
        Pusta lista, gdy encji nie ma na dysku albo Kùzu zawiodło.
        """
        cap = max(1, min(int(limit), 500))
        if self.conn is None:
            return []
        names = self.resolve_entity_names_on_disk(name)
        if not names:
            return []
        out: List[Dict[str, Any]] = []
        with self._graph_lock:
            for resolved in names:
                try:
                    rows = self._drain(self.conn.execute(
                        f"""
                        MATCH (a:Entity)-[r:RELATION]->(b:Entity)
                        WHERE a.name = $name OR b.name = $name
                        RETURN a.name, r.predicate, b.name, r.confidence, r.timestamp
                        LIMIT {cap}
                        """,
                        {"name": resolved},
                    ))
                except Exception as exc:
                    logger.debug("Kuzu get_relations_on_disk error (%s): %s", resolved, exc)
                    continue
                for subj, pred, obj, conf, ts in rows:
                    out.append({
                        "subject": str(subj),
                        "predicate": str(pred) if pred is not None else "related_to",
                        "object": str(obj),
                        "confidence": _as_float(conf, 1.0),
                        "timestamp": _as_float(ts, 0.0),
                    })
                    if len(out) >= cap:
                        return out
        return out

    def personalized_pagerank(
        self,
        personalization: Optional[Dict[str, float]] = None,
        alpha: float = 0.85,
    ) -> Dict[str, float]:
        """Compute Personalized PageRank over the in-memory causal graph."""
        with self._graph_lock:
            if self.nx_graph.number_of_nodes() == 0:
                return {}
            return nx.pagerank(self.nx_graph, alpha=alpha, personalization=personalization)
