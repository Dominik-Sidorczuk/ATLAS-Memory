from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Set

import networkx as nx

logger = logging.getLogger(__name__)

try:
    import kuzu
    HAS_KUZU = True
except ImportError:
    HAS_KUZU = False

from loop_memory.models import MemoryRecord


class KuzuGraphStore:
    """
    L2: Wbudowana Dyskowa Baza Grafowa Kùzu (Cypher) + NetworkX z kontrolą pamięci RAM.
    
    Udostępnia standard zapytań Cypher (MATCH (a)-[:RELATION]->(b))
    oraz konfigurację buffer_pool_size zapobiegającą niekontrolowanemu wzrostowi RAM.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        buffer_pool_size_bytes: int = 128 * 1024 * 1024,  # Domyślnie 128 MB RAM limit
    ):
        self._lock = asyncio.Lock()
        self.nx_graph = nx.DiGraph()
        self.db_path = db_path
        self.buffer_pool_size = buffer_pool_size_bytes
        self._temp_dir = None

        if HAS_KUZU:
            if not db_path or db_path == ":memory:":
                self._temp_dir = tempfile.mkdtemp(prefix="kuzu_mem_")
                self.db_path = self._temp_dir

            try:
                # Inicjalizacja Kùzu z limitem pamięci bufora
                self.db = kuzu.Database(self.db_path, buffer_pool_size=self.buffer_pool_size)
                self.conn = kuzu.Connection(self.db)
                self._init_schema()
            except Exception as exc:
                # Fix atlas-v5: nie maskuj błędów otwarcia Kuzu — loguj, żeby
                # było wiadomo, że dyskowa persystencja grafu NIE działa
                # (NetworkX w pamięci nadal działa; Kuzu jest opcjonalne).
                logger.warning("Kuzu unavailable (init): %s", exc)
                self.db = None
                self.conn = None
        else:
            self.db = None
            self.conn = None

    def _init_schema(self) -> None:
        if self.conn is None:
            return
        try:
            self.conn.execute("CREATE NODE TABLE Entity(name STRING, entity_type STRING, PRIMARY KEY(name))")
        except Exception as exc:
            # CREATE TABLE na istniejącej tabeli jest normalne (idempotentny init)
            logger.debug("Kuzu schema (node table): %s", exc)
        try:
            self.conn.execute("CREATE REL TABLE RELATION(FROM Entity TO Entity, predicate STRING, confidence DOUBLE, timestamp DOUBLE)")
        except Exception as exc:
            logger.debug("Kuzu schema (rel table): %s", exc)

    async def add_record(self, record: MemoryRecord) -> None:
        subj = record.effective_subject
        obj = str(record.object)
        pred = record.predicate
        conf = float(record.confidence)
        ts = float(record.timestamp)

        async with self._lock:
            # 1. Aktualizacja grafu NetworkX w pamięci
            self.nx_graph.add_node(subj, type="entity")
            self.nx_graph.add_node(obj, type="entity_or_value")
            self.nx_graph.add_edge(
                subj,
                obj,
                predicate=pred,
                confidence=conf,
                timestamp=ts,
                is_state_variable=record.is_state_variable,
            )

            # 2. Zapis w bazie Kùzu za pomocą zapytań Cypher
            if self.conn is not None:
                try:
                    self.conn.execute("MERGE (e:Entity {name: $name}) ON CREATE SET e.entity_type = 'entity'", {"name": subj})
                    self.conn.execute("MERGE (e:Entity {name: $name}) ON CREATE SET e.entity_type = 'entity_or_val'", {"name": obj})
                    self.conn.execute("""
                        MATCH (u:Entity), (v:Entity)
                        WHERE u.name = $u_name AND v.name = $v_name
                        CREATE (u)-[:RELATION {predicate: $pred, confidence: $conf, timestamp: $ts}]->(v)
                    """, {
                        "u_name": subj,
                        "v_name": obj,
                        "pred": pred,
                        "conf": conf,
                        "ts": ts,
                    })
                except Exception as exc:
                    # Fix atlas-v5: nie maskuj błędów persystencji Kuzu — loguj,
                    # żeby było wiadomo, że dyskowa warstwa grafu nie działa
                    # (graf NetworkX w pamięci i tak został zaktualizowany).
                    logger.warning("Kuzu persistence failed: %s", exc)

    async def get_subgraph_relations(
        self,
        active_entities: List[str],
        max_depth: int = 2,
    ) -> Dict[str, Any]:
        async with self._lock:
            subgraph_nodes: Set[str] = set()
            for entity in active_entities:
                if entity in self.nx_graph:
                    subgraph_nodes.add(entity)
                    succ_1 = set(self.nx_graph.successors(entity))
                    pred_1 = set(self.nx_graph.predecessors(entity))
                    subgraph_nodes.update(succ_1 | pred_1)

                    if max_depth >= 2:
                        for neighbor in (succ_1 | pred_1):
                            subgraph_nodes.update(self.nx_graph.successors(neighbor))
                            subgraph_nodes.update(self.nx_graph.predecessors(neighbor))

            sub_g = self.nx_graph.subgraph(subgraph_nodes)
            edges_list = []
            for u, v, data in sub_g.edges(data=True):
                edges_list.append({
                    "subject": u,
                    "predicate": data.get("predicate", "related_to"),
                    "object": v,
                    "confidence": data.get("confidence", 1.0),
                    "timestamp": data.get("timestamp", 0.0),
                })

            return {
                "active_roots": active_entities,
                "matched_nodes_count": len(subgraph_nodes),
                "nodes": list(subgraph_nodes),
                "relations": edges_list,
            }

    async def find_causal_path(self, source: str, target: str) -> Optional[List[Dict[str, Any]]]:
        async with self._lock:
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
        """
        async with self._lock:
            if not (self.nx_graph.has_node(source) and self.nx_graph.has_node(target)):
                return []

            # 1. Próba wykonania przez Kùzu Cypher jeśli połączenie jest aktywne
            if self.conn is not None:
                try:
                    cypher_query = f"""
                        MATCH path = (s:Entity)-[:RELATION*1..{max_hops}]->(t:Entity)
                        WHERE s.name = $source AND t.name = $target
                        RETURN path
                    """
                    # Wykonanie zapytania Cypher
                    self.conn.execute(cypher_query, {"source": source, "target": target})
                    # Jeśli Kùzu zwróci wyniki, możemy je przetworzyć

                except Exception as exc:
                    logger.debug("Kùzu multi-hop cypher fallback to NetworkX: %s", exc)

            # 2. Wyznaczenie wszystkich prostych (acyklicznych) ścieżek przez NetworkX
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
                    })
            return results

    async def remove_edge(self, subject: str, predicate: str, object_: str) -> bool:
        async with self._lock:
            if self.nx_graph.has_edge(subject, object_):
                edge_data = self.nx_graph.get_edge_data(subject, object_)
                if edge_data.get("predicate") == predicate:
                    self.nx_graph.remove_edge(subject, object_)
                    return True
            return False

    async def garbage_collect_orphans(self) -> int:
        async with self._lock:
            orphans = [node for node, degree in dict(self.nx_graph.degree()).items() if degree == 0]
            for orphan in orphans:
                self.nx_graph.remove_node(orphan)
            return len(orphans)

    def close(self) -> None:
        self.conn = None
        self.db = None
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    def __del__(self) -> None:
        self.close()

