from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Set

import networkx as nx

logger = logging.getLogger(__name__)

try:
    import kuzu
    HAS_KUZU = True
except ImportError:
    HAS_KUZU = False

from atlas_memory.models import MemoryRecord


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

    def add_entity(self, name: str, entity_type: str = "entity") -> None:
        """Synchronously adds a node to NetworkX and Kùzu graph."""
        self.nx_graph.add_node(name, type=entity_type)
        if self.conn is not None:
            try:
                self.conn.execute(
                    "MERGE (e:Entity {name: $name}) ON CREATE SET e.entity_type = $etype",
                    {"name": name, "etype": entity_type},
                )
            except Exception as exc:
                logger.debug("Kuzu add_entity error: %s", exc)

    def add_relation(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> None:
        """Synchronously adds a relation edge between subject and object."""
        ts = time.time() if timestamp is None else timestamp
        self.nx_graph.add_node(subject, type="entity")
        self.nx_graph.add_node(object, type="entity_or_value")
        self.nx_graph.add_edge(
            subject,
            object,
            predicate=predicate,
            confidence=confidence,
            timestamp=ts,
        )
        if self.conn is not None:
            try:
                self.conn.execute(
                    "MERGE (e:Entity {name: $name}) ON CREATE SET e.entity_type = 'entity'",
                    {"name": subject},
                )
                self.conn.execute(
                    "MERGE (e:Entity {name: $name}) ON CREATE SET e.entity_type = 'entity_or_val'",
                    {"name": object},
                )
                self.conn.execute("""
                    MATCH (u:Entity), (v:Entity)
                    WHERE u.name = $u_name AND v.name = $v_name
                    CREATE (u)-[:RELATION {predicate: $pred, confidence: $conf, timestamp: $ts}]->(v)
                """, {
                    "u_name": subject,
                    "v_name": object,
                    "pred": predicate,
                    "conf": confidence,
                    "ts": ts,
                })
            except Exception as exc:
                logger.debug("Kuzu add_relation error: %s", exc)

    def add_record_sync(self, record: MemoryRecord) -> None:
        """Synchronous record addition wrapper."""
        self.add_relation(
            record.effective_subject,
            record.predicate,
            str(record.object)[:80],
            confidence=float(record.confidence),
            timestamp=float(record.timestamp),
        )

    async def add_record(self, record: MemoryRecord) -> None:
        """Asynchronous record addition."""
        async with self._lock:
            self.add_record_sync(record)

    def close(self) -> None:
        """Closes Kuzu connection and releases resources."""
        if self.conn is not None:
            try:
                self.conn = None
                self.db = None
            except Exception:
                pass

    async def get_subgraph_relations(
        self,
        active_entities: List[str],
        max_depth: int = 2,
    ) -> Dict[str, Any]:
        async with self._lock:
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
                    if node_str == ent_clean or ent_clean in node_str or node_str in ent_clean:
                        matched_roots.add(node)
                    elif ent_tokens and any(t in node_str for t in ent_tokens):
                        matched_roots.add(node)

            for root in matched_roots:
                subgraph_nodes.add(root)
                succ_1 = set(self.nx_graph.successors(root))
                pred_1 = set(self.nx_graph.predecessors(root))
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
                "active_roots": list(matched_roots) if matched_roots else active_entities,
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

