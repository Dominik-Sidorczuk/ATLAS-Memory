from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, Optional

from atlas_memory.core.models import MemoryRecord
from atlas_memory.l2_semantic.kuzu.graph_queries import _MAX_HOPS_LIMIT, KuzuGraphQueriesMixin
from atlas_memory.l2_semantic.kuzu.ram_multigraph import (
    AtlasMultiDiGraph,
    KuzuRamHydrationMixin,
    MultiEdgeData,
)
from atlas_memory.l2_semantic.kuzu.schema_driver import (
    HAS_KUZU,
    KUZU_BUFFER_MB_DEFAULT,
    KUZU_BUFFER_POOL_BYTES,
    KuzuSchemaDriverMixin,
    KuzuWriteError,
    _as_float,
)

logger = logging.getLogger(__name__)

if HAS_KUZU:
    import kuzu


class KuzuGraphStore(KuzuSchemaDriverMixin, KuzuRamHydrationMixin, KuzuGraphQueriesMixin):
    """
    L2: Wbudowana Dyskowa Baza Grafowa Kùzu (Cypher) + NetworkX z kontrolą pamięci RAM.
    Fasada delegująca DDL/migracje do KuzuSchemaDriverMixin, hydrację RAM do KuzuRamHydrationMixin
    oraz zapytania Cypher/PPR do KuzuGraphQueriesMixin.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        buffer_pool_size_bytes: int = KUZU_BUFFER_POOL_BYTES,
    ):
        self._lock = asyncio.Lock()
        # V71.2 (G3): blokada dla mutacji/odczytów nx_graph i dostępu do self.conn.
        # Kolejność blokad ZAWSZE: self._lock (asyncio) -> self._graph_lock. Nigdy odwrotnie.
        self._graph_lock = threading.RLock()
        self.nx_graph = AtlasMultiDiGraph()
        self.nx_graph._lock = self._graph_lock
        self.db_path = db_path
        self.buffer_pool_size = buffer_pool_size_bytes
        self._temp_dir = None

        if HAS_KUZU:
            if not db_path or db_path == ":memory:":
                self._temp_dir = tempfile.mkdtemp(prefix="kuzu_mem_")
                self.db_path = os.path.join(self._temp_dir, "kuzu_db")

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
                if self._temp_dir and os.path.exists(self._temp_dir):
                    shutil.rmtree(self._temp_dir, ignore_errors=True)
                    self._temp_dir = None
        else:
            self.db = None
            self.conn = None


    def create_entity(self, name: str, entity_type: str = "entity") -> None:
        """V104.5 (FIX-1): Ensures an Entity node exists before linking CONTAINS/SYNONYM.

        V124 (F-041): parameter renamed from ``kind`` to ``entity_type`` to
        match the ``DualNodeGraphStoreProtocol`` contract (core/protocols.py).
        Positional call sites are unaffected; kwarg callers used ``kind=``
        only inside tests (aligned in the same commit).
        """
        self.add_entity(name=name, entity_type=entity_type)

    def add_entity(self, name: str, entity_type: str = "entity") -> None:
        """Synchronously adds a node to NetworkX and Kùzu graph."""
        with self._graph_lock:
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
        session_id: Optional[str] = None,
    ) -> None:
        """
        Synchronously adds a relation edge between subject and object.

        V72.1 (G1): ta sama trojka (subject, predicate, object) ma w Kùzu DOKŁADNIE
        jeden rekord RELATION — zapis idzie przez ``MERGE`` na związku, a powtórzenie
        aktualizuje ``timestamp``/``confidence`` na MAX(stara, nowa) (nigdy nie
        obniża). Różne predykaty między tą samą parą pozostają osobnymi rekordami.
        NetworkX (RAM) zachowuje dotychczasową semantykę jednej krawędzi na parę.
        """
        ts = time.time() if timestamp is None else timestamp
        with self._graph_lock:
            self.nx_graph.add_node(subject, type="entity")
            self.nx_graph.add_node(object, type="entity_or_value")
            # V72.5: MultiDiGraph trzyma OSOBNĄ krawędź dla każdego predykatu (key=predicate).
            # Powtórzenie tej samej trójki relacyjnej scala atrybuty w górę: MAX(stare, nowe).
            prev = self.nx_graph.get_edge_data(subject, object, key=predicate) or {}
            prev_conf = prev.get("confidence")
            prev_ts = prev.get("timestamp")
            merged_conf = float(prev_conf) if isinstance(prev_conf, (int, float)) and prev_conf > confidence else confidence
            merged_ts = float(prev_ts) if isinstance(prev_ts, (int, float)) and prev_ts > ts else ts
            self.nx_graph.add_edge(
                subject,
                object,
                key=predicate,
                predicate=predicate,
                confidence=merged_conf,
                timestamp=merged_ts,
                session_id=session_id,
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
                    # V72.1 (G1): MERGE na związku z guardem MAX — powtórzenie trojki
                    # aktualizuje tylko w górę (CASE WHEN), więc confidence/timestamp
                    # nigdy nie regresują. Zweryfikowane sondą na kuzu 0.11.3.
                    self.conn.execute("""
                        MATCH (u:Entity), (v:Entity)
                        WHERE u.name = $u_name AND v.name = $v_name
                        MERGE (u)-[r:RELATION {predicate: $pred}]->(v)
                        ON CREATE SET r.confidence = $conf, r.timestamp = $ts
                        ON MATCH SET
                            r.confidence = CASE WHEN $conf > r.confidence THEN $conf ELSE r.confidence END,
                            r.timestamp  = CASE WHEN $ts > r.timestamp THEN $ts ELSE r.timestamp END
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
        sess_id = getattr(record, "source_session_id", None) or (
            record.metadata.get("session_id") if isinstance(record.metadata, dict) else None
        )
        with self._graph_lock:
            self.add_relation(
                record.effective_subject,
                record.predicate,
                str(record.object)[:80],
                confidence=float(record.confidence),
                timestamp=float(record.timestamp),
                session_id=sess_id,
            )

    async def add_record(self, record: MemoryRecord) -> None:
        """Asynchronous record addition."""
        async with self._lock:
            self.add_record_sync(record)

    def close(self) -> Dict[str, Any]:
        """Zamyka Kùzu PO scaleniu WAL-a i zwraca prawdomówny raport zamknięcia.

        Defekt naprawiany tutaj (zmierzony 2026-09-13 na kopii produkcyjnej bazy):
        ``close()`` tylko zrzucało referencje, więc WAL (2,6 MB) nigdy nie był
        scalany. Kùzu 0.11.3 odtwarza WAL przy otwarciu i pinuje z tego tytułu
        ~0,4-0,5 GB puli buforów: przy 512 MB pada już ``MATCH ()-[r:RELATION]->()
        RETURN count(r)`` ("buffer pool is full and no memory could be freed"), a
        hydracja zwraca 0/0 z ``partial=True`` — przy następnym starcie cykl się
        powtarza i WAL rośnie dalej. Z progiem 1024 MB ten sam ``CHECKPOINT`` i skan
        przechodzą (2518 węzłów / 2181 RELATION), a plik WAL znika.

        Kolejność i bramka są istotne: ``CHECKPOINT`` przy wyczerpanej puli potrafi
        wywrócić proces SIGSEGV-em w ``db.close()`` (zmierzone: 1 z 2 wariantów),
        dlatego po nieudanym checkpointcie **nie** zamykamy ``db`` — zrzucamy tylko
        uchwyty i raportujemy powód (ADR-002: brak wartości = jawne pole + powód,
        nigdy ciche zero).
        """
        report: Dict[str, Any] = {
            "checkpointed": False,
            "db_closed": False,
            "reason": None,
            "pool_bytes": int(getattr(self, "buffer_pool_size", 0) or 0),
        }
        if self.conn is None:
            report["reason"] = "no_connection"
        else:
            try:
                self.conn.execute("CHECKPOINT")
                report["checkpointed"] = True
            except Exception as exc:
                report["reason"] = f"checkpoint_failed: {type(exc).__name__}: {str(exc)[:120]}"
        if report["checkpointed"] and getattr(self, "db", None) is not None:
            try:
                self.db.close()
                report["db_closed"] = True
            except Exception as exc:
                report["reason"] = f"db_close_failed: {type(exc).__name__}: {str(exc)[:120]}"
        self.conn = None
        self.db = None
        self.last_close_report = report
        if getattr(self, "_temp_dir", None) and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
        if report["db_closed"] or report["reason"] not in (None, "no_connection"):
            logger.info("KuzuGraphStore.close: %s", report)
        else:
            logger.debug("KuzuGraphStore.close (noop): %s", report)
        return report

    def __del__(self) -> None:
        self.close()


KuzuCausalGraph = KuzuGraphStore

__all__ = [
    "AtlasMultiDiGraph",
    "HAS_KUZU",
    "KUZU_BUFFER_MB_DEFAULT",
    "KUZU_BUFFER_POOL_BYTES",
    "KuzuCausalGraph",
    "KuzuGraphStore",
    "KuzuWriteError",
    "MultiEdgeData",
    "_MAX_HOPS_LIMIT",
    "_as_float",
]
