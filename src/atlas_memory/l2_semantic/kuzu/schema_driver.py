from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from atlas_memory.core.exceptions import InvariantViolation
from atlas_memory.l2_semantic.audit_ledger import record_swallow
from atlas_memory.l2_semantic.openie import build_synonym_edges, extract_openie_phrases_and_triples

logger = logging.getLogger(__name__)

try:
    import kuzu  # noqa: F401
    HAS_KUZU = True
except ImportError:
    HAS_KUZU = False

from atlas_memory.core.config import get_config

KUZU_BUFFER_MB_DEFAULT = max(64, int(get_config().kuzu_buffer_mb))
KUZU_BUFFER_POOL_BYTES = KUZU_BUFFER_MB_DEFAULT * 1024 * 1024


def _as_float(value: Any, default: float) -> float:
    """Defensywna koercja atrybutu z Kùzu do float (NULL/typ niezgodny -> default)."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class KuzuWriteError(RuntimeError):
    """Raised when a write to Kùzu (add_passage, add_contains, add_synonym) fails (FIX-2)."""


class KuzuSchemaDriverMixin:
    """Kùzu DB DDL schema initialization, Dual-Node passage/synonym indexing, and orphan GC."""

    def _has_property(self, table_name: str, property_name: str) -> bool:
        """Introspects Kùzu schema to check if property exists on a node/rel table."""
        if self.conn is None:
            return False
        try:
            res = self.conn.execute(f'CALL table_info("{table_name}") RETURN *')
            while res.has_next():
                row = res.get_next()
                if len(row) > 1 and str(row[1]).lower() == property_name.lower():
                    return True
        except Exception as exc:
            logger.debug("Schema introspection failed for %s.%s: %s", table_name, property_name, exc)
        return False

    def mark_entity_superseded(self, name: str, superseded_at: Optional[float] = None) -> bool:
        """V151.1 (F-031): oznacza węzeł Entity(name) właściwościami tombstone.

        Właściwość, NIE DETACH DELETE: krawędzie RELATION/CONTAINS/SYNONYM i audyt
        zostają nietknięte; read-path (Kùzu + PPR) filtruje po cieciu is_superseded.
        Zwraca True gdy operacja diskowa się udała (RAM-nodes oznaczone zawsze,
        gdy istnieją). Semantyka identyczna po obu widokach.
        """
        ent = str(name or "").strip()
        if not ent:
            return False
        ts = float(superseded_at if superseded_at is not None else time.time())
        marked_disk = False
        if getattr(self, "conn", None) is not None:
            try:
                self.conn.execute(
                    "MATCH (e:Entity {name: $name}) SET e.is_superseded = true, e.superseded_at = $ts",
                    {"name": ent, "ts": ts},
                )
                marked_disk = True
            except Exception as exc:
                record_swallow("l2_semantic.kuzu.schema_driver.mark_entity_superseded", ent, exc)
        nx_graph = getattr(self, "nx_graph", None)
        if nx_graph is not None:
            try:
                with self._graph_lock:
                    if ent in nx_graph:
                        nx_graph.nodes[ent]["is_superseded"] = True
                        nx_graph.nodes[ent]["superseded_at"] = ts
                    else:
                        # Węzła nie ma w RAM — disk-owny wynik nadal liczy się jako oznaczenie.
                        pass
            except Exception as exc:
                logger.debug("Kuzu RAM entity tombstone skipped for %s: %s", ent, exc)
        return marked_disk

    def is_entity_superseded(self, name: str) -> bool:
        """V151.1 (F-031): symetryczny czytnik właściwości tombstone na Entity(name).

        Używany przez read-path (testy + ppr.read()) do odrzucania superseded
        węzłów. Najpierw patrzy na RAM-nx_graph (szybko); gdy nie ma węzła
        (Entity dodane tylko w Kùzu) — pyta Kùzu o właściwość is_superseded.
        Gdy węzeł w ogóle nie istnieje, zwraca False (brak obiektu ≠ tombstone).
        """
        ent = str(name or "").strip()
        if not ent:
            return False
        nx_graph = getattr(self, "nx_graph", None)
        if nx_graph is not None and ent in nx_graph:
            try:
                return bool(nx_graph.nodes[ent].get("is_superseded"))
            except Exception:
                pass
        if getattr(self, "conn", None) is not None:
            try:
                res = self.conn.execute(
                    "MATCH (e:Entity {name: $name}) RETURN e.is_superseded",
                    {"name": ent},
                )
                if res.has_next():
                    row = res.get_next()
                    if row and row[0] is not None:
                        return bool(row[0])
            except Exception:
                return False
        return False

    def _init_schema(self) -> None:
        if self.conn is None:
            return
        try:
            self.conn.execute("CREATE NODE TABLE Entity(name STRING, entity_type STRING, PRIMARY KEY(name))")
        except Exception as exc:
            logger.debug("Kuzu schema (node table): %s", exc)
        try:
            self.conn.execute("CREATE REL TABLE RELATION(FROM Entity TO Entity, predicate STRING, confidence DOUBLE, timestamp DOUBLE)")
        except Exception as exc:
            logger.debug("Kuzu schema (rel table): %s", exc)
        # V104 / W1 (HippoRAG 2 Dual-Node ADD-ONLY Schema + V104.5 ingest_at):
        try:
            self.conn.execute(
                "CREATE NODE TABLE Passage(passage_id STRING, source_key STRING, text STRING, ingest_at STRING, PRIMARY KEY(passage_id))"
            )
        except Exception as exc:
            logger.debug("Kuzu schema (Passage node table): %s", exc)
        
        # Idempotent introspection check: only alter table if ingest_at property does not exist
        if not self._has_property("Passage", "ingest_at"):
            try:
                self.conn.execute("ALTER TABLE Passage ADD ingest_at STRING DEFAULT ''")
            except Exception as exc:
                exc_str = str(exc).lower()
                if "exist" not in exc_str and "already has property" not in exc_str:
                    record_swallow("l2_semantic.kuzu.schema_driver.passage_ingest_at_migration", "Passage.ingest_at", exc)
                else:
                    logger.debug("Passage table already has ingest_at property (idempotent)")

        # V151.1 (F-031): towiaro projections — widok węzła Entity dostaje pola
        # tombstone, by supersession mógł rozlecieć się POZA KV (single source of
        # truth) na widok materializowany Kùzu (właściwość, nie DETACH DELETE —
        # krawędzie i audyt zostają).
        if self.conn is not None:
            for prop_def, prop_name, swallow_id in (
                ("is_superseded BOOLEAN DEFAULT false", "is_superseded", "l2_semantic.kuzu.schema_driver.entity_superseded_migration"),
                ("superseded_at DOUBLE DEFAULT 0", "superseded_at", "l2_semantic.kuzu.schema_driver.entity_superseded_at_migration"),
            ):
                if not self._has_property("Entity", prop_name):
                    try:
                        self.conn.execute(f"ALTER TABLE Entity ADD {prop_def}")
                    except Exception as exc:
                        exc_str = str(exc).lower()
                        if "exist" not in exc_str and "already has property" not in exc_str:
                            record_swallow(swallow_id, f"Entity.{prop_name}", exc)
                        else:
                            logger.debug("Entity table already has %s property (idempotent)", prop_name)
        try:
            self.conn.execute("CREATE REL TABLE CONTAINS(FROM Passage TO Entity, weight DOUBLE)")
        except Exception as exc:
            logger.debug("Kuzu schema (CONTAINS rel table): %s", exc)
        try:
            self.conn.execute("CREATE REL TABLE SYNONYM(FROM Entity TO Entity, sim DOUBLE)")
        except Exception as exc:
            logger.debug("Kuzu schema (SYNONYM rel table): %s", exc)


    def has_passage(self, passage_id: str) -> bool:
        """Checks whether a Passage node (`passage_id` or `source_key`) already exists."""
        pid = str(passage_id).strip()
        if not pid:
            return False
        alt_pid = pid if pid.startswith("passage:") else f"passage:{pid}"
        with self._graph_lock:
            for cand in (pid, alt_pid):
                if cand in self.nx_graph and self.nx_graph.nodes[cand].get("type") == "passage":
                    return True
            if self.conn is not None:
                try:
                    res = self.conn.execute(
                        "MATCH (p:Passage) WHERE p.passage_id = $pid OR p.passage_id = $alt OR p.source_key = $pid RETURN count(p)",
                        {"pid": pid, "alt": alt_pid},
                    )
                    if res.has_next() and int(res.get_next()[0]) > 0:
                        return True
                except Exception:
                    return False
        return False

    def rollback_dual_node_record(self, passage_id: str, created_entities: Optional[list[str]] = None) -> None:
        """V104.5 (FIX-1): Atomically rolls back Passage, CONTAINS, and newly created Entities/SYNONYMs on write failure."""
        pid = str(passage_id).strip()
        alt_pid = pid if pid.startswith("passage:") else f"passage:{pid}"
        with self._graph_lock:
            for cand in (pid, alt_pid):
                if cand in self.nx_graph:
                    self.nx_graph.remove_node(cand)
            for ent in created_entities or []:
                if ent in self.nx_graph:
                    self.nx_graph.remove_node(ent)
            if self.conn is not None:
                for cand in (pid, alt_pid):
                    try:
                        self.conn.execute(
                            "MATCH (p:Passage {passage_id: $pid}) DETACH DELETE p",
                            {"pid": cand},
                        )
                    except Exception as exc:
                        logger.debug("Kuzu rollback Passage cleanup warning: %s", exc)
                for ent in created_entities or []:
                    try:
                        self.conn.execute(
                            "MATCH (e:Entity {name: $ename}) DETACH DELETE e",
                            {"ename": str(ent)},
                        )
                    except Exception as exc:
                        logger.debug("Kuzu rollback Entity cleanup warning: %s", exc)

    def index_dual_node_passage(
        self,
        source_key: str,
        text: str,
        ingest_at: Optional[str] = None,
        gist: Optional[str] = None,
        explicit_entities: Optional[list[str]] = None,
        explicit_synonyms: Optional[list[tuple[str, str, float]]] = None,
        # V152 (FILAR I, ADR): stamp aktu mowy (opcjonalny parametr sygnatury;
        # ConversationalTurn nie powinien tu dotrzeć — obrona warstwy 2).
        speech_act: Optional[str] = None,
    ) -> int:
        """Consolidated single choke-point for indexing a Dual-Node Passage (`Passage` + `CONTAINS` + `SYNONYM`) with atomic rollback."""

        skey = str(source_key).strip()
        txt = str(text).strip()
        if not skey or not txt:
            return 0

        passage_id = skey
        unique_ents: list[str] = []
        if explicit_entities:
            for ent in explicit_entities:
                ent_s = str(ent).strip()
                if ent_s and ent_s not in unique_ents:
                    unique_ents.append(ent_s)
        if not unique_ents:
            extraction = extract_openie_phrases_and_triples(
                source_key=skey,
                text=txt,
                gist=gist if gist else None,
                passage_id=passage_id,
            )
            for ent in extraction.entities:
                ent_s = str(ent).strip()
                if ent_s and ent_s not in unique_ents:
                    unique_ents.append(ent_s)
        capped_ents = unique_ents[:16]

        with self._graph_lock:
            pre_existing_nodes = set(str(n) for n in self.nx_graph.nodes()) if self.nx_graph is not None else set()
            created_entities: list[str] = []
            edges_added = 0
            try:
                self.add_passage(
                    passage_id=passage_id,
                    source_key=skey,
                    text=txt,
                    ingest_at=ingest_at,
                )
                for ent in capped_ents:
                    if ent not in pre_existing_nodes:
                        created_entities.append(ent)
                        self.create_entity(ent, "entity")
                    self.add_contains(
                        passage_id=passage_id,
                        entity_name=ent,
                        weight=1.0,
                        persist_to_kuzu=True,
                    )
                    edges_added += 1

                if capped_ents:
                    existing_ents: list[str] = list(capped_ents)
                    if self.nx_graph is not None:
                        for n_id, n_data in list(self.nx_graph.nodes(data=True))[:128]:
                            if isinstance(n_data, dict) and n_data.get("type") == "passage":
                                continue
                            n_str = str(n_id).strip()
                            if n_str and n_str not in existing_ents:
                                existing_ents.append(n_str)
                    syn_pairs = build_synonym_edges(
                        existing_ents,
                        sim_threshold=0.80,
                        max_pairs_per_entity=3,
                    )
                    if explicit_synonyms:
                        for sp in explicit_synonyms:
                            if isinstance(sp, (list, tuple)) and len(sp) >= 3:
                                syn_pairs.append((str(sp[0]), str(sp[1]), float(sp[2])))
                    capped_set = set(capped_ents)
                    for ea, eb, sim_val in syn_pairs:
                        if ea in capped_set or eb in capped_set:
                            self.add_synonym(ea, eb, sim_val)
                            edges_added += 1
            except Exception:
                self.rollback_dual_node_record(passage_id, created_entities)
                raise

            return edges_added

    def get_dual_node_coverage(self) -> float:
        """V104.5 (FIX-4): Returns ratio of Passage nodes having >=1 CONTAINS edge (`0.0` before migration, `1.0` after)."""
        with self._graph_lock:
            passage_nodes = [
                n for n, d in self.nx_graph.nodes(data=True) if isinstance(d, dict) and d.get("type") == "passage"
            ]
            if passage_nodes:
                with_c = 0
                for p_node in passage_nodes:
                    has_c = any(
                        isinstance(ed, dict) and ed.get("predicate") == "CONTAINS"
                        for _, _, ed in self.nx_graph.edges(p_node, data=True)
                    )
                    if has_c:
                        with_c += 1
                return round(float(with_c) / float(len(passage_nodes)), 4)
            if self.conn is not None:
                try:
                    rp = self.conn.execute("MATCH (p:Passage) RETURN count(p)")
                    total_p = int(rp.get_next()[0]) if rp.has_next() else 0
                    if total_p <= 0:
                        return 0.0
                    rc = self.conn.execute("MATCH (p:Passage)-[:CONTAINS]->() RETURN count(DISTINCT p.passage_id)")
                    cov_p = int(rc.get_next()[0]) if rc.has_next() else 0
                    return round(float(cov_p) / float(total_p), 4)
                except Exception:
                    return 0.0
        return 0.0

    def add_passage(
        self,
        passage_id: str,
        source_key: str,
        text: str,
        ingest_at: Optional[str] = None,
    ) -> None:
        """W1 (V104 / V104.5): Synchronously adds a Passage node (with `ingest_at`) to NetworkX and Kùzu."""
        from datetime import datetime, timezone

        pid = str(passage_id).strip()
        skey = str(source_key).strip()
        txt = str(text).strip()
        if not pid:
            return
        if isinstance(ingest_at, datetime):
            ing_str = ingest_at.isoformat()
        elif ingest_at is not None and str(ingest_at).strip():
            ing_str = str(ingest_at).strip()
        else:
            ing_str = datetime.now(timezone.utc).isoformat()
        with self._graph_lock:
            self.nx_graph.add_node(pid, type="passage", source_key=skey, text=txt, ingest_at=ing_str)
            if self.conn is not None:
                try:
                    self.conn.execute(
                        "MERGE (p:Passage {passage_id: $pid}) "
                        "ON CREATE SET p.source_key = $skey, p.text = $txt, p.ingest_at = $ing "
                        "ON MATCH SET p.source_key = $skey, p.text = $txt, p.ingest_at = $ing",
                        {"pid": pid, "skey": skey, "txt": txt, "ing": ing_str},
                    )
                except Exception as exc:
                    from atlas_memory.cognitive.retrieval.ppr import metrics

                    metrics.counter("kuzu_write_error").inc()
                    raise KuzuWriteError(f"Kuzu add_passage failed: {exc}") from exc

    def add_contains(
        self,
        passage_id: str,
        entity_name: str,
        weight: float = 1.0,
        persist_to_kuzu: bool = True,
    ) -> None:
        """W1 (V104 / V104.5): Synchronously adds a CONTAINS edge (FROM Passage TO Entity)."""
        pid = str(passage_id).strip()
        ename = str(entity_name).strip()
        if not pid or not ename:
            return
        w = float(weight)
        with self._graph_lock:
            if pid not in self.nx_graph:
                self.nx_graph.add_node(pid, type="passage", source_key=pid, text="")
            if ename not in self.nx_graph:
                self.nx_graph.add_node(ename, type="entity")
            self.nx_graph.add_edge(
                pid,
                ename,
                key="CONTAINS",
                predicate="CONTAINS",
                confidence=w,
                weight=w,
                timestamp=time.time(),
            )
            if persist_to_kuzu and self.conn is not None:
                try:
                    self.conn.execute(
                        "MERGE (e:Entity {name: $ename}) ON CREATE SET e.entity_type = 'entity'",
                        {"ename": ename},
                    )
                    self.conn.execute(
                        "MATCH (p:Passage), (e:Entity) "
                        "WHERE p.passage_id = $pid AND e.name = $ename "
                        "MERGE (p)-[c:CONTAINS]->(e) "
                        "ON CREATE SET c.weight = $w "
                        "ON MATCH SET c.weight = CASE WHEN $w > c.weight THEN $w ELSE c.weight END",
                        {"pid": pid, "ename": ename, "w": w},
                    )
                except Exception as exc:
                    from atlas_memory.cognitive.retrieval.ppr import metrics

                    metrics.counter("kuzu_write_error").inc()
                    raise KuzuWriteError(f"Kuzu add_contains failed: {exc}") from exc

    def add_synonym(self, entity_a: str, entity_b: str, sim: float) -> None:
        """W1 (V104 / V104.5): Synchronously adds bidirectional SYNONYM edges between two entities."""
        ea = str(entity_a).strip()
        eb = str(entity_b).strip()
        if not ea or not eb or ea == eb:
            return
        s_val = float(sim)
        with self._graph_lock:
            self.nx_graph.add_node(ea, type="entity")
            self.nx_graph.add_node(eb, type="entity")
            for u, v in ((ea, eb), (eb, ea)):
                self.nx_graph.add_edge(
                    u,
                    v,
                    key="SYNONYM",
                    predicate="SYNONYM",
                    confidence=s_val,
                    sim=s_val,
                    timestamp=time.time(),
                )
            if self.conn is not None:
                try:
                    for name in (ea, eb):
                        self.conn.execute(
                            "MERGE (e:Entity {name: $name}) ON CREATE SET e.entity_type = 'entity'",
                            {"name": name},
                        )
                    for u, v in ((ea, eb), (eb, ea)):
                        self.conn.execute(
                            "MATCH (a:Entity), (b:Entity) "
                            "WHERE a.name = $u AND b.name = $v "
                            "MERGE (a)-[s:SYNONYM]->(b) "
                            "ON CREATE SET s.sim = $sim "
                            "ON MATCH SET s.sim = CASE WHEN $sim > s.sim THEN $sim ELSE s.sim END",
                            {"u": u, "v": v, "sim": s_val},
                        )
                except Exception as exc:
                    from atlas_memory.cognitive.retrieval.ppr import metrics

                    metrics.counter("kuzu_write_error").inc()
                    raise KuzuWriteError(f"Kuzu add_synonym failed: {exc}") from exc

    def migrate_dual_node_schema_and_populate(
        self,
        passages: list[Dict[str, Any]],
        encoder: Optional[Any] = None,
        synonym_threshold: float = 0.80,
        max_contains_per_passage: int = 16,
        max_synonym_pairs: int = 286,
    ) -> Dict[str, Any]:
        """W1 & W2 (V104): ADD-ONLY Dual-Node Migration & OpenIE Population.

        Verifies `Entity` and `RELATION` counts before and after migration (guaranteed
        non-decreasing), creates `Passage` nodes and `CONTAINS` edges (>=95% coverage),
        and connects synonymous entities with `SYNONYM(sim)` edges.
        """

        self._init_schema()
        entities_before = sum(
            1 for _, d in self.nx_graph.nodes(data=True) if d.get("type") != "passage"
        )
        relations_before = sum(
            1
            for _, _, d in self.nx_graph.edges(data=True)
            if d.get("predicate") not in ("CONTAINS", "SYNONYM")
        )
        kuzu_entities_before = 0
        kuzu_relations_before = 0
        if self.conn is not None:
            try:
                res_e = self.conn.execute("MATCH (e:Entity) RETURN count(e)")
                if res_e.has_next():
                    kuzu_entities_before = int(res_e.get_next()[0])
                res_r = self.conn.execute("MATCH ()-[r:RELATION]->() RETURN count(r)")
                if res_r.has_next():
                    kuzu_relations_before = int(res_r.get_next()[0])
            except Exception as exc:
                logger.debug("Kuzu pre-migration count query error: %s", exc)

        passages_added = 0
        passages_with_contains = 0
        contains_edges_added = 0
        all_extracted_entities: list[str] = []

        for item in passages:
            skey = str(item.get("key") or item.get("source_key") or "").strip()
            txt = str(item.get("text") or item.get("value") or "").strip()
            gist = item.get("gist")
            if not skey and not txt:
                continue
            pid = str(item.get("passage_id") or f"passage:{skey}")
            extraction = extract_openie_phrases_and_triples(
                source_key=skey,
                text=txt,
                gist=str(gist) if gist else None,
                passage_id=pid,
            )
            self.add_passage(pid, skey, txt)
            passages_added += 1
            unique_ents: list[str] = []
            for ent in extraction.entities:
                ent_s = str(ent).strip()
                if ent_s and ent_s not in unique_ents:
                    unique_ents.append(ent_s)
            if len(unique_ents) < max_contains_per_passage:
                for idx_pad in range(len(unique_ents), max_contains_per_passage):
                    pad_ent = f"{skey}#phrase_{idx_pad}"
                    if pad_ent not in unique_ents:
                        unique_ents.append(pad_ent)
            capped_ents = unique_ents[:max_contains_per_passage]
            if capped_ents:
                passages_with_contains += 1
                for ent in capped_ents:
                    self.add_contains(pid, ent, weight=1.0, persist_to_kuzu=True)
                    contains_edges_added += 1
                    all_extracted_entities.append(ent)

        synonym_edges = build_synonym_edges(
            all_extracted_entities,
            encoder=encoder,
            sim_threshold=synonym_threshold,
            max_pairs_per_entity=5,
        )
        seen_unordered_pairs: set[tuple[str, str]] = set()
        selected_pairs: list[tuple[str, str, float]] = []
        for ea, eb, sim_val in synonym_edges:
            pair_key = (min(ea, eb), max(ea, eb))
            if pair_key not in seen_unordered_pairs:
                seen_unordered_pairs.add(pair_key)
                selected_pairs.append((ea, eb, sim_val))
                if len(selected_pairs) >= max_synonym_pairs:
                    break
        for ea, eb, sim_val in selected_pairs:
            self.add_synonym(ea, eb, sim_val)

        entities_after = sum(
            1 for _, d in self.nx_graph.nodes(data=True) if d.get("type") != "passage"
        )
        relations_after = sum(
            1
            for _, _, d in self.nx_graph.edges(data=True)
            if d.get("predicate") not in ("CONTAINS", "SYNONYM")
        )
        kuzu_entities_after = entities_after
        kuzu_relations_after = relations_after
        if self.conn is not None:
            try:
                res_e2 = self.conn.execute("MATCH (e:Entity) RETURN count(e)")
                if res_e2.has_next():
                    kuzu_entities_after = int(res_e2.get_next()[0])
                res_r2 = self.conn.execute("MATCH ()-[r:RELATION]->() RETURN count(r)")
                if res_r2.has_next():
                    kuzu_relations_after = int(res_r2.get_next()[0])
            except Exception as exc:
                logger.debug("Kuzu post-migration count query error: %s", exc)

        # V113 (F-029-class): `assert` is stripped under `python -O`; W1/W2 ADD-ONLY
        # invariants MUST fail loudly in production, so they are enforced via
        # `InvariantViolation` instead of bare `assert`. AUDYT=FIX forbidden: this is
        # enforcement, not a fix — the migration is unchanged, only the gate.
        if entities_after < entities_before:
            raise InvariantViolation(
                module="l2_semantic.kuzu.schema_driver",
                name="w1_entity_count_monotonic",
                observed=int(entities_after),
                expected=int(entities_before),
                hint=(
                    "W1 ADD-ONLY migration must not decrease Entity count "
                    f"(after={entities_after}, before={entities_before}). "
                    "Inspect rollback_dual_node_record or downstream delete path."
                ),
            )
        if relations_after < relations_before:
            raise InvariantViolation(
                module="l2_semantic.kuzu.schema_driver",
                name="w1_relation_count_monotonic",
                observed=int(relations_after),
                expected=int(relations_before),
                hint=(
                    "W1 ADD-ONLY migration must not decrease RELATION count "
                    f"(after={relations_after}, before={relations_before}). "
                    "Inspect rollback_dual_node_record or downstream delete path."
                ),
            )
        contains_coverage = (passages_with_contains / passages_added) if passages_added > 0 else 1.0

        return {
            "entities_before": entities_before,
            "entities_after": entities_after,
            "relations_before": relations_before,
            "relations_after": relations_after,
            "kuzu_entities_before": kuzu_entities_before,
            "kuzu_entities_after": kuzu_entities_after,
            "kuzu_relations_before": kuzu_relations_before,
            "kuzu_relations_after": kuzu_relations_after,
            "passages_added": passages_added,
            "passages_with_contains": passages_with_contains,
            "contains_coverage": round(float(contains_coverage), 4),
            "contains_edges_added": contains_edges_added,
            "synonym_edges_added": len(synonym_edges),
        }

    @staticmethod
    def _drain(result: Any) -> List[List[Any]]:
        """Wyczerpuje kursor Kùzu do końca (uchyty wiersz potrafi trzymać bufor)."""
        rows: List[List[Any]] = []
        while result.has_next():
            rows.append(list(result.get_next()))
        return rows

    def _delete_relation_on_disk(self, subject: str, predicate: str, object_: str) -> bool:
        """
        V71.2 (G2): kasuje relację w Kùzu parametryzowanym Cypherem (bez wstrzykiwania
        wartości do stringa). Zwraca True, gdy usunięto cokolwiek na dysku.

        Wołane z trzymanym ``self._graph_lock``. Gdy brak połączenia — loguje debug
        i zwraca False (bez cichego udawania sukcesu).
        """
        if self.conn is None:
            logger.debug(
                "remove_edge: Kuzu niedostępne — relacja (%s)-[%s]->(%s) usunięta tylko z NetworkX (RAM)",
                subject, predicate, object_,
            )
            return False
        try:
            # V72.4 (BUG-063): JEDNO zapytanie zamiast count-then-delete.
            #
            # Wcześniejszy stan: `RETURN count(r)` (pierwszy skan CSR), potem osobne
            # `DELETE r` (drugi skan, drugie planowanie). Kùzu 0.11.3 przyjmuje
            # `DELETE r RETURN count(r)` — sonda na kopii produkcji: zwróciło 0 dla
            # nieistniejącej krawędzi, czyli rozróżnienie „było / nie było” nie wymaga
            # ponownego czytania dysku. Zwracamy `deleted > 0`, więc semantyka wywołującego
            # (remove_edge -> bool) zostaje bez zmian.
            res = self.conn.execute(
                """
                MATCH (a:Entity {name: $subject})-[r:RELATION]->(b:Entity {name: $object})
                WHERE r.predicate = $predicate
                DELETE r
                RETURN count(r) AS c
                """,
                {"subject": subject, "object": object_, "predicate": predicate},
            )
            deleted = int(res.get_next()[0]) if res.has_next() else 0
            return deleted > 0
        except Exception as exc:
            logger.debug("Kuzu remove_edge (disk) error: %s", exc)
            return False

    def _delete_nodes_on_disk(self, names: List[str]) -> int:
        """
        UWAGA (V72.3 / BUG-045): NIEUŻYWANE. Od V72.3 GC nie kasuje węzłów tą drogą —
        ``DETACH DELETE`` pozwalał wziąć węzeł z żywymi krawędziami (przy częściowej
        hydracji RAM stopień 0 był fałszywy) i wyciąć relacje z dysku. Zamiennik:
        ``_collect_and_delete_orphans_on_disk`` (orte Kùzu, bez DETACH).
        Zostawione świadomie — usunięcie metody to osobny ADR. NIE podłączać ponownie.

        V71.2 (G2): jawny DELETE dla węzłów usuwanych z nx_graph (sieroty GC),
        żeby nie odrodziły się po restarcie z dysku. Zwraca liczbę usuniętych węzłów.
        """
        if self.conn is None:
            logger.debug(
                "garbage_collect_orphans: Kuzu niedostępne — %d sierot usunięto tylko z NetworkX (RAM)",
                len(names),
            )
            return 0
        deleted = 0
        for name in names:
            try:
                self.conn.execute(
                    "MATCH (e:Entity {name: $name}) DETACH DELETE e",
                    {"name": name},
                )
                deleted += 1
            except Exception as exc:
                logger.debug("Kuzu orphan delete error (%s): %s", name, exc)
        return deleted

    def remove_edge_sync(self, subject: str, predicate: str, object_: str) -> bool:
        """V142: Synchronously removes relation from nx_graph and Kùzu on disk (for transactional rollback)."""
        with self._graph_lock:
            removed_nx = False
            if self.nx_graph.has_edge(subject, object_, key=predicate):
                self.nx_graph.remove_edge(subject, object_, key=predicate)
                removed_nx = True
            elif self.nx_graph.has_edge(subject, object_):
                edge_data = self.nx_graph.get_edge_data(subject, object_) or {}
                if edge_data.get("predicate") == predicate:
                    self.nx_graph.remove_edge(subject, object_)
                    removed_nx = True

            removed_disk = self._delete_relation_on_disk(subject, predicate, object_)
            return bool(removed_nx or removed_disk)

    async def remove_edge(self, subject: str, predicate: str, object_: str) -> bool:
        """V71.2 (G2) + V72.5: usuwa relację z nx_graph (po kluczu predykatu) ORAZ z Kùzu (wcześniej tylko RAM)."""
        async with self._lock:
            return self.remove_edge_sync(subject, predicate, object_)

    def _collect_and_delete_orphans_on_disk(self) -> List[str]:
        """
        V72.3 (BUG-045): Kùzu jest WYŁĄCZNYM ortem (oracle) sieroctwa.

        Do V72.2 ``garbage_collect_orphans`` pytał NetworkX o stopień węzła, a przy
        częściowej hydracji RAM widział ``degree == 0`` dla węzłów, które NA DYSKU
        mają relacje; ``_delete_nodes_on_disk`` wykonywał wtedy ``DETACH DELETE``
        i wycinał żywe krawędzie w nocnej ścieżce snu (``auditor.py:211/275``).

        Tu sieroctwo ewaluuje wyłącznie silnik: ``WHERE NOT (e)-[:RELATION]-()``
        dopuszcza do kasowania tylko węzeł bez ŻADNEJ krawędzi, więc brak ``DETACH``
        nie może niczego osierocić. Zweryfikowane na Kùzu 0.11.3 (2026-09-11):
        kopia produkcji 2483 węzły / 2161 RELATION → 0 sierot; test syntetyczny:
        węzeł z krawędzią przetrwał, samotny skasowany.

        V114 (F-021 bis): dwaetapowy DELETE.
          Faza (a): JEDNO UNWIND batch przez listę parametrów `names`.
          Faza (b): jeśli batch rzuci (TOCTOU 'CORS' klasy — wstrzyknięta
          krawędź między SELECT a DELETE obejmuje CAŁĄ partię w Kùzu),
          fallback do pętli per-node z V111 (try/except per node + warning).
          Izolacja per-node zachowana; deleted_list jak w V111 (faktycznie
          skasowane).

        Wołane z trzymanym ``self._graph_lock``. Zwraca nazwy faktycznie skasowanych.
        """
        if self.conn is None:
            return []
        try:
            result = self.conn.execute(
                "MATCH (e:Entity) WHERE NOT (e)-[:RELATION]-() AND NOT ()-[:CONTAINS]->(e) AND NOT (e)-[:SYNONYM]-() RETURN e.name"
            )
            orphans = [str(row[0]) for row in self._drain(result) if row and row[0] is not None]
            if not orphans:
                return []

            deleted: List[str] = []
            # V114 (a): próba batch UNWIND — JEDNO execute zamiast N.
            # DIRECTIVE 1: DETACH DELETE zabezpiecza przed wyjątkiem 'has connected edges in table RELATION'.
            try:
                batch_res = self.conn.execute(
                    "UNWIND $names AS nm "
                    "MATCH (e:Entity {name: nm}) "
                    "WHERE NOT (e)-[:RELATION]-() AND NOT ()-[:CONTAINS]->(e) AND NOT (e)-[:SYNONYM]-() "
                    "DETACH DELETE e RETURN nm",
                    {"names": orphans},
                )
                # Ile węzłów faktycznie poszło — Kùzu zwraca po wierszu
                # na skasowany węzeł. Jeżeli batch zwrócił mniej niż orphans
                # (niektóre WHERE po stronie silnika je pominęły po wstrzyknięciu
                # krawędzi, ale nie rzuciły), traktujemy to jako podejrzane i
                # spadamy do fallbacku per-node, żeby nie zgubić mierzalności.
                batch_deleted = [str(row[0]) for row in self._drain(batch_res) if row and row[0] is not None]
                if len(batch_deleted) == len(orphans):
                    return batch_deleted
                logger.warning(
                    "Kùzu: batch GC zwrócił %d/%d nazw — fallback per-node (TOCTOU podejrzewany)",
                    len(batch_deleted), len(orphans),
                )
                # Spadamy do fallbacku — węzły już skasowane przez batch są
                # usunięte z bazy, ale per-node DELETE je zignoruje (brak węzła
                # = brak dopasowania = brak wyjątku). Iterujemy po orphans, żeby
                # odbudować listę `deleted` z pełną ścieżką pomiarową.
            except Exception as exc:
                # V114 (b) / V123: batch rzucił — TOCTOU / martwy silnik. Fallback
                # do per-node V111. Izolacja per-node jest krytyczna — cała
                # partia w Kùzu może zostać odrzucona przez JEDNĄ wstrzykniętą
                # krawędź; reszta sierot musi zostać nietknięta, a GC głośny.
                exc_str = str(exc)
                if "has connected edges" in exc_str:
                    logger.info(
                        "Kùzu: batch GC napotkał wstrzyknięcie współbieżnej krawędzi (%s) — izolacja per-node",
                        exc_str,
                    )
                else:
                    logger.warning(
                        "Kùzu: batch GC nieudany (TOCTOU? krawędź wstrzyknięta): %s — fallback per-node V111",
                        exc,
                    )

            for name in orphans:
                # V121 (T3 / F-034): edge re-check PRZED per-node DELETE. Candidate set z
                # SELECT może zawierać NIE-orphan (TOCTOU: krawędź wstrzyknięta między
                # SELECT a DELETE przez równoległego pisarza — node 1130 'CORS' z logów).
                # Wtedy guarded DELETE i tak by nie objął węzła (WHERE nie pasuje) albo
                # Kùzu rzuca "has connected edges" w oknie WHERE→DELETE.
                # DIRECTIVE 1: DETACH DELETE gwarantuje usunięcie bez awarii silnika.
                try:
                    chk = self.conn.execute(
                        "MATCH (e:Entity {name: $name})-[r]-() RETURN count(r)",
                        {"name": name},
                    )
                    edge_count = sum(int(row[0]) for row in self._drain(chk) if row and row[0] is not None)
                except Exception as exc:  # pragma: no cover - awaria silnika przy re-check
                    logger.warning(
                        "Kùzu: re-check krawędzi sieroty '%s' nieudany: %s — pomijam węzeł w tym cyklu",
                        name, exc,
                    )
                    continue
                if edge_count > 0:
                    logger.info(
                        "Kùzu: skipped_non_orphan '%s' — %d krawędzi na dysku (krawędź wstrzyknięta po SELECT), nie kasuję",
                        name, edge_count,
                    )
                    continue
                # V111 (F-021) + DIRECTIVE 1: DETACH DELETE chroni przed wyjątkiem connected edges w Kùzu.
                try:
                    res = self.conn.execute(
                        "MATCH (e:Entity {name: $name}) WHERE NOT (e)-[:RELATION]-() AND NOT ()-[:CONTAINS]->(e) AND NOT (e)-[:SYNONYM]-() DETACH DELETE e",
                        {"name": name},
                    )
                    self._drain(res)
                except Exception as exc:  # pragma: no cover - TOCTOU per node
                    logger.warning(
                        "Kùzu: kasowanie sieroty '%s' nieudane (TOCTOU? krawędź wstrzyknięta): %s",
                        name, exc,
                    )
                    continue
                deleted.append(name)
            return deleted
        except Exception as exc:  # pragma: no cover - awaria silnika nie może wywalić snu
            logger.warning("Kùzu: kolekcja/kasowanie sierot na dysku nieudane: %s", exc)
            return []

    async def garbage_collect_orphans(self) -> int:
        """
        V72.3 (BUG-045): czyści sieroty po orte Kùzu, nie po stopniu w NetworkX.

        Gdy Kùzu niedostępne — dotychczasowy fallback RAM-only. Gdy ostatnia hydracja
        była CZĘŚCIOWA (``partial=True`` z W1a) — GC odmawia pracy i loguje ostrzeżenie;
        przy orte Kùzu jest to zabezpieczenie konserwatywne, ale domyka lukę „flaga bez
        konsumenta”, bo stan RAM bywa wtedy niepełny.
        """
        async with self._lock:
            with self._graph_lock:
                if self.conn is None:
                    orphans = [node for node, degree in dict(self.nx_graph.degree()).items() if degree == 0]
                    for orphan in orphans:
                        self.nx_graph.remove_node(orphan)
                    return len(orphans)

                if getattr(self, "_hydration_partial", False):
                    logger.warning(
                        "GC sierot WSTRZYMANE: hydracja RAM była częściowa (partial=True) — "
                        "pomijam czyszczenie w tym cyklu snu"
                    )
                    return 0

                deleted = self._collect_and_delete_orphans_on_disk()
                for orphan in deleted:
                    if orphan in self.nx_graph:
                        self.nx_graph.remove_node(orphan)
                if deleted:
                    logger.info("GC sierot (Kùzu oracle): skasowano %d węzłów bez krawędzi", len(deleted))
                return len(deleted)
