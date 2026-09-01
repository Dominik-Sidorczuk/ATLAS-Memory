"""Mnemosyne SQLite Ingest Engine for ATLAS Vault.

Synchronizes legacy Mnemosyne memory records (working_memory, facts, canonical_facts,
episodic_memory, consolidated_facts) into the hardware-accelerated ATLAS Memory Vault:
- SQLite Verified KV Store (~/.hermes/atlas/atlas.db)
- Kùzu Graph Store (~/.hermes/atlas/kuzu)
- RaBitQ / Qdrant Vector Store (~/.hermes/atlas/qdrant)
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from atlas_memory.models import EpistemicSource, MemoryRecord

logger = logging.getLogger(__name__)

DEFAULT_MNEMOSYNE_DB_PATH = Path.home() / ".hermes" / "mnemosyne" / "data" / "mnemosyne.db"
DEFAULT_HERMES_MEMORY_MD_PATH = Path.home() / ".hermes" / "memories" / "MEMORY.md"


class MnemosyneIngestEngine:
    """Ingests legacy Mnemosyne SQLite databases and Hermes MEMORY.md into ATLAS Memory Vault."""

    def __init__(
        self,
        db_path: Path | str = DEFAULT_MNEMOSYNE_DB_PATH,
        memory_md_path: Path | str = DEFAULT_HERMES_MEMORY_MD_PATH,
    ) -> None:
        self.db_path = Path(db_path)
        self.memory_md_path = Path(memory_md_path)

    @property
    def is_available(self) -> bool:
        """Returns True if either the source Mnemosyne database or MEMORY.md exists and is readable."""
        has_db = self.db_path.exists() and os.access(self.db_path, os.R_OK)
        has_md = self.memory_md_path.exists() and os.access(self.memory_md_path, os.R_OK)
        return has_db or has_md

    def extract_memory_md_records(self, path: Optional[Path | str] = None) -> List[MemoryRecord]:
        """Extracts facts from Hermes MEMORY.md and USER.md files (separated by §)."""
        target_files: List[Path] = []
        if path:
            p = Path(path)
            if p.is_dir():
                target_files.extend(sorted(p.glob("*.md")))
            elif p.exists():
                target_files.append(p)
        else:
            memories_dir = self.memory_md_path.parent
            if memories_dir.is_dir():
                target_files.extend(sorted([f for f in memories_dir.glob("*.md") if not f.name.endswith(".lock")]))
            elif self.memory_md_path.exists():
                target_files.append(self.memory_md_path)

        records: List[MemoryRecord] = []

        for target_path in target_files:
            try:
                content = target_path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("Could not read %s: %s", target_path, exc)
                continue

            origin_name = "hermes_user_md" if target_path.stem.upper() == "USER" else "hermes_memory_md"
            raw_sections = [s.strip() for s in content.split("§") if s.strip()]

            for idx, sec in enumerate(raw_sections):
                lines = [line.strip() for line in sec.splitlines() if line.strip()]
                if not lines:
                    continue
                first_line = lines[0]
                if ":" in first_line:
                    subject = first_line.split(":", 1)[0].strip()
                else:
                    words = first_line.split()
                    subject = " ".join(words[:5]) if words else f"fact_{idx}"

                records.append(
                    MemoryRecord(
                        subject=subject[:80],
                        predicate="stated_memory",
                        object=sec,
                        importance_score=0.98 if origin_name == "hermes_user_md" else 0.90,
                        source_type=EpistemicSource.USER_EXPLICIT,
                        source_session_id="hermes_persistent",
                        metadata={
                            "origin": origin_name,
                            "section_index": idx,
                            "source_file": str(target_path),
                        },
                    )
                )

        logger.info("Extracted %d records from %d memory markdown files", len(records), len(target_files))
        return records

    def sync_memory_md_into_engine(self, engine: Any, path: Optional[Path | str] = None) -> Dict[str, Any]:
        """Syncs all entries from MEMORY.md and USER.md into ATLAS engine KV store and Graph."""
        records = self.extract_memory_md_records(path)
        if not records:
            return {"status": "ok", "ingested": 0, "source": "memories"}

        count = 0
        for rec in records:
            origin_prefix = "user_md" if rec.metadata.get("origin") == "hermes_user_md" else "memory_md"
            var_key = f"hermes:{origin_prefix}:{rec.subject}"
            if hasattr(engine, "kv") and engine.kv is not None:
                try:
                    engine.kv.set_sync(var_key, rec.object, confidence=1.0, metadata=rec.metadata, reason="sync_memory_md")
                    count += 1
                except Exception as exc:
                    logger.debug("Failed to set %s in KV: %s", var_key, exc)
            if hasattr(engine, "graph") and engine.graph is not None:
                try:
                    engine.graph.add_relation(rec.subject, "stated_memory", str(rec.object)[:80])
                except Exception:
                    pass

        return {"status": "ok", "ingested": count, "total_sections": len(records), "source": "memories"}

    def extract_records(self) -> List[MemoryRecord]:
        """Extracts all facts and working memory items from Mnemosyne SQLite and MEMORY.md."""
        records: List[MemoryRecord] = []

        # 1. Extract from MEMORY.md first
        records.extend(self.extract_memory_md_records())

        if not (self.db_path.exists() and os.access(self.db_path, os.R_OK)):
            logger.info("Mnemosyne DB not found at %s — skipping legacy sqlite ingest", self.db_path)
            return records

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            # 1. Ingest working_memory
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT id, content, importance, session_id, created_at, event_date FROM working_memory"
                )
                for row in cur.fetchall():
                    content = row["content"] or ""
                    importance = float(row["importance"] or 0.7)
                    session_id = row["session_id"] or "hermes_default"
                    created_at = row["created_at"] or ""
                    
                    words = content.split()
                    subject = " ".join(words[:4]) if words else "working_memory"
                    
                    records.append(
                        MemoryRecord(
                            subject=subject[:80],
                            predicate="working_fact",
                            object=content,
                            importance_score=min(1.0, max(0.1, importance)),
                            source_type=EpistemicSource.USER_EXPLICIT if importance > 0.8 else EpistemicSource.TOOL_OUTPUT,
                            source_session_id=session_id,
                            metadata={
                                "origin": "mnemosyne_working_memory",
                                "mnemosyne_id": row["id"],
                                "event_date": row["event_date"],
                                "created_at": created_at,
                            },
                        )
                    )
            except sqlite3.OperationalError as exc:
                logger.debug("Table working_memory not present or readable: %s", exc)

            # 2. Ingest facts (structured triples)
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT fact_id, session_id, subject, predicate, object, confidence, created_at FROM facts"
                )
                for row in cur.fetchall():
                    records.append(
                        MemoryRecord(
                            subject=str(row["subject"])[:80],
                            predicate=str(row["predicate"])[:50],
                            object=str(row["object"]),
                            confidence=float(row["confidence"] or 1.0),
                            source_type=EpistemicSource.EXTERNAL_DOC,
                            source_session_id=row["session_id"] or "hermes_default",
                            metadata={
                                "origin": "mnemosyne_facts",
                                "fact_id": row["fact_id"],
                                "created_at": row["created_at"],
                            },
                        )
                    )
            except sqlite3.OperationalError as exc:
                logger.debug("Table facts not present: %s", exc)

            # 3. Ingest canonical_facts
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT category, name, body, source, confidence FROM canonical_facts WHERE valid_until IS NULL"
                )
                for row in cur.fetchall():
                    category = row["category"] or "general"
                    name = row["name"] or "fact"
                    records.append(
                        MemoryRecord(
                            subject=f"{category}:{name}"[:80],
                            predicate="canonical_spec",
                            object=str(row["body"]),
                            confidence=float(row["confidence"] or 1.0),
                            importance_score=0.95,
                            source_type=EpistemicSource.USER_EXPLICIT,
                            metadata={
                                "origin": "mnemosyne_canonical_facts",
                                "source": row["source"],
                            },
                        )
                    )
            except sqlite3.OperationalError as exc:
                logger.debug("Table canonical_facts not present: %s", exc)

            # 4. Ingest episodic_memory
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT id, content, importance, session_id, created_at FROM episodic_memory"
                )
                for row in cur.fetchall():
                    content = row["content"] or ""
                    words = content.split()
                    subject = " ".join(words[:4]) if words else "episodic_turn"
                    records.append(
                        MemoryRecord(
                            subject=subject[:80],
                            predicate="episodic_event",
                            object=content,
                            importance_score=float(row["importance"] or 0.6),
                            source_type=EpistemicSource.AGENT_INFERENCE,
                            source_session_id=row["session_id"] or "hermes_default",
                            metadata={
                                "origin": "mnemosyne_episodic_memory",
                                "id": row["id"],
                                "created_at": row["created_at"],
                            },
                        )
                    )
            except sqlite3.OperationalError as exc:
                logger.debug("Table episodic_memory not present: %s", exc)

        finally:
            conn.close()

        logger.info("Extracted %d total records from Mnemosyne & MEMORY.md", len(records))
        return records

    def sync_into_atlas_engine(self, engine: Any) -> Dict[str, Any]:
        """Ingests all extracted records directly into the Atlas HybridMemoryEngine."""
        records = self.extract_records()
        if not records:
            return {"ingested": 0, "status": "no_records_or_missing_sources"}

        ingested_count = 0
        kv_count = 0
        graph_count = 0

        for rec in records:
            try:
                # 1. Write to KV Store (Source of Truth ledger)
                if hasattr(engine, "kv") and engine.kv is not None:
                    var_key = f"mnemosyne:{rec.metadata.get('origin', 'raw')}:{rec.subject}:{rec.predicate}"
                    engine.kv.set_sync(var_key, rec.object, metadata=rec.metadata)
                    kv_count += 1

                # 2. Write to Kùzu Knowledge Graph if present
                if hasattr(engine, "graph") and engine.graph is not None:
                    try:
                        engine.graph.add_entity(rec.subject, entity_type="Concept")
                        engine.graph.add_entity(rec.object[:80], entity_type="Observation")
                        engine.graph.add_relation(rec.subject, rec.predicate, rec.object[:80])
                        graph_count += 1
                    except Exception as g_exc:
                        logger.debug("Graph insert error for %s: %s", rec.subject, g_exc)

                ingested_count += 1
            except Exception as exc:
                logger.debug("Failed to ingest record %s: %s", rec.subject, exc)

        # Seed core architectural and domain causal dependencies
        if hasattr(engine, "graph") and engine.graph is not None:
            domain_triples = [
                ("pixi_environment", "hosts", "python314_nogil", 1.0),
                ("python314_nogil", "powers", "atlas_daemon_kuzu_qdrant", 1.0),
                ("python314_nogil", "critical_for", "no_gil_multithreading", 1.0),
                ("remove_python314_nogil", "breaks", "atlas_c_rust_libraries", 1.0),
                ("GHK-Cu", "influences", "collagen_synthesis", 1.0),
                ("GHK-Cu", "restricted_by", "max_dose_le_2_to_3mg", 1.0),
                ("dose_gt_3mg", "causes", "copper_toxicity_and_diminishing_returns", 1.0),
                ("KlowStack", "contains", "GHK-Cu", 1.0),
                ("KlowStack", "requires", "colonoscopy_cea_fit_oncology_screening", 1.0),
                ("omitting_screening", "critical_for", "tumor_stimulation_via_angiogenesis", 1.0),
                ("KlowStack", "requires_condition", "storage_at_4C", 1.0),
                ("Crawl4AI", "runs_on", "localhost_11235", 1.0),
                ("Obsidian_MCP", "runs_on", "localhost_27124", 1.0),
            ]
            for s, p, o, conf in domain_triples:
                try:
                    engine.graph.add_relation(s, p, o, confidence=conf)
                    graph_count += 1
                except Exception:
                    pass

        return {
            "status": "ok",
            "total_extracted": len(records),
            "ingested_records": ingested_count,
            "kv_records_synced": kv_count,
            "graph_triples_synced": graph_count,
        }
