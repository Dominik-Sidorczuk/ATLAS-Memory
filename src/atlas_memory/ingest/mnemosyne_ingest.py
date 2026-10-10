"""Mnemosyne SQLite Ingest Engine for ATLAS Vault.

Synchronizes legacy Mnemosyne memory records (working_memory, facts, canonical_facts,
episodic_memory, consolidated_facts) into the hardware-accelerated ATLAS Memory Vault:
- SQLite Verified KV Store (~/.hermes/atlas/atlas.db)
- Kùzu Graph Store (~/.hermes/atlas/kuzu)
- Qdrant Vector Store (~/.hermes/atlas/qdrant)
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from atlas_memory.cognitive.models import HERMES_DEFAULT_SESSION
from atlas_memory.cognitive.write_path import EpistemicWritePath
from atlas_memory.core.linguistics import LanguageAssetManager, is_discourse_garbage_subject
from atlas_memory.core.models import EpistemicSource, MemoryRecord
from atlas_memory.ingest.domain_rules import DEFAULT_APP_EXTRACTOR, DomainRelationExtractor

logger = logging.getLogger(__name__)

DEFAULT_MNEMOSYNE_DB_PATH = Path.home() / ".hermes" / "mnemosyne" / "data" / "mnemosyne.db"
DEFAULT_HERMES_MEMORY_MD_PATH = Path.home() / ".hermes" / "memories" / "MEMORY.md"

# V88/T3: kandydujące ścieżki plików pamięci Hermes w kolejności precedencji.
_HERMES_MEMORY_MD_CANDIDATES: Tuple[str, ...] = (
    "memories/MEMORY.md",
    "memories/memory.md",
    "MEMORY.md",
    "memory.md",
)


def hermes_home_dir() -> Path:
    """Katalog domowy Hermes liczony W CZASIE WYWOŁANIA (env HERMES_HOME lub ~/.hermes).

    Uwaga: `DEFAULT_HERMES_MEMORY_MD_PATH` jest zamrażany przy imporcie modułu, więc
    nie nadaje się do rozstrzygania ścieżek w procesie, którego HOME/HERMES_HOME
    różni się od środowiska importu (daemon, testy, worktree).
    """
    env_home = os.environ.get("HERMES_HOME")
    if env_home:
        return Path(env_home).expanduser()
    return Path.home() / ".hermes"


def memory_md_candidates() -> List[Path]:
    """Zwraca listę kandydujących ścieżek MEMORY.md (kanoniczna + układ legacy)."""
    home = hermes_home_dir()
    return [home / rel for rel in _HERMES_MEMORY_MD_CANDIDATES]


def resolve_default_memory_md_path() -> Path:
    """Pierwsza istniejąca ścieżka pamięci Hermes; fallback = ścieżka kanoniczna."""
    candidates = memory_md_candidates()
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


class MnemosyneIngestEngine:
    """Ingests legacy Mnemosyne SQLite databases and Hermes MEMORY.md into ATLAS Memory Vault."""

    DEFAULT_EXTRACTOR: DomainRelationExtractor = DEFAULT_APP_EXTRACTOR

    def __init__(
        self,
        db_path: Path | str = DEFAULT_MNEMOSYNE_DB_PATH,
        memory_md_path: Optional[Path | str] = None,
    ) -> None:
        self.db_path = Path(db_path)
        # V88/T3: ścieżka pamięci jest rozwiązywana W CZASIE WYWOŁANIA konstruktora.
        # Wcześniej domyślna wartość była zamrażana przy imporcie modułu
        # (DEFAULT_HERMES_MEMORY_MD_PATH), więc proces z innym HOME/HERMES_HOME
        # czytał (nieistniejący) plik macierzystego użytkownika i cicho zwracał 0 rekordów.
        self._explicit_memory_md = memory_md_path is not None
        self.memory_md_path = (
            resolve_default_memory_md_path() if memory_md_path is None else Path(memory_md_path)
        )

    @property
    def is_available(self) -> bool:
        """Returns True if either the source Mnemosyne database or MEMORY.md exists and is readable."""
        has_db = self.db_path.exists() and os.access(self.db_path, os.R_OK)
        has_md = self.memory_md_path.exists() and os.access(self.memory_md_path, os.R_OK)
        return has_db or has_md

    def _collect_default_sources(self) -> List[Path]:
        """Pliki źródłowe domyślne: katalog `memories/` albo łańcuch kandydatów (legacy)."""
        if self._explicit_memory_md:
            parent = self.memory_md_path.parent
            if parent.is_dir():
                return sorted(f for f in parent.glob("*.md") if not f.name.endswith(".lock"))
            return [self.memory_md_path] if self.memory_md_path.exists() else []

        memories_dir = self.memory_md_path.parent
        if memories_dir.is_dir() and memories_dir.name == "memories":
            return sorted(f for f in memories_dir.glob("*.md") if not f.name.endswith(".lock"))
        existing = [c for c in memory_md_candidates() if c.is_file()]
        if existing:
            return existing
        return [self.memory_md_path] if self.memory_md_path.exists() else []

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
            # V88/T3: bez jawnego `path` rozstrzygamy źródło ponownie przy każdym
            # wywołaniu (katalog HOME lub HERMES_HOME mógł zmienić się po imporcie).
            if not self._explicit_memory_md:
                self.memory_md_path = resolve_default_memory_md_path()
            target_files = self._collect_default_sources()
            if not target_files:
                logger.warning(
                    "Brak źródła pamięci Hermes do ingesti — żadna ścieżka nie istnieje: %s "
                    "(sprawdź HERMES_HOME albo utwórz MEMORY.md)",
                    [str(c) for c in memory_md_candidates()],
                )

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

                # V151 (Wektor I, H2): bariera dyskursywna (POS/gramatyka) —
                # zaimki/wyrazy pytające wcielone w temat sekcji ('jak_obecnie',
                # 'który_obecnie_nie', 'czy_warto', 'gdzie_ta_usługa', 'nie_ma_…')
                # nie są encjami — nie wchodzą do KV, Qdrant ani Kùzu. Śmieci
                # rodzą fałszywe konflikty w online'owym ConflictArbiter.
                if is_discourse_garbage_subject(subject):
                    logger.info(
                        "[V151] Sekcja MEMORY.md odcięta przez barierę dyskursywną: subject=%r (origin=%s idx=%d)",
                        subject, origin_name, idx,
                    )
                    continue

                records.append(
                    MemoryRecord(
                        subject=subject,
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

        logger.debug("Extracted %d records from %d memory markdown files", len(records), len(target_files))
        return records

    @classmethod
    def extract_and_insert_multihop_relations(
        cls,
        graph: Any,
        subject: str,
        content: str,
        write_path: Optional[Any] = None,
        extractor: Optional[DomainRelationExtractor] = None,
    ) -> int:
        """Extracts sub-entities, causal relations, and multi-hop edges from text into the knowledge graph."""
        if graph is None or not content:
            return 0
        if not callable(getattr(graph, "add_relation", None)):
            return 0

        content_clean = content.strip()
        if len(content_clean) < 20 and ":" not in content_clean:
            return 0

        edges_added = 0
        clean_sub = subject.strip()
        wp = write_path or EpistemicWritePath(graph_store=graph)

        def _add_edge(u: str, pred: str, v: str, conf: float = 0.90) -> None:
            nonlocal edges_added
            u_clean = u.strip()
            v_clean = v.strip()
            pred_clean = pred.strip()
            if u_clean and v_clean and u_clean.lower() != v_clean.lower():
                try:
                    committed = wp.commit_relation(u_clean, pred_clean, v_clean, confidence=conf)
                    if committed:
                        add_ent = getattr(graph, "add_entity", None)
                        if callable(add_ent):
                            add_ent(u_clean, entity_type="Concept")
                            add_ent(v_clean, entity_type="Concept")
                        edges_added += 1
                except Exception as ge:
                    logger.debug("Failed to add relation (%s)-[%s]->(%s): %s", u_clean, pred_clean, v_clean, ge)

        stop_node_words = (
            LanguageAssetManager.get_instance().get_combined_stopwords(("pl", "en"))
            | {"bezpiecznego", "opcja", "np", "user", "agent", "brak"}
        )

        def _extract_noun_phrase(token: Any) -> str:
            """Ekstrahuje nominalną frazę rzeczownikową z użyciem doc.noun_chunks lub drzewa zależności."""
            try:
                if hasattr(token.doc, "noun_chunks"):
                    for chunk in token.doc.noun_chunks:
                        if token in chunk:
                            return chunk.text.strip()
            except (NotImplementedError, AttributeError):
                pass

            allowed_deps = (
                "amod", "nmod", "nmod:arg", "nmod:flat", "nmod:poss",
                "appos", "flat", "compound", "det", "case", "fixed"
            )
            tokens = sorted(
                [t for t in token.subtree if t.dep_ in allowed_deps or t == token],
                key=lambda t: t.i
            )
            return " ".join(t.text for t in tokens).strip()

        # 2. Neuro-Symbolic Dependency Parsing via spaCy (V155 Filar I)
        # Ekstrakcja relacji z pełnego drzewa zależności składniowych spaCy (fast_lemma_only=False).
        # Encje stanowią nominalne frazy rzeczownikowe, a predykat wynika z rdzenia czasownikowego (ROOT/VERB).
        try:
            from atlas_memory.core.linguistics import get_spacy_nlp
            nlp = get_spacy_nlp("pl", fast_lemma_only=False)
            if nlp is not None:
                for line in content.splitlines():
                    line_s = line.strip()
                    if not line_s or len(line_s) < 5:
                        continue
                    # Ignoruj surowe symbole i separatory pozbawione struktury gramatycznej
                    if line_s.startswith(("-", "*", "•")) and ":" in line_s:
                        continue
                    doc = nlp(line_s)
                    for token in doc:
                        if token.pos_ in ("VERB", "AUX"):
                            pred = token.lemma_.lower().strip()
                            if not pred:
                                continue
                            subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubj:pass", "csubj")]
                            objects = [c for c in token.children if c.dep_ in ("obj", "dobj", "xcomp")]
                            for s in subjects:
                                for o in objects:
                                    s_phrase = _extract_noun_phrase(s)
                                    o_phrase = _extract_noun_phrase(o)
                                    if len(s_phrase) >= 2 and len(o_phrase) >= 2:
                                        if s_phrase.lower() not in stop_node_words and o_phrase.lower() not in stop_node_words:
                                            # Pewność odzwierciedla pewność parsera (0.95 dla ROOT, 0.80 dla gałęzi podrzędnych)
                                            conf = 0.95 if token.dep_ == "ROOT" else 0.80
                                            _add_edge(s_phrase, pred, o_phrase, conf=conf)
        except Exception as exc:
            logger.debug("[MnemosyneIngest] spaCy dependency extraction skipped or failed: %s", exc)

        # 3. Dependency keywords & domain rules via pluggable DomainRelationExtractor (V83 F1)
        active_extractor = extractor or cls.DEFAULT_EXTRACTOR
        active_extractor.extract_and_apply(clean_sub, content, _add_edge, stop_node_words)

        # 5. Bullet points / key-value specs
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        for line in lines[1:]:
            if line.startswith(("-", "*", "•")):
                line_body = line.lstrip("-*• ").strip()
                if ":" in line_body:
                    b_key, b_val = line_body.split(":", 1)
                    b_key_clean = b_key.strip()
                    b_val_clean = b_val.strip()
                    if len(b_key_clean) >= 2 and len(b_val_clean) >= 2:
                        _add_edge(clean_sub, "defines_item", b_key_clean)
                        _add_edge(b_key_clean, "has_spec", b_val_clean)

        return edges_added

    def sync_memory_md_into_engine(self, engine: Any, path: Optional[Path | str] = None) -> Dict[str, Any]:
        """Syncs all entries from MEMORY.md and USER.md into ATLAS engine KV store, Vector Store, and Graph.

        Tożsamość wpisu pamięci zewnętrznej (MEMORY.md / USER.md):
        Wybór: stabilna tożsamość oparta na parze (origin, section_id), gdzie origin = hermes_memory_md / hermes_user_md,
        a section_id to kanoniczny temat sekcji (subject), z kluczem KV postaci `hermes:{origin_prefix}:{rec.subject}`.
        Uzasadnienie:
        1. Tożsamość konceptualna: Wpis w MEMORY.md reprezentuje trwałą domenę wiedzy (np. konfiguracja Obsidiana,
           protokół Klow Stack, reguły raportowania). Treść sekcji podlega ewolucji i edycjom w czasie.
        2. Gdyby kluczem był hash treści sha256(text), każda modyfikacja (np. zmiana portu, aktualizacja wersji)
           generowałaby nowy klucz, pozostawiając poprzednią wersję jako zombie w bazie KV i wektorach (brak supersede).
        3. Stabilny klucz (origin, section_id) zapewnia, że edycja pliku markdown aktualizuje (supersede) dokładnie ten sam
           rekord w KV, Qdrant (upsert_key=var_key) oraz w Kùzu, nie tworząc bliźniaka ani nie zanieczyszczając bazy.
        4. Idempotencja: Ponowny ingest tej samej sekcji (o niezmienionym haszu treści) jest wykrywany jako NO-OP:
           nie dodaje nowych wpisów do KV ani audytu, nie reindeksuje Qdranta i nie dodaje duplikatów do grafu.
        """
        import hashlib

        records = self.extract_memory_md_records(path)
        if not records:
            return {"status": "ok", "ingested": 0, "updated": 0, "source": "memories"}

        count = 0
        updated = 0
        vectors_indexed = 0
        graph_edges_added = 0

        for rec in records:
            origin_prefix = "user_md" if rec.metadata.get("origin") == "hermes_user_md" else "memory_md"
            var_key = f"hermes:{origin_prefix}:{rec.subject}"

            # Oblicz deterministyczny hash znormalizowanej treści
            raw_text = str(rec.object).strip()
            norm_content = re.sub(r"\s+", " ", raw_text)
            content_hash = hashlib.sha256(norm_content.encode("utf-8")).hexdigest()
            rec.metadata["content_hash"] = content_hash

            # Idempotency check przeciwko aktualnemu stanowi KV Store
            existing = None
            if hasattr(engine, "kv") and engine.kv is not None:
                try:
                    existing = engine.kv.get_sync(var_key)
                except Exception as ex:
                    logger.debug("Failed to check existing key %s in KV: %s", var_key, ex)

            if existing is not None:
                existing_val = existing.get("value") if isinstance(existing, dict) else getattr(existing, "value", None)
                existing_norm = re.sub(r"\s+", " ", str(existing_val).strip()) if existing_val is not None else ""
                existing_meta = existing.get("metadata", {}) if isinstance(existing, dict) else {}
                existing_hash = existing_meta.get("content_hash") if isinstance(existing_meta, dict) else None

                if (content_hash == existing_hash) or (norm_content == existing_norm):
                    # IDEMPOTENT NO-OP: Treść sekcji w bazie jest dokładnie identyczna.
                    # Pomijamy ponowny zapis KV (brak zanieczyszczania state_audit_log),
                    # pomijamy reindeksację Qdrant i dodawanie krawędzi w Kùzu.
                    continue
                else:
                    is_update = True
            else:
                is_update = False

            # 1 & 2. Write unconditionally through EpistemicWritePath choke point (F1)
            write_path = getattr(engine, "write_path", None) or EpistemicWritePath(
                kv_store=getattr(engine, "kv", None),
                vector_store=getattr(engine, "vector_store", None),
                graph_store=getattr(engine, "graph", None),
                conflict_arbiter=getattr(engine, "conflict_arbiter", None),
            )
            try:
                reason = "sync_memory_md_update" if is_update else "sync_memory_md"
                wp_res = write_path.write(
                    key=var_key,
                    value=rec.object,
                    confidence=1.0,
                    metadata=rec.metadata,
                    source_type="user_explicit",
                    reason=reason,
                    vector_record=rec,
                )
                if is_update:
                    updated += 1
                else:
                    count += 1
                if wp_res.get("status") == "ok":
                    vectors_indexed += 1
                    graph_edges_added += int(wp_res.get("edges_added", 0))
            except Exception as exc:
                logger.debug("WritePath failed for %s: %s", var_key, exc)

        unchanged = max(0, len(records) - count - updated)
        if count > 0 or updated > 0 or graph_edges_added > 0:
            logger.info(
                "[INGEST:MEMORY_MD] sections=%d | new=%d updated=%d unchanged=%d edges_added=%d",
                len(records), count, updated, unchanged, graph_edges_added,
            )
        else:
            logger.debug(
                "[INGEST:MEMORY_MD] sections=%d | new=%d updated=%d unchanged=%d edges_added=%d (no-op)",
                len(records), count, updated, unchanged, graph_edges_added,
            )
        return {
            "status": "ok",
            "ingested": count,
            "updated": updated,
            "unchanged": unchanged,
            "vectors_indexed": vectors_indexed,
            "graph_edges_added": graph_edges_added,
            "total_sections": len(records),
            "source": "memories",
        }

    def extract_records(self, include_memory_md: bool = True) -> List[MemoryRecord]:
        """Extracts all facts and working memory items from Mnemosyne SQLite database."""
        records: List[MemoryRecord] = []

        # BUG-072: MEMORY.md i USER.md są obsługiwane WYŁĄCZNIE przez sync_memory_md_into_engine(),
        # która nadaje im kanoniczny klucz 'hermes:<origin>:<subject>'. Usunięto extract_memory_md_records()
        # z tej ścieżki, ponieważ sync_into_atlas_engine() zapisywał te same sekcje pod prefiksem
        # 'fact:<subject>:stated_memory', tworząc 54 dokładne duplikaty w bazie produkcyjnej.
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
                    content = (row["content"] or "").strip()
                    importance = float(row["importance"] or 0.7)
                    session_id = row["session_id"] or HERMES_DEFAULT_SESSION
                    created_at = row["created_at"] or ""

                    # Filter out raw conversational chatter, prompt headers, and low-importance dialogue turns
                    if not content or content.startswith(("[USER]", "[ASSISTANT]", "System:", "##", "User:")) or importance < 0.8:
                        continue

                    # Extract concise semantic subject
                    if ":" in content and len(content.split(":", 1)[0].split()) <= 4:
                        subject = content.split(":", 1)[0].strip()
                    else:
                        words = [w for w in content.split() if not w.startswith(("[", "<", "@"))]
                        subject = " ".join(words[:4]) if words else "working_fact"

                    # V151 (Wektor I, H2): bariera dyskursywna — zaimek/wyraz
                    # pytający jako head tematu nie jest encją (zero zapisów do KV).
                    if is_discourse_garbage_subject(subject):
                        logger.info("[V151] Wiersz working_memory odcięty przez barierę dyskursywną: subject=%r", subject)
                        continue

                    records.append(
                        MemoryRecord(
                            subject=subject,
                            predicate="working_fact",
                            object=content,
                            importance_score=min(1.0, max(0.1, importance)),
                            source_type=EpistemicSource.USER_EXPLICIT if importance > 0.85 else EpistemicSource.TOOL_OUTPUT,
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
                    obj_str = str(row["object"] or "").strip()
                    if obj_str.lower() in ("set", "implemented", "true", "false", "ok", "none"):
                        continue
                    records.append(
                        MemoryRecord(
                            subject=str(row["subject"]).strip(),
                            predicate=str(row["predicate"]).strip(),
                            object=str(row["object"]),
                            confidence=float(row["confidence"] or 1.0),
                            source_type=EpistemicSource.EXTERNAL_DOC,
                            source_session_id=row["session_id"] or HERMES_DEFAULT_SESSION,
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
                    # V151 (H2): bariera dyskursywna na canonical_facts.name.
                    if is_discourse_garbage_subject(name):
                        logger.info("[V151] Wiersz canonical_facts odcięty przez barierę dyskursywną: name=%r", name)
                        continue
                    records.append(
                        MemoryRecord(
                            subject=f"{category}:{name}",
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
                    # V151 (H2): bariera dyskursywna na episodic_memory (subject = first 4 words).
                    if is_discourse_garbage_subject(subject):
                        logger.info("[V151] Wiersz episodic_memory odcięty przez barierę dyskursywną: subject=%r", subject)
                        continue
                    records.append(
                        MemoryRecord(
                            subject=subject,
                            predicate="episodic_event",
                            object=content,
                            importance_score=float(row["importance"] or 0.6),
                            source_type=EpistemicSource.AGENT_INFERENCE,
                            source_session_id=row["session_id"] or HERMES_DEFAULT_SESSION,
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

        logger.debug("Extracted %d total records from Mnemosyne SQLite", len(records))
        return records

    def sync_into_atlas_engine(
        self,
        engine: Any,
        seed_triples: Optional[List[Tuple[str, str, str, float]]] = None,
        skip_memory_md: bool = False,
    ) -> Dict[str, Any]:
        """Ingests all extracted records directly into the Atlas HybridMemoryEngine."""
        records = [
            r for r in self.extract_records(include_memory_md=not skip_memory_md)
            if (r.metadata or {}).get("origin") not in ("hermes_memory_md", "hermes_user_md") or not skip_memory_md
        ]
        if not records:
            return {"ingested": 0, "status": "no_records_or_missing_sources"}

        ingested_count = 0
        kv_count = 0
        kv_new_inserted = 0
        kv_updated = 0
        graph_count = 0

        for rec in records:
            try:
                clean_sub = re.sub(r'[^a-zA-Z0-9_:]', '_', rec.subject.strip().lower()).strip('_')
                clean_pred = re.sub(r'[^a-zA-Z0-9_:]', '_', rec.predicate.strip().lower()).strip('_')
                var_key = f"fact:{clean_sub}:{clean_pred}"

                # Filter out raw conversation chatter or prompt noise
                if var_key.startswith(("fact:conversation___user", "fact:conversation", "fact:user_", "fact:assistant_")):
                    continue

                # 1 & 3. Write unconditionally through EpistemicWritePath choke point (F1)
                write_path = getattr(engine, "write_path", None) or EpistemicWritePath(
                    kv_store=getattr(engine, "kv", None),
                    vector_store=getattr(engine, "vector_store", None),
                    graph_store=getattr(engine, "graph", None),
                    conflict_arbiter=getattr(engine, "conflict_arbiter", None),
                )
                meta = dict(rec.metadata or {})
                meta["source_type"] = rec.source_type.value
                meta["confidence"] = rec.confidence
                meta["importance_score"] = rec.importance_score

                existing = engine.kv.get_sync(var_key) if hasattr(engine, "kv") and engine.kv is not None else None
                if existing is not None:
                    existing_val = existing.get("value") if isinstance(existing, dict) else getattr(existing, "value", None)
                    existing_norm = re.sub(r"\s+", " ", str(existing_val).strip()) if existing_val is not None else ""
                    norm_content = re.sub(r"\s+", " ", str(rec.object).strip())
                    if norm_content == existing_norm:
                        kv_count += 1
                        ingested_count += 1
                        continue
                    kv_updated += 1
                else:
                    kv_new_inserted += 1

                wp_res = write_path.write(
                    key=var_key,
                    value=rec.object,
                    confidence=rec.confidence,
                    metadata=meta,
                    source_type=rec.source_type.value,
                    vector_record=rec,
                    reason="mnemosyne_ingest",
                )
                kv_count += 1
                if isinstance(wp_res, dict):
                    graph_count += int(wp_res.get("edges_added", 0))
                if hasattr(engine, "graph") and engine.graph is not None and not hasattr(engine.graph, "upsert_passage"):
                    if not rec.subject.startswith(("[", "<", "working_fact", "episodic_event", "conversation___user", "conversation")) and len(rec.subject.strip()) >= 3:
                        clean_subj = re.sub(r'[^a-zA-Z0-9_:]', '_', rec.subject.strip()).strip('_')[:64]
                        clean_obj = re.sub(r'[^a-zA-Z0-9_:]', '_', str(rec.object).strip()).strip('_')[:64]
                        clean_pred = re.sub(r'[^a-zA-Z0-9_:]', '_', rec.predicate.strip()).strip('_')[:48]
                        if clean_subj and clean_obj and clean_pred and not clean_subj.startswith(("conversation___user", "conversation", "fact_conversation")):
                            try:
                                if write_path.commit_relation(clean_subj, clean_pred, clean_obj, confidence=float(rec.confidence)):
                                    if hasattr(engine.graph, "add_entity"):
                                        engine.graph.add_entity(clean_subj, entity_type="Concept")
                                        engine.graph.add_entity(clean_obj, entity_type="Observation")
                                    graph_count += 1
                            except Exception as g_exc:
                                logger.debug("Graph insert error for %s: %s", clean_subj, g_exc)

                ingested_count += 1
            except Exception as exc:
                logger.debug("Failed to ingest record %s: %s", rec.subject, exc)

        # Seed optional user-provided causal dependencies if provided
        if hasattr(engine, "graph") and engine.graph is not None and seed_triples:
            write_path = getattr(engine, "write_path", None) or EpistemicWritePath(graph_store=engine.graph)
            for s, p, o, conf in seed_triples:
                try:
                    if write_path.commit_relation(s, p, o, confidence=conf):
                        graph_count += 1
                except Exception as dt_exc:
                    logger.debug("Seed triple insert error: %s", dt_exc)

        unchanged_mnemo = max(0, ingested_count - kv_new_inserted - kv_updated)
        by_origin = {}
        for r in records:
            orig = (r.metadata or {}).get("origin", "unknown").replace("mnemosyne_", "")
            by_origin[orig] = by_origin.get(orig, 0) + 1
        origin_str = ", ".join(f"{k}={v}" for k, v in sorted(by_origin.items()))
        if kv_new_inserted > 0 or kv_updated > 0 or graph_count > 0:
            logger.info(
                "[INGEST:MNEMOSYNE] total=%d (%s) | new=%d updated=%d unchanged=%d edges_added=%d",
                len(records), origin_str, kv_new_inserted, kv_updated, unchanged_mnemo, graph_count,
            )
        else:
            logger.debug(
                "[INGEST:MNEMOSYNE] total=%d (%s) | new=%d updated=%d unchanged=%d edges_added=%d (no-op)",
                len(records), origin_str, kv_new_inserted, kv_updated, unchanged_mnemo, graph_count,
            )
        return {
            "status": "ok",
            "ingested": ingested_count,
            "total_extracted": len(records),
            "ingested_records": ingested_count,
            "kv_count": kv_count,
            "kv_records_synced": kv_count,
            "kv_new_inserted": kv_new_inserted,
            "kv_updated": kv_updated,
            "kv_unchanged": unchanged_mnemo,
            "graph_count": graph_count,
            "graph_triples_synced": graph_count,
        }
