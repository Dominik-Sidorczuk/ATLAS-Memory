#!/usr/bin/env python3
"""ATLAS KV Deduplication Tool (BUG-072).

Scans `state_variables` in atlas.db for records with exact identical content,
reports duplicate groups with their content hashes and candidate keys,
and safely cleans up duplicate twin records (e.g. legacy fact:...:stated_memory
twins of canonical hermes:... records).

Default mode is strictly DRY-RUN. Deletions are executed only when `--apply`
is explicitly specified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from atlas_memory.l2_semantic.kv_store import VerifiedKVStore  # noqa: E402
from atlas_memory.l2_semantic.qdrant_store import QdrantVectorStore  # noqa: E402
from atlas_memory.l2_semantic.vector_backend import load_vector_backend  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("atlas_dedup_kv")


class DedupResult(int):
    """V72.7: Wynik deduplikacji z osobnymi licznikami usuniętych rekordów KV i wektorów.

    Zachowuje kompatybilność z `int` (dla asercji `assert deleted == 1`) oraz krotek `(kv, vec)`.
    """

    def __new__(cls, kv_deleted: int, vectors_deleted: int):
        obj = super().__new__(cls, kv_deleted)
        obj.kv_deleted = kv_deleted
        obj.vectors_deleted = vectors_deleted
        return obj

    def __iter__(self):
        yield self.kv_deleted
        yield self.vectors_deleted

    def __repr__(self) -> str:
        return f"DedupResult(kv_deleted={self.kv_deleted}, vectors_deleted={self.vectors_deleted})"

DEFAULT_DB_PATH = Path.home() / ".hermes" / "atlas" / "atlas.db"
DEFAULT_QDRANT_PATH = Path.home() / ".hermes" / "atlas" / "qdrant"


def normalize_content(val: Any) -> str:
    """Normalizes value string or JSON into consistent whitespace representation."""
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        s = json.dumps(val, sort_keys=True, ensure_ascii=False)
    else:
        s = str(val).strip()
        if s.startswith('"') and s.endswith('"') and len(s) >= 2:
            try:
                s = json.loads(s)
            except Exception:
                pass
    return re.sub(r"\s+", " ", str(s).strip())


def compute_content_hash(normalized_text: str) -> str:
    """Computes deterministic SHA-256 hash of normalized text."""
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def rank_key_for_retention(key: str, timestamp: float = 0.0) -> Tuple[int, float, int]:
    """Ranks keys to decide which one to KEEP in a duplicate group.

    Priority order:
    1. Canonical hermes:memory_md:* / hermes:user_md:* (highest priority)
    2. Other hermes:* keys
    3. fact:* or other keys
    Ties broken by newest timestamp and longer subject length.
    """
    if key.startswith(("hermes:memory_md:", "hermes:user_md:")):
        tier = 3
    elif key.startswith("hermes:"):
        tier = 2
    else:
        tier = 1
    return (tier, timestamp, len(key))


def find_duplicates(db_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Scans database and returns groups of duplicate records keyed by content_hash."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("SELECT key, value, confidence, timestamp, metadata FROM state_variables")
        rows = cur.fetchall()
    finally:
        conn.close()

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        norm = normalize_content(r["value"])
        if not norm:
            continue
        c_hash = compute_content_hash(norm)
        groups[c_hash].append({
            "key": r["key"],
            "value": r["value"],
            "norm": norm,
            "confidence": float(r["confidence"] or 1.0),
            "timestamp": float(r["timestamp"] or 0.0),
            "metadata": r["metadata"],
        })

    # Return only groups with more than 1 entry
    return {h: recs for h, recs in groups.items() if len(recs) > 1}


def plan_deduplication(dup_groups: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], int]:
    """Plans which key to KEEP and which keys to DELETE for each group."""
    plan = []
    total_deletes = 0

    for c_hash, recs in dup_groups.items():
        sorted_recs = sorted(
            recs,
            key=lambda r: rank_key_for_retention(r["key"], r["timestamp"]),
            reverse=True,
        )
        keep = sorted_recs[0]
        to_delete = sorted_recs[1:]
        total_deletes += len(to_delete)

        plan.append({
            "content_hash": c_hash,
            "sample_snippet": keep["norm"][:80],
            "keep_key": keep["key"],
            "delete_keys": [d["key"] for d in to_delete],
            "all_keys": [r["key"] for r in recs],
        })

    return plan, total_deletes


def execute_deduplication(
    db_path: Path,
    qdrant_path: Optional[Path],
    plan: List[Dict[str, Any]],
) -> DedupResult:
    """Executes deletion of duplicate keys using VerifiedKVStore and QdrantVectorStore.

    V72.7 (FINDING-075):
    - Wczytuje konfigurację backendu wektorowego (load_vector_backend), aby używać właściwej kolekcji.
    - Jeśli qdrant_path jest podane, awaria inicjalizacji lub kasowania wektorów NIE jest
      cicho tłumiona — podnosi błąd RuntimeError.
    - Usuwa wektory zarówno po kluczu dokładnym, jak i prefiksowo (dla uciętych kluczy KV).
    - Chroni hermes:* przed kasowaniem prefiksowym.
    - Zwraca DedupResult(kv_deleted, vectors_deleted).
    """
    kv = VerifiedKVStore(db_path=str(db_path))
    v_store = None
    if qdrant_path is not None:
        qp = Path(qdrant_path)
        if not qp.exists():
            raise FileNotFoundError(f"Qdrant directory not found at {qp}")
        try:
            backend = load_vector_backend()
            v_store = QdrantVectorStore(
                collection_name=backend.collection,
                dimension=backend.dimension,
                path=str(qp),
            )
        except Exception as exc:
            logger.error("Could not initialize QdrantVectorStore at %s: %s", qp, exc)
            raise RuntimeError(f"Failed to initialize QdrantVectorStore at {qp}: {exc}") from exc

    kv_deleted = 0
    vectors_deleted = 0
    try:
        for item in plan:
            for del_key in item["delete_keys"]:
                logger.info("Deleting duplicate key '%s' (keeping '%s')...", del_key, item["keep_key"])
                # 1. KV Store deletion (with SHA-256 audit entry)
                ok = kv.delete_sync(del_key)
                if ok:
                    kv_deleted += 1

                # 2. Qdrant vector deletion (by upsert_key and prefix)
                if v_store is not None:
                    try:
                        # Exact deletions
                        d1 = v_store.delete_by_upsert_key_sync(del_key)
                        d2 = v_store.delete_by_upsert_key_sync(f"kv:{del_key}")
                        vectors_deleted += (d1 + d2)

                        # Prefix deletions (only for non-canonical keys to catch keys truncated in KV)
                        if not del_key.startswith("hermes:") and not del_key.startswith("kv:hermes:"):
                            p1 = v_store.delete_by_upsert_key_prefix_sync(del_key)
                            p2 = v_store.delete_by_upsert_key_prefix_sync(f"kv:{del_key}")
                            vectors_deleted += (p1 + p2)
                    except Exception as ve:
                        logger.error("Qdrant vector deletion failed for key '%s': %s", del_key, ve)
                        raise RuntimeError(f"Vector deletion failed for key '{del_key}': {ve}") from ve

        return DedupResult(kv_deleted, vectors_deleted)
    finally:
        if v_store is not None:
            v_store.close()
        kv.close_sync()


def main() -> None:
    parser = argparse.ArgumentParser(description="ATLAS KV Store Deduplication Tool")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH), help="Path to atlas.db")
    parser.add_argument("--qdrant-path", type=str, default=str(DEFAULT_QDRANT_PATH), help="Path to qdrant storage")
    parser.add_argument("--no-qdrant", action="store_true", help="Skip Qdrant vector deletion")
    parser.add_argument("--apply", action="store_true", help="Execute deletions (default: dry-run)")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Dry-run preview mode (default)")
    args = parser.parse_args()

    # If --apply is explicitly specified, disable dry-run
    is_apply = args.apply
    db_file = Path(os.path.expanduser(args.db_path))
    if args.no_qdrant or not args.qdrant_path:
        qdrant_dir = None
    else:
        qdrant_dir = Path(os.path.expanduser(args.qdrant_path))

    logger.info("Scanning database: %s", db_file)
    dup_groups = find_duplicates(db_file)
    plan, total_deletes = plan_deduplication(dup_groups)

    print("=" * 80)
    print(" 🔍 ATLAS KV DEDUPLICATION REPORT")
    print("=" * 80)
    print(f"Total duplicate groups: {len(plan)}")
    print(f"Total records to delete: {total_deletes}")
    print("-" * 80)

    for idx, item in enumerate(plan, 1):
        if idx <= 5 or idx > len(plan) - 2:
            print(f"Group {idx:2d} [hash: {item['content_hash'][:16]}...] snippet: \"{item['sample_snippet']}...\"")
            print(f"  KEEP:   {item['keep_key']}")
            for dk in item["delete_keys"]:
                print(f"  DELETE: {dk}")
        elif idx == 6:
            print(f"  ... ({len(plan) - 7} groups omitted for brevity) ...")

    print("-" * 80)

    if not is_apply:
        print("💡 MODE: DRY-RUN (no changes made to database).")
        print("   To execute deletions, pass `--apply`.")
        print("=" * 80)
        sys.exit(0)

    print("⚠️  MODE: --apply ACTIVATED. Executing deletions...")
    try:
        res = execute_deduplication(db_file, qdrant_dir, plan)
        print(f"✅ Successfully deleted {res.kv_deleted} duplicate KV records and {res.vectors_deleted} Qdrant vectors.")
        print("=" * 80)
    except Exception as exc:
        logger.error("❌ Deduplication execution failed: %s", exc)
        print(f"❌ Deduplication execution failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
