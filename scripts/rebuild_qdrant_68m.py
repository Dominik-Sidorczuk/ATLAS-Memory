#!/usr/bin/env python3
"""Budowa NOWEJ kolekcji 512D (PolDense-68M) z KV — bez odbudowy starego 384D.

Dlaczego nie migracja punkt-po-punkcie: stara kolekcja `hermes_episodic_memory` ma
9969 punktów, ale tylko 511 unikalnych tekstów. Źródłem prawdy jest KV
(`state_variables`, 639 rekordów), a warstwa wektorowa narosła przez dopisywanie
(ingest Mnemosyne woła `insert_sync` bez `upsert_key` → id z timestampu → duplikaty).

Ten skrypt stawia czystą bazę 512D:
  * czyta KV (read-only) i pomija rekordy oznaczone jako superseded,
  * normalizuje tekst (V70.2: literalne `\\uXXXX` → znaki),
  * zapisuje przez `insert_sync(..., upsert_key="kv:<klucz>")` → ID deterministyczne,
    więc powtórny przebieg jest idempotentny, a nie mnoży punktów.

Stara kolekcja 384D pozostaje nietknięta = rollback przez usunięcie
`vector_backend.json`. Wymaga ZATRZYMANEGO daemona (jeden właściciel magazynu).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas_memory.core.models import MemoryRecord  # noqa: E402
from atlas_memory.l2_semantic.onnx_encoder import OnnxDenseEncoder  # noqa: E402
from atlas_memory.l2_semantic.qdrant_store import QdrantVectorStore  # noqa: E402
from atlas_memory.l2_semantic.textnorm import normalize_stored_text  # noqa: E402

DEFAULT_DB = Path.home() / ".hermes" / "atlas" / "atlas.db"
DEFAULT_QDRANT_DIR = Path.home() / ".hermes" / "atlas" / "qdrant"
DEFAULT_COLLECTION = "hermes_episodic_memory_68m"


def iter_kv_rows(db_path: Path, skip_superseded: bool) -> Tuple[List[Tuple[str, str, float, float, Dict[str, Any]]], int]:
    """Zwraca (rekordy, liczba pominiętych superseded) z `state_variables`."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT key, value, confidence, timestamp, metadata FROM state_variables"
        ).fetchall()
    finally:
        con.close()

    out: List[Tuple[str, str, float, float, Dict[str, Any]]] = []
    superseded = 0
    for key, value, confidence, timestamp, metadata in rows:
        meta: Dict[str, Any] = {}
        if metadata:
            try:
                parsed = json.loads(metadata)
                if isinstance(parsed, dict):
                    meta = parsed
            except (TypeError, json.JSONDecodeError):
                meta = {}
        if skip_superseded and (meta.get("is_superseded") or meta.get("superseded_by")):
            superseded += 1
            continue
        try:
            text = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            text = value
        if not isinstance(text, str) or not text.strip():
            continue
        out.append((str(key), normalize_stored_text(text), float(confidence or 1.0), float(timestamp or 0.0), meta))
    return out, superseded


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--qdrant-dir", default=str(DEFAULT_QDRANT_DIR))
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--dimension", type=int, default=512)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--keep-superseded", action="store_true", help="nie pomijaj rekordów unieważnionych")
    ap.add_argument("--report", default="/tmp/atlas_rebuild_68m_report.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    records, superseded = iter_kv_rows(Path(args.db).expanduser(), skip_superseded=not args.keep_superseded)
    report: Dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "db": str(args.db),
        "collection": args.collection,
        "dimension": args.dimension,
        "records_to_embed": len(records),
        "superseded_skipped": superseded,
    }
    print(f"[1/4] KV: {len(records)} rekordów do embeddingu, pominięte superseded: {superseded}")
    if args.dry_run:
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[dry-run] nic nie zapisano")
        return 0

    print(f"[2/4] otwieram nową kolekcję {args.collection} ({args.dimension}D) w {args.qdrant_dir}")
    encoder = OnnxDenseEncoder(dimension=args.dimension, model_dir=args.model_dir)
    store = QdrantVectorStore(
        collection_name=args.collection,
        location=str(Path(args.qdrant_dir).expanduser()),
        dimension=args.dimension,
        encoder=encoder,
        owner="atlas_rebuild_68m",
    )
    before = store.client.count(collection_name=args.collection, exact=True).count

    print(f"[3/4] embedding + zapis ({len(records)} rekordów, id deterministyczne kv:<klucz>)")
    t0 = time.time()
    written = 0
    for key, text, confidence, timestamp, meta in records:
        record = MemoryRecord(
            subject=f"kv:{key}",
            predicate="state_is",
            object=text,
            confidence=confidence,
            timestamp=timestamp,
            is_state_variable=True,
            metadata={**meta, "kv_key": key},
        )
        store.insert_sync(record, raw_text=text, upsert_key=f"kv:{key}")
        written += 1
        if written % 50 == 0:
            print(f"      {written}/{len(records)}  {time.time()-t0:.1f}s")
    report["written"] = written
    report["seconds"] = round(time.time() - t0, 2)
    report["ms_per_record"] = round(1000 * report["seconds"] / max(written, 1), 1)

    print("[4/4] weryfikacja kolekcji")
    info = store.client.get_collection(args.collection)
    report["points_before"] = before
    report["points_after"] = info.points_count
    vectors_cfg: Any = info.config.params.vectors
    report["collection_dim"] = int(
        vectors_cfg.size if hasattr(vectors_cfg, "size") else list(vectors_cfg.values())[0].size
    )
    probe, _ = store.client.scroll(collection_name=args.collection, limit=1, with_payload=False, with_vectors=True)
    probe_vec = probe[0].vector if probe and probe[0].vector is not None else None
    if isinstance(probe_vec, dict):
        dense_part = probe_vec.get("dense") or probe_vec.get("") or next(iter(probe_vec.values()), None)
        report["probe_vector_dim"] = len(dense_part) if isinstance(dense_part, list) else None
    elif isinstance(probe_vec, list):
        report["probe_vector_dim"] = len(probe_vec)
    else:
        report["probe_vector_dim"] = None
    store.close()
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("points_after", "collection_dim", "probe_vector_dim", "ms_per_record")}, ensure_ascii=False))
    print(f"RAPORT: {args.report}")
    ok = report["collection_dim"] == args.dimension and report["probe_vector_dim"] == args.dimension
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
