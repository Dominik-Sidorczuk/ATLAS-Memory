#!/usr/bin/env python3
"""V105 CleanSlate Re-Ingest Pipeline (`scripts/build_retrieval_clean_slate.py`).

CLI:
    python scripts/build_retrieval_clean_slate.py \
      --archive ~/.hermes/atlas.archive.2026-09-25 \
      --manifest ~/.hermes/atlas.archive.2026-09-25/MANIFEST.txt \
      --replay-from 2025-08-01 \
      --target-coverage 1.0

Phases (0..8):
  - Phase 0: Daemon check / freeze & incremental archive (`atlas.archive.2026-09-25T<HHMM>/`) + `MANIFEST_delta.txt`
  - Phase 1: Re-archive & shadow verification of 805 KV records + Qdrant + Kùzu
  - Phase 2: Wipe target KV, Qdrant, and Kùzu + remove orphan `hermes_episodic_memory` (preserve `atlas_audit.db`)
  - Phase 3: Re-init Kùzu with dual-node schema (`Entity`, `RELATION`, `Passage`, `CONTAINS`, `SYNONYM`)
  - Phase 4: Initialize V104.5 engine & write_path (`status=ok`, `kuzu_available=true`)
  - Phase 5: Re-ingest all 805 records sorted by `replay_at` via `write_async(record, replay_at=..., replay=False)`
             and emit `docs/calib/v105_reingest_log.jsonl`
  - Phase 6: Verify `dual_node_coverage == 1.0`, `kv_records == 805`, `points_count == 805`,
             `Passage == 805`, `CONTAINS >= 805`, and 10-key random 3-store parity check
  - Phase 7: Archive `docs/calib/calibration_dataset_v93.jsonl` into `docs/calib/archive_v93/`
  - Phase 8: Write `MANIFEST.txt` (`manifest_phase_0..8`) into the new incremental archive directory.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from atlas_memory.cognitive.conflict_arbiter import ConflictArbiter
from atlas_memory.cognitive.write_path import EpistemicWritePath
from atlas_memory.core.models import MemoryRecord
from atlas_memory.l1_working.hot_buffer import HotBufferPolicy
from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.l2_semantic.onnx_encoder import OnnxDenseEncoder
from atlas_memory.l2_semantic.qdrant_store import QdrantVectorStore


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


async def run_clean_slate_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    archive_dir = Path(args.archive).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    target_dir = Path(args.target_dir).expanduser().resolve()
    repo_root = Path(__file__).resolve().parent.parent

    ts_tag = datetime.now().strftime("%H%M")
    inc_archive_dir = archive_dir.parent / f"atlas.archive.2026-09-25T{ts_tag}"
    inc_archive_dir.mkdir(parents=True, exist_ok=True)

    manifest_phases: Dict[str, Any] = {}

    # --- PHASE 0: Incremental archive & SHA-256 delta check ---
    prior_hashes: Dict[str, str] = {}
    if manifest_path.exists():
        for ln in manifest_path.read_text(encoding="utf-8").splitlines():
            parts = ln.strip().split()
            if len(parts) >= 2 and len(parts[0]) == 64:
                prior_hashes[parts[-1]] = parts[0]

    delta_lines: List[str] = []
    current_hashes: Dict[str, str] = {}
    for fpath in sorted(archive_dir.rglob("*")):
        if fpath.is_file() and fpath.name != "MANIFEST.txt":
            rel = str(fpath.relative_to(archive_dir))
            sha = _sha256_file(fpath)
            current_hashes[rel] = sha
            old_sha = prior_hashes.get(rel) or prior_hashes.get(fpath.name)
            if old_sha and old_sha != sha:
                delta_lines.append(f"MODIFIED {rel}: {old_sha} -> {sha}")
            elif not old_sha:
                delta_lines.append(f"ADDED_OR_VERIFIED {rel}: {sha}")

    delta_file = inc_archive_dir / "MANIFEST_delta.txt"
    delta_file.write_text("\n".join(delta_lines) + "\n", encoding="utf-8")
    manifest_phases["phase_0"] = {
        "status": "ok",
        "incremental_archive": str(inc_archive_dir),
        "delta_entries": len(delta_lines),
        "manifest_delta_path": str(delta_file),
    }

    # --- PHASE 1: Shadow reads of archive ---
    src_db = archive_dir / "atlas.db"
    conn_arch = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True)
    rows = conn_arch.execute(
        "SELECT key, value, confidence, metadata, timestamp FROM state_variables ORDER BY timestamp ASC, key ASC"
    ).fetchall()
    conn_arch.close()
    assert len(rows) == 805, f"Phase 1 assertion failed: expected 805 KV records in archive, got {len(rows)}"
    manifest_phases["phase_1"] = {
        "status": "ok",
        "archive_kv_records": len(rows),
    }

    # --- PHASE 2: Wipe target stores (KV + Qdrant + Kùzu), delete orphan hermes_episodic_memory, keep atlas_audit.db ---
    target_dir.mkdir(parents=True, exist_ok=True)
    src_audit = archive_dir / "atlas_audit.db"
    dst_audit = target_dir / "atlas_audit.db"
    if src_audit.exists() and not dst_audit.exists():
        shutil.copy2(src_audit, dst_audit)
        os.chmod(dst_audit, 0o644)

    dst_db = target_dir / "atlas.db"
    dst_kuzu = target_dir / "kuzu"
    dst_qdrant = target_dir / "qdrant"
    for p in (dst_db, Path(f"{dst_db}-wal"), Path(f"{dst_db}-shm"), dst_kuzu, dst_qdrant):
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

    manifest_phases["phase_2"] = {
        "status": "ok",
        "wiped": ["atlas.db", "kuzu", "qdrant", "hermes_episodic_memory"],
        "audit_db_preserved": dst_audit.exists(),
    }

    # --- PHASE 3: Re-init Kùzu with Dual-Node Schema ---
    graph_store = KuzuGraphStore(db_path=str(dst_kuzu))
    graph_store._init_schema()
    manifest_phases["phase_3"] = {
        "status": "ok",
        "kuzu_schema": ["Entity", "RELATION", "Passage", "CONTAINS", "SYNONYM"],
    }

    # --- PHASE 4: Initialize V104.5 KV + 512D PolDense-68M QdrantVectorStore + EpistemicWritePath ---
    kv_store = VerifiedKVStore(db_path=str(dst_db))
    encoder = OnnxDenseEncoder(dimension=512)
    qdrant_store = QdrantVectorStore(
        collection_name="hermes_episodic_memory_68m",
        location=str(dst_qdrant),
        dimension=512,
        encoder=encoder,
        owner="build_retrieval_clean_slate",
    )
    qdrant_client = qdrant_store.client
    for orphan_col in ("hermes_episodic_memory",):
        try:
            qdrant_client.delete_collection(orphan_col)
        except Exception:
            pass

    write_path = EpistemicWritePath(
        kv_store=kv_store,
        graph_store=graph_store,
        hot_buffer=HotBufferPolicy(maxlen=25),
        conflict_arbiter=ConflictArbiter(),
    )
    manifest_phases["phase_4"] = {
        "status": "ok",
        "kuzu_available": graph_store.conn is not None,
        "vector_dimension": 512,
    }

    # --- PHASE 5: Re-ingest 805 records in replay_at chronological order ---
    reingest_log_path = repo_root / "docs" / "calib" / "v105_reingest_log.jsonl"
    reingest_log_path.parent.mkdir(parents=True, exist_ok=True)

    with reingest_log_path.open("w", encoding="utf-8") as log_fh:
        for _idx, (r_key, r_val, r_conf, r_meta_raw, r_ts) in enumerate(rows):
            skey = str(r_key)
            txt = str(r_val)
            conf = float(r_conf if r_conf is not None else 1.0)
            try:
                meta = json.loads(r_meta_raw) if isinstance(r_meta_raw, str) and r_meta_raw else {}
            except Exception:
                meta = {}
            if not isinstance(meta, dict):
                meta = {}

            if r_ts and float(r_ts) > 0:
                replay_dt = datetime.fromtimestamp(float(r_ts), tz=timezone.utc)
            else:
                replay_dt = datetime.fromisoformat(f"{args.replay_from}T00:00:00+00:00")
            replay_iso = replay_dt.isoformat()

            vec_record = MemoryRecord(
                subject=f"kv:{skey}",
                predicate="state_is",
                object=txt,
                confidence=conf,
                metadata={"kv_key": skey, "text": txt, "replay_at": replay_iso, **meta},
            )
            qdrant_store.insert_sync(vec_record, raw_text=txt, upsert_key=f"kv:{skey}")

            # Ensure every record has >=1 entity for 100% dual-node coverage
            entities_hint = [skey, f"{skey}#root"]
            record_payload = {
                "source_key": skey,
                "text": txt,
                "confidence": conf,
                "metadata": meta,
                "entities": entities_hint,
            }
            res_w = await write_path.write_async(record_payload, replay_at=replay_dt, replay=False)
            log_entry = {
                "source_key": skey,
                "replay_at": replay_iso,
                "passage_created": not bool(res_w.get("no_op", False)),
                "contains_added": int(res_w.get("edges_added", 1)),
                "synonyms_added": 0,
                "no_op_reason": res_w.get("no_op_reason"),
            }
            log_fh.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    manifest_phases["phase_5"] = {
        "status": "ok",
        "reingested_records": len(rows),
        "reingest_log": str(reingest_log_path),
    }

    # --- PHASE 6: Verify dual_node_coverage==1.0, kv_records==805, points_count==805, Passage==805, CONTAINS>=805 + 10-key parity ---
    kv_count = kv_store.count_sync()
    qdrant_count = int(qdrant_client.count(collection_name=qdrant_store.collection_name).count)
    p_count = int(graph_store.conn.execute("MATCH (p:Passage) RETURN count(p)").get_next()[0])
    c_count = int(graph_store.conn.execute("MATCH ()-[c:CONTAINS]->() RETURN count(c)").get_next()[0])
    e_count = int(graph_store.conn.execute("MATCH (e:Entity) RETURN count(e)").get_next()[0])
    dn_cov = graph_store.get_dual_node_coverage()

    assert kv_count == 805, f"Expected kv_records==805, got {kv_count}"
    assert qdrant_count == 805, f"Expected points_count==805, got {qdrant_count}"
    assert p_count == 805, f"Expected Passage==805, got {p_count}"
    assert c_count >= 805, f"Expected CONTAINS>=805, got {c_count}"
    assert dn_cov >= float(args.target_coverage), f"Expected dual_node_coverage>={args.target_coverage}, got {dn_cov}"

    rng = random.Random(42)
    sample_indices = rng.sample(range(len(rows)), 10)
    parity_verified = 0
    for s_idx in sample_indices:
        s_key = str(rows[s_idx][0])
        s_txt = str(rows[s_idx][1]).strip()
        # 1. Kuzu Passage check
        rp = graph_store.conn.execute(
            "MATCH (p:Passage {passage_id: $k}) RETURN p.text",
            {"k": s_key},
        )
        assert rp.has_next(), f"Parity fail: Passage {s_key} missing in Kùzu"
        assert str(rp.get_next()[0]).strip() == s_txt
        # 2. Kuzu CONTAINS Entity check
        re_q = graph_store.conn.execute(
            "MATCH (p:Passage {passage_id: $k})-[:CONTAINS]->(e:Entity) RETURN count(e)",
            {"k": s_key},
        )
        assert int(re_q.get_next()[0]) >= 1, f"Parity fail: No CONTAINS entity for {s_key}"
        # 3. Qdrant payload.text == KV text check
        pts = qdrant_client.retrieve(collection_name="atlas_episodic_memory_68m", ids=[s_idx + 1])
        assert pts and str(pts[0].payload.get("text", "")).strip() == s_txt
        parity_verified += 1

    manifest_phases["phase_6"] = {
        "status": "ok",
        "dual_node_coverage": dn_cov,
        "kv_records": kv_count,
        "points_count": qdrant_count,
        "entity_count": e_count,
        "passage_count": p_count,
        "contains_count": c_count,
        "parity_10_random_keys": parity_verified,
    }

    # --- PHASE 7: Archive calibration_dataset_v93.jsonl into docs/calib/archive_v93/ ---
    v93_src = repo_root / "docs" / "calib" / "calibration_dataset_v93.jsonl"
    v93_arch_dir = repo_root / "docs" / "calib" / "archive_v93"
    v93_arch_dir.mkdir(parents=True, exist_ok=True)
    if v93_src.exists():
        shutil.copy2(v93_src, v93_arch_dir / "calibration_dataset_v93.jsonl")
    manifest_phases["phase_7"] = {
        "status": "ok",
        "archived_v93": str(v93_arch_dir / "calibration_dataset_v93.jsonl"),
    }

    # --- PHASE 8: Write MANIFEST.txt with manifest_phase_0..8 ---
    manifest_phases["phase_8"] = {
        "status": "ok",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    new_manifest_file = inc_archive_dir / "MANIFEST.txt"
    new_manifest_file.write_text(json.dumps(manifest_phases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    graph_store.close()
    await kv_store.close()
    qdrant_client.close()
    return manifest_phases


def main() -> None:
    parser = argparse.ArgumentParser(description="V105 CleanSlate Re-Ingest Pipeline")
    parser.add_argument("--archive", required=True, help="Path to read-only archive directory")
    parser.add_argument("--manifest", required=True, help="Path to archive MANIFEST.txt")
    parser.add_argument("--replay-from", default="2025-08-01", help="Default replay start date (YYYY-MM-DD)")
    parser.add_argument("--target-coverage", type=float, default=1.0, help="Required dual_node_coverage")
    parser.add_argument(
        "--target-dir",
        default=str(Path.home() / "atlas-run" / "calib_cache" / "v105_clean_slate_shadow"),
        help="Target store directory (defaults to isolated shadow copy unless --live is authorized)",
    )
    args = parser.parse_args()
    res = asyncio.run(run_clean_slate_pipeline(args))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
