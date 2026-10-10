#!/usr/bin/env python3
"""V152 (Filar II, warstwa 3): jednorazowy, deterministyczny rebuild widoków.

Trójstronny fizyczny purge (KV -> Qdrant -> Kuzu) rekordow dyskursywnych
(ROI: strict list dyrektywy + is_superseded) oraz seed kanonicznego faktu
HippoRAG 2 przez chokepoint EpistemicWritePath.

Uwaga: skrypt nie definiuje WLASNEJ listy slow — klasyfikacja przez
is_discourse_garbage_subject (istniejacy prymityw) + strict keys dyrektywy
(ROI). Uruchomic PRZY ZATRZYMANYM demonie (Kuzu/Qdrant embedded lock).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from atlas_memory.engine import HybridMemoryEngine  # noqa: E402

PROD_DIR = os.path.expanduser("~/.hermes/atlas")
DB_PATH = os.path.join(PROD_DIR, "atlas.db")
QDRANT_LOC = os.path.join(PROD_DIR, "qdrant")
KUZU_PATH = os.path.join(PROD_DIR, "kuzu")

# ROI z dyrektywy — nie blacklist ogolny: precyzyjnie wskazane klucze produkcyjne.
STRICT_PREFIXES = (
    "jak_obecnie",
    "który_obecnie_nie",
    "nie_ma_porównania_jak",
    "bazą_danych_powiązana",
)

HIPPORAG_KEY = "module:retrieval:hipporag2"
HIPPORAG_PREDICATE = "state_value"
HIPPORAG_VALUE = (
    "HippoRAG 2 to asocjacyjny silnik grafowy oparty na Personalized PageRank (igraph) "
    "na grafie Dual-Node (Passage/Entity) z adaptacyjnym budżetem czasowym oraz "
    "warunkowanym tunelowaniem aksjomatów."
)
AUDIT_LOG_PATH = os.path.join(PROD_DIR, "purge_audit.jsonl")


def _subject_of_key(key: str) -> str:
    seg = key.split(":")
    if len(seg) >= 3 and seg[0] in ("hermes", "kv", "atlas"):
        return seg[1]
    return seg[0] if seg else key


def _is_purge_target(key: str, metadata_raw: str | None) -> bool:
    subj = _subject_of_key(key)
    for p in STRICT_PREFIXES:
        if (
            key == p
            or key.startswith(f"{p}:")
            or key.startswith(f"{p}_")
            or key.startswith(p)
            or subj == p
            or subj.startswith(f"{p}:")
            or subj.startswith(f"{p}_")
            or subj.startswith(p)
        ):
            return True
    # Clean up stale/malformed seed variants
    if key == "module:retrieval:hipporag2:module_engine":
        return True
    try:
        meta = json.loads(metadata_raw or "{}")
    except (TypeError, ValueError):
        meta = {}
    if meta.get("is_superseded") is True or meta.get("tombstone") is True:
        # superseded -> fizyczna ewizja z widokow (Filar II kompensacja)
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="tylko raport, zero zapisu")
    ap.add_argument("--skip-seed", action="store_true", help="pominac seed HippoRAG 2")
    args = ap.parse_args()

    for p in (DB_PATH, QDRANT_LOC, KUZU_PATH):
        if not os.path.exists(p):
            print(f"[FATAL] brak sciezki: {p} (demon musi byc zatrzymany)")
            return 2

    eng = HybridMemoryEngine.create_default(
        db_path=DB_PATH, qdrant_location=QDRANT_LOC, kuzu_path=KUZU_PATH
    )

    conn = eng.kv.ensure_connection()  # sqlite3.Connection (demon zatrzymany)
    rows = conn.execute("SELECT key, value, metadata FROM state_variables").fetchall()

    targets = [(k, v, m) for k, v, m in rows if _is_purge_target(k, m)]
    print(f"[AUDYT] {len(rows)} rekordow KV; cel purgu: {len(targets)}")
    for k, _, _ in targets[:20]:
        print(f"   - {k}")
    if args.dry_run:
        print("[DRY-RUN] zatrzymano przed jakakolwiek mutacja")
        return 0

    audit_records = []
    now_ts = os.popen("date -u +\"%Y-%m-%dT%H:%M:%SZ\"").read().strip()

    purged_kv = purged_vec = purged_graph = 0
    for key, _val, _meta_raw in targets:
        # 1) widok wektorowy — usuwamy po kluczu, kv:klucz i canon
        vec_del_cnt = 0
        for uk in (key, f"kv:{key}", _subject_of_key(key)):
            try:
                removed = eng.vector_store.delete_by_upsert_key_sync(uk)
                vec_del_cnt += int(removed or 0)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] qdrant delete {uk!r}: {exc}")
        purged_vec += vec_del_cnt

        # 2) widok grafowy — entity / passage po kluczu i subjekcie (DETACH DELETE)
        subj = _subject_of_key(key)
        graph_del_cnt = 0
        try:
            with eng.graph._graph_lock:
                if eng.graph.conn is not None:
                    for name_candidate in (subj, key):
                        try:
                            eng.graph.conn.execute(
                                "MATCH (e:Entity) WHERE e.name = $name OR e.name STARTS WITH $name DETACH DELETE e",
                                {"name": name_candidate},
                            )
                            graph_del_cnt += 1
                        except Exception:
                            pass
                        try:
                            eng.graph.conn.execute(
                                "MATCH (p:Passage) WHERE p.passage_id = $name OR p.passage_id STARTS WITH $name OR p.source_key = $name OR p.source_key STARTS WITH $name DETACH DELETE p",
                                {"name": name_candidate},
                            )
                            graph_del_cnt += 1
                        except Exception:
                            pass
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] kuzu delete {key!r}: {exc}")
        purged_graph += graph_del_cnt

        # 3) KV — atomowe usuniecie
        kv_del_ok = False
        try:
            if eng.kv.delete_sync(key):
                kv_del_ok = True
                purged_kv += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] kv delete {key!r}: {exc}")

        audit_records.append({
            "timestamp": now_ts,
            "action": "purge",
            "key": key,
            "kv_deleted": kv_del_ok,
            "vector_deleted_count": vec_del_cnt,
            "graph_deleted_count": graph_del_cnt,
        })

    # Deep explicit sweep for STRICT_PREFIXES across Qdrant and Kuzu
    for p in STRICT_PREFIXES:
        for uk_cand in (p, f"kv:{p}", f"{p}:state_value", f"kv:{p}:state_value"):
            try:
                rem = eng.vector_store.delete_by_upsert_key_sync(uk_cand)
                purged_vec += int(rem or 0)
            except Exception:
                pass
        try:
            with eng.graph._graph_lock:
                if eng.graph.conn is not None:
                    eng.graph.conn.execute(
                        "MATCH (e:Entity) WHERE e.name = $p OR e.name STARTS WITH $p DETACH DELETE e", {"p": p}
                    )
                    eng.graph.conn.execute(
                        "MATCH (pass:Passage) WHERE pass.passage_id = $p OR pass.passage_id STARTS WITH $p OR pass.source_key = $p OR pass.source_key STARTS WITH $p DETACH DELETE pass", {"p": p}
                    )
                    purged_graph += 1
        except Exception:
            pass

    # Purge any stale variants of hipporag2 from Kuzu
    try:
        with eng.graph._graph_lock:
            if eng.graph.conn is not None:
                eng.graph.conn.execute(
                    "MATCH (p:Passage) WHERE p.passage_id CONTAINS 'module_engine' OR p.source_key CONTAINS 'module_engine' DETACH DELETE p"
                )
                eng.graph.conn.execute(
                    "MATCH (e:Entity) WHERE e.name CONTAINS 'module_engine' DETACH DELETE e"
                )
    except Exception as exc:
        print(f"[WARN] kuzu stale hipporag purge note: {exc}")

    print(f"[PURGE] KV={purged_kv} Qdrant-points={purged_vec} Kuzu-entity={purged_graph}")

    if not args.skip_seed:
        from atlas_memory.core.models import MemoryRecord

        vr = MemoryRecord(
            subject=HIPPORAG_KEY,
            predicate=HIPPORAG_PREDICATE,
            object=HIPPORAG_VALUE,
            confidence=1.0,
            timestamp=float(conn.execute("SELECT strftime('%s','now')").fetchone()[0] or 0),
            is_state_variable=True,
            metadata={"kv_key": HIPPORAG_KEY, "origin": "v152_seed", "predicate": HIPPORAG_PREDICATE},
        )

        # 1. KV write
        try:
            eng.kv.set_sync(
                key=HIPPORAG_KEY,
                value=HIPPORAG_VALUE,
                confidence=1.0,
                metadata={"predicate": HIPPORAG_PREDICATE, "source_type": "user_explicit", "origin": "v152_seed"},
            )
            print(f"[SEED] KV set_sync({HIPPORAG_KEY}) -> OK")
        except Exception as exc:
            print(f"[FATAL] KV set_sync failed: {exc}")
            return 3

        # 2. Qdrant direct deterministic write-through
        try:
            raw_text = f"{HIPPORAG_KEY} hipporag2 dual-node PPR: {HIPPORAG_VALUE}"
            eng.vector_store.insert_sync(
                vr,
                raw_text=raw_text,
                upsert_key=HIPPORAG_KEY,
            )
            print(f"[SEED] Qdrant insert_sync({HIPPORAG_KEY}) -> OK")
        except Exception as exc:
            print(f"[FATAL] Qdrant insert_sync failed: {exc}")
            return 4

        # 3. Kuzu dual-node passage & entity relation
        try:
            if hasattr(eng.graph, "index_dual_node_passage"):
                eng.graph.index_dual_node_passage(
                    source_key=HIPPORAG_KEY,
                    text=f"hipporag2 dual-node PPR: {HIPPORAG_VALUE}",
                    explicit_entities=["HippoRAG 2", "hipporag2", "PPR", "Personalized PageRank", "igraph", "Dual-Node"],
                )
                print(f"[SEED] Kuzu index_dual_node_passage({HIPPORAG_KEY}) -> OK")
            eng.graph.add_record_sync(vr)
            print(f"[SEED] Kuzu add_record_sync({HIPPORAG_KEY}) -> OK")
        except Exception as exc:
            print(f"[WARN] Kuzu add_record_sync note: {exc}")

        # Weryfikacja projekcji wektorowej kanonicznego faktu
        got = conn.execute("SELECT COUNT(*) FROM state_variables WHERE key=?", (HIPPORAG_KEY,)).fetchone()[0]
        try:
            point_count = eng.vector_store.count_by_upsert_key_sync(HIPPORAG_KEY)
        except Exception:
            point_count = "n/a"
        print(f"[VERIFY] KV po seedzie: {got}; Qdrant points pod kluczem {HIPPORAG_KEY}: {point_count}")

        audit_records.append({
            "timestamp": now_ts,
            "action": "seed",
            "key": HIPPORAG_KEY,
            "predicate": HIPPORAG_PREDICATE,
            "value": HIPPORAG_VALUE,
            "kv_count": got,
            "qdrant_points": point_count,
        })

    # Zapis raportu do ~/.hermes/atlas/purge_audit.jsonl
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            for rec in audit_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[AUDIT] Raport dopisany do {AUDIT_LOG_PATH} ({len(audit_records)} wpisow)")
    except Exception as exc:
        print(f"[WARN] Nie udalo sie zapisac raportu audit: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
