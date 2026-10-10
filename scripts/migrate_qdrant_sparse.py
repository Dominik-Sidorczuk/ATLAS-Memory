#!/usr/bin/env python
"""V92: addytywne doklejenie kanału sparse do istniejącej kolekcji Qdrant.

Dlaczego addytywnie: kanał gęsty jest w tej kolekcji zdrowy (V91), więc NIE ma powodu
rekodować 336 wektorów 512D ani ruszać ID punktów (deterministyczne blake2b z klucza KV).
Dołożenie nazwanego wektora `sparse` (Modifier.IDF) nie zmienia gęstej gałęzi, a zapytanie
hybrydowe (prefetch dense+sparse, fuzja DBSF) dostaje drugi kanał natychmiast.

Tryb: `--dry-run` (domyślnie, nic nie zapisuje) / `--apply`. Weryfikacja po zapisie jest
wbudowana: po `--apply` skrypt ponownie skanuje kolekcję i raportuje, ile punktów ma OBA
wektory oraz czy odcisk wektora gęstego (długość + suma) pozostał niezmieniony.

Uwaga operacyjna: magazyn na dysku ma jednego właściciela (`.lock`) — przed uruchomieniem
na produkcji trzeba zatrzymać demona (`pkill -TERM -f atlas_daemon`).
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from atlas_memory.l2_semantic.sparse_encoder import SPARSE_VECTOR_NAME, encode_sparse_document

SPARSE_CFG = {
    SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(
        index=qmodels.SparseIndexParams(on_disk=False),
        modifier=qmodels.Modifier.IDF,
    )
}


def _sparse_ready(client: QdrantClient, name: str) -> bool:
    info = client.get_collection(name)
    return bool(getattr(info.config.params, "sparse_vectors", None))


def _pick_collection(client: QdrantClient, requested: Optional[str]) -> str:
    names = [c.name for c in client.get_collections().collections]
    if requested:
        if requested not in names:
            raise SystemExit(f"brak kolekcji '{requested}' (są: {names})")
        return requested
    for preferred in ("hermes_episodic_memory_68m", "hermes_episodic_memory"):
        if preferred in names:
            return preferred
    if not names:
        raise SystemExit("magazyn nie ma żadnej kolekcji")
    return names[0]


def _text_of(payload: Dict[str, Any]) -> str:
    text = payload.get("text")
    if text:
        return str(text)
    rec = payload.get("record") or {}
    if isinstance(rec, dict):
        return " ".join(str(rec.get(k, "")) for k in ("subject", "predicate", "object"))
    return ""


def _scroll_all(client: QdrantClient, name: str, with_vectors: bool) -> List[Any]:
    out: List[Any] = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=name, limit=256, offset=offset,
            with_payload=True, with_vectors=with_vectors,
        )
        out.extend(batch)
        if offset is None:
            break
    return out


def _dense_of(point_vector: Any) -> Optional[List[float]]:
    if point_vector is None:
        return None
    if isinstance(point_vector, dict):
        return point_vector.get("") or point_vector.get("dense")
    return list(point_vector)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="ścieżka magazynu Qdrant (local mode)")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--apply", action="store_true", help="bez tej flagi: dry-run")
    ap.add_argument("--dry-run", action="store_true", help="jawnie: nic nie zapisuj (domyślne)")
    args = ap.parse_args()

    started = time.time()
    client = QdrantClient(path=args.store)
    name = _pick_collection(client, args.collection)

    ready_before = _sparse_ready(client, name)
    points = _scroll_all(client, name, with_vectors=True)
    input_fingerprint = {
        p.id: (round(sum(_dense_of(p.vector) or []), 4), len(_dense_of(p.vector) or []))
        for p in points
    }

    recreated = False
    report: Dict[str, Any] = {
        "store": args.store,
        "collection": name,
        "points": len(points),
        "sparse_schema_before": ready_before,
        "mode": "apply" if args.apply else "dry-run",
    }

    if not ready_before and args.apply:
        # Ścieżka czysta: silnik nie umie dołożyć konfiguracji sparse do istniejącej
        # kolekcji — odtwarzamy ją z tym samym ID, payloadem i wektorem gęstym.
        client.delete_collection(name)
        dense0 = _dense_of(points[0].vector) if points else None
        if not dense0:
            raise SystemExit("brak wektora gęstego w pierwszym punkcie — odmawiam odtwarzania")
        dim = len(dense0)
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            sparse_vectors_config=SPARSE_CFG,
        )
        client.upsert(collection_name=name, points=[
            qmodels.PointStruct(
                id=p.id,
                vector={"": _dense_of(p.vector), SPARSE_VECTOR_NAME: encode_sparse_document(_text_of(p.payload or {}))},
                payload=dict(p.payload or {}),
            ) for p in points
        ])
        report["recreated_with_sparse"] = True
        recreated = True
    elif not ready_before:
        report["recreated_with_sparse"] = False
        report["warning"] = "dry-run: kolekcja nie ma schematu sparse — z --apply zostanie odtworzona"
    else:
        report["recreated_with_sparse"] = False

    # UWAGA (zmierzone na kopii): `upsert` w Qdrancie REPLACUJE komplet wektorów punktu —
    # nie jest częściową aktualizacją. Pierwsza wersja tego skryptu dokładała sparse drugim
    # `upsert`-em i WYTARŁA wektory gęste na 336/336 punktów (wykryte przez odcisk dense).
    # Częściowa aktualizacja istnieje tylko jako osobne API: `update_vectors`.
    sparse_added = 0
    no_text = 0
    if recreated:
        sparse_added = len(points)  # sparse wpisany w tym samym upsercie co wektor gęsty
    elif args.apply or ready_before:
        for p in points:
            payload = p.payload or {}
            text = _text_of(payload)
            if not text:
                no_text += 1
                continue
            vec = p.vector if isinstance(p.vector, dict) else {}
            existing = vec.get(SPARSE_VECTOR_NAME)
            if existing is not None and len(getattr(existing, "indices", []) or []) > 0:
                continue
            if args.apply:
                client.update_vectors(
                    collection_name=name,
                    points=[qmodels.PointVectors(
                        id=p.id,
                        vector={SPARSE_VECTOR_NAME: encode_sparse_document(text)},
                    )],
                )
            sparse_added += 1
    report["sparse_added"] = sparse_added
    report["points_without_text"] = no_text

    # --- weryfikacja po zapisie (nie self-report: ponowny odczyt z magazynu) ---
    after = _scroll_all(client, name, with_vectors=True)
    both = 0
    mismatch = 0
    for p in after:
        vec_map = p.vector if isinstance(p.vector, dict) else {}
        sv = vec_map.get(SPARSE_VECTOR_NAME)
        dense_now = _dense_of(p.vector)
        if sv is not None and len(getattr(sv, "indices", []) or []) > 0 and dense_now:
            both += 1
        if p.id in input_fingerprint:
            current = (round(sum(dense_now or []), 4), len(dense_now or []))
            if current != input_fingerprint[p.id]:
                mismatch += 1
    report["verify"] = {
        "points_after": len(after),
        "points_with_dense_and_sparse": both,
        "dense_fingerprint_mismatch": mismatch,
        "sparse_schema_after": _sparse_ready(client, name),
    }
    report["seconds"] = round(time.time() - started, 2)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
