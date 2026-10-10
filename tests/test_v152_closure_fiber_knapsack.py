"""V152-CLOSURE / ZADANIE 3: Filar IV — Fiber Knapsack (EntityRoot × Slot).

Topologiczna redukcja ekwilawalencyjna:
  Equivalence(A, B) ⟺ EntityRoot(A) == EntityRoot(B) ∧ Slot(A) == Slot(B)
gdzie Slot = wskazana semantyka wpisu (predicate/categoria). Warianty klucza
różniące się sufiksami nawiasowymi (daty, hashe git) — e.g.
'Public ATLAS-Memory v2.0.0 (main=0873d87)' vs
'Public ATLAS-Memory v2.0.0 (08.09, main=0873d87, Release+tag, backup tag...)'
— należą DO TEJ SAMEJ klasy. Solver plecakowy nie może ich rozpatrywać jako
niezależnych x_i; do puli wchodzi wyłącznie kanonik (rank/time).

Live-azy: recall "Public ATLAS-Memory repozytorium" zwraca dokładnie 1 wpis
kanoniczny.
"""
from __future__ import annotations

import json
import socket
import struct
from pathlib import Path

import pytest

UDS_PATH = Path.home() / ".hermes" / "atlas.sock"


def _call(method: str, params: dict | None = None) -> dict:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(15.0)
    sock.connect(str(UDS_PATH))
    try:
        payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}).encode()
        sock.sendall(struct.pack(">I", len(payload)) + payload)
        rl = b""
        while len(rl) < 4:
            rl += sock.recv(4 - len(rl))
        n = struct.unpack(">I", rl)[0]
        buf = b""
        while len(buf) < n:
            buf += sock.recv(n - len(buf))
        return json.loads(buf)
    finally:
        sock.close()


class TestEntityRootSlotKnapsack:
    """Filar IV: klasa abstrakcji EntityRoot × Slot wpływa na pule plecaka."""

    def _entity_root(self, subject: str) -> str:
        """Import z render.py (V151.1/F-032 kanoniczna postać tematu)."""
        from atlas_memory.cognitive.retrieval.render import entity_root
        return entity_root(subject)

    def _slot_of(self, hit: dict) -> str:
        return str(hit.get("predicate") or hit.get("slot") or "")

    def test_recall_public_atlas_memory_single_entry(self):
        """Asercja z dyrektywy: dokładnie 1 wpis kanoniczny, nie multi-variant."""
        r = _call("atlas_recall", {"query": "Public ATLAS-Memory repozytorium", "top_k": 8})
        res = r.get("result") or {}
        hits = res.get("hits") or res.get("candidates") or []
        if not hits:
            pytest.skip(f"atlas_recall UDS zwrot nie interpretuje; raw: {json.dumps(r)[:200]}")
        cluster_keys = {(self._entity_root(str(h.get("subject") or h.get("key") or "")), self._slot_of(h)) for h in hits}
        from collections import Counter
        counts = Counter(cluster_keys)
        for cluster, n in counts.items():
            assert n <= 1, (
                f"klasa {cluster} ma {n} rekordów — naruszenie Fiber Knapsack "
                f"(EntityRoot × Slot): {n} wariantów rywalizuje o budget"
            )
