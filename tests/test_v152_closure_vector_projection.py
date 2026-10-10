"""V152-CLOSURE / ZADANIE 1: deterministyczna projekcja wektorowa kanonika hipporag2.

BUG-603: `module:retrieval:hipporag2` istnieje w KV ( reclaimed ) i Kùzu, ale ma
0 punktów w Qdrant — atlas_recall "hipporag2" zwraca pustkę. Inwariant triady:
fakt KV ma sprzężony rzut w KAŻDYM widoku; brak rzutu = dyskwalifikacja.
"""
from __future__ import annotations

import json
import socket
import struct
from pathlib import Path

import pytest

SOCK_PATH = Path.home() / ".hermes" / "atlas" / "sock" / "atlas.sock"
UDS_PATH = Path.home() / ".hermes" / "atlas.sock"
CANON_KEY = "module:retrieval:hipporag2"
CANON_VAL = (
    "HippoRAG 2 to asocjacyjny silnik grafowy oparty na Personalized PageRank (igraph) "
    "na grafie Dual-Node (Passage/Entity) z adaptacyjnym budzetem czasowym oraz "
    "warunkowanym tunelowaniem aksjomatow."
)


def _call(method: str, params: dict | None = None) -> dict:
    """Minimalny klient JSON-RPC 2.0 over UDS (length-prefix 4B BE)."""
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


@pytest.mark.order(1)
class TestVctorProjectionCanon:
    """Inwariant VP-1: kanonik KV musi mieć sprzężony punkt wektorowy w Qdrant."""

    def test_daemon_alive(self):
        assert UDS_PATH.exists() or SOCK_PATH.exists(), "daemon nie działa — UDS nie istnieje"

    def test_hipporag2_kv_fact_exists(self):
        r = _call("get", {"key": CANON_KEY}).get("result") or {}
        found = bool(r.get("found")) or (r.get("state") is not None)
        # fallback: klucz może być z sufiksem predykatu
        if not found:
            r2 = _call("atlas_recall", {"query": "hipporag2 dual-node PPR", "top_k": 5}).get("result") or {}
            for h in r2.get("hits", []) or []:
                if CANON_KEY in str(h.get("key", h.get("subject", ""))):
                    found = True
                    break
        assert found, f"kanonik nie istnieje w KV: {r}"

    def test_hipporag2_vector_hit(self):
        """Asercja z dyrektywy: recall przez wektor/g Gra seamusz score > 0.70."""
        r = _call("atlas_recall", {"query": "hipporag2 dual-node PPR", "top_k": 4})
        res = r.get("result") or {}
        hits = res.get("hits") or res.get("candidates") or []
        assert hits, f"recall zwraca pustkę: {r}"
        canon_hits = [h for h in hits if CANON_KEY in str(h.get("key") or h.get("subject") or "")]
        assert canon_hits, f"kanonik {CANON_KEY} nie znaleziony w trafieniach: {hits}"
        best = max(canon_hits, key=lambda h: h.get("score", h.get("similarity", 0.0)) or 0.0)
        score = best.get("score") or best.get("similarity") or 0.0
        src = str(best.get("source") or best.get("match_type") or "")
        assert score > 0.70, f"najsilniejsze trafienie {score:.2f} poniżej progu 0.70: {best}"
        assert ("vector" in src) or ("hybrid" in src) or (not src), (
            f"źródło {src!r} nie jest vector_dense/hybrid: {best}"
        )
