"""V152 FILAR II — kanoniczny point_id Qdranta, niezmienniczy względem
prefiksów transportowych. Test NIE wymaga żywego magazynu Qdrant — weryfikuje
wyłącznie czyste funkcje + serializację filtra (bez importu qdrant_client)."""
from __future__ import annotations

import os
import sys
import unittest

_REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)


class TestCanonicalPointId(unittest.TestCase):
    """_point_id_for_key MUSI być funkcją normalize_upsert_key (kanon)."""

    def test_same_key_with_different_transport_prefixes_yields_same_point_id(self):
        from atlas_memory.l2_semantic.qdrant_store import _point_id_for_key, normalize_upsert_key
        bare = "hermes:x:state_value"
        canon = normalize_upsert_key(bare)
        with_kv = f"kv:{bare}"
        self.assertEqual(
            _point_id_for_key(bare), _point_id_for_key(canon),
            "FILAR II-2 violated: canon and bare disagree",
        )
        self.assertEqual(
            _point_id_for_key(canon), _point_id_for_key(with_kv),
            "FILAR II-2 violated: kv: prefix diverges from canon",
        )

    def test_distinct_canonical_keys_yield_distinct_point_ids(self):
        from atlas_memory.l2_semantic.qdrant_store import _point_id_for_key
        ids = {
            _point_id_for_key(k)
            for k in (
                "hermes:auth_service:has_port",
                "hermes:auth_service:state_value",
                "hermes:hipporag2:module_is",
            )
        }
        self.assertEqual(len(ids), 3, "Distinct canonical keys must yield distinct point ids")


class TestUpsertKeyFilterSymmetry(unittest.TestCase):
    """_upsert_key_filter MUSI w conditions pokrywać wszystkie transportowe
    warianty (canon, kv:canon, kv_key==canon). Weryfikacja serializowanego
    filtra (nie wymaga klienta Qdrant)."""

    def test_filter_conditions_cover_canon_and_kv_prefix_and_kv_key(self):
        from atlas_memory.l2_semantic.qdrant_store import QdrantVectorStore, normalize_upsert_key

        # Test jest statyczny (bez klienta); jeśli wymaga importu qdrant_client,
        # spróbujemy: import może być lekki.
        try:
            from qdrant_client.http import models as qmodels  # noqa
        except Exception as exc:
            self.skipTest(f"qdrant_client import unavailable: {exc}")
        filt = QdrantVectorStore._upsert_key_filter("auth_service:has_port")
        conds = [
            {"key": c.key, "match": getattr(c.match, "value", None)}
            for c in getattr(filt, "should", []) or []
        ]
        canon = normalize_upsert_key("auth_service:has_port")
        self.assertIn({"key": "upsert_key", "match": canon}, conds)
        self.assertIn({"key": "upsert_key", "match": f"kv:{canon}"}, conds)
        self.assertIn({"key": "kv_key", "match": canon}, conds)


if __name__ == "__main__":
    unittest.main()
