"""FIX-1 (F-001): Write-Path Dual-Node Wiring, Idempotent Replay (`replay_at`), and Atomic Rollback Tests.

Scenarios:
  a) Write new record -> 1 Passage + N CONTAINS + M SYNONYM
  b) Replay of the same record with `replay_at=datetime(2026, 8, 1)` -> no-op, audit log "replay_no_op"
  c) Write with error in `add_synonym` -> atomic rollback, 0 Passage / CONTAINS / SYNONYM
  d) Write with `replay_at` -> Cypher query `MATCH (p:Passage) WHERE p.ingest_at = replay_at` returns record.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from atlas_memory.cognitive.conflict_arbiter import ConflictArbiter
from atlas_memory.cognitive.write_path import EpistemicWritePath
from atlas_memory.l1_working.hot_buffer import HotBufferPolicy
from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore, KuzuWriteError
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore


@pytest.mark.asyncio
async def test_write_path_dual_node_scenarios_a_b_c_d(tmp_path: Path, monkeypatch) -> None:
    kuzu_dir = str(tmp_path / "kuzu_wp")
    kv_path = str(tmp_path / "kv_wp.sqlite")
    store = KuzuGraphStore(db_path=kuzu_dir)
    kv = VerifiedKVStore(db_path=kv_path)
    try:
        # Pre-seed a morphologically close entity so SYNONYM edge >= 0.80 is formed
        store.create_entity("kalibracja_modelu", "entity")
        wp = EpistemicWritePath(
            kv_store=kv,
            graph_store=store,
            hot_buffer=HotBufferPolicy(maxlen=20),
            conflict_arbiter=ConflictArbiter(),
        )

        # Scenario (a) & (d): Write new record with replay_at=datetime(2026, 8, 1)
        replay_dt = datetime(2026, 8, 1)
        replay_iso = replay_dt.isoformat()
        record = {
            "source_key": "fact:kalibracja_dual_node",
            "text": "Kalibracja modelu wymaga walidacji progów PAVA na zbiorze testowym.",
            "entities": ["kalibracja_modelu_pava", "walidacja_pava"],
            "synonyms": [("kalibracja_modelu_pava", "kalibracja_modelu", 0.91)],
        }
        res_a = await wp.write_async(record, replay_at=replay_dt, replay=False)
        assert res_a["status"] == "ok"

        p_cnt = int(store.conn.execute("MATCH (p:Passage) RETURN count(p)").get_next()[0])
        c_cnt = int(store.conn.execute("MATCH ()-[c:CONTAINS]->() RETURN count(c)").get_next()[0])
        s_cnt = int(store.conn.execute("MATCH ()-[s:SYNONYM]->() RETURN count(s)").get_next()[0])
        assert p_cnt == 1, "Scenario (a): Expected 1 Passage node"
        assert c_cnt >= 2, "Scenario (a): Expected N >= 2 CONTAINS edges"
        assert s_cnt >= 2, "Scenario (a): Expected M >= 2 bidirectional SYNONYM edges"

        # Scenario (d): Query `MATCH (p:Passage) WHERE p.ingest_at = replay_at` returns record
        res_d = store.conn.execute(
            "MATCH (p:Passage) WHERE p.ingest_at = $replay_at RETURN p.passage_id, p.source_key",
            {"replay_at": replay_iso},
        )
        assert res_d.has_next(), "Scenario (d): Expected Passage with ingest_at == replay_at"
        row_d = res_d.get_next()
        assert row_d[0] == "fact:kalibracja_dual_node"
        assert row_d[1] == "fact:kalibracja_dual_node"

        # Scenario (b): Replay same record with replay_at=datetime(2026, 8, 1) -> no-op, audit log "replay_no_op"
        res_b = await wp.write_async(record, replay_at=replay_dt, replay=True)
        assert res_b["status"] == "ok"
        assert res_b.get("no_op") is True
        assert res_b.get("no_op_reason") == "replay_no_op"
        assert any(e.get("reason") == "replay_no_op" for e in wp.audit_log)

        p_cnt_b = int(store.conn.execute("MATCH (p:Passage) RETURN count(p)").get_next()[0])
        c_cnt_b = int(store.conn.execute("MATCH ()-[c:CONTAINS]->() RETURN count(c)").get_next()[0])
        s_cnt_b = int(store.conn.execute("MATCH ()-[s:SYNONYM]->() RETURN count(s)").get_next()[0])
        assert p_cnt_b == p_cnt, "Scenario (b): Replay must not duplicate Passage"
        assert c_cnt_b == c_cnt, "Scenario (b): Replay must not duplicate CONTAINS"
        assert s_cnt_b == s_cnt, "Scenario (b): Replay must not duplicate SYNONYM"

        # Scenario (c): Write with error in `add_synonym` -> atomic rollback, no Passage/CONTAINS/SYNONYM added
        def _boom_synonym(*args, **kwargs):
            raise KuzuWriteError("Simulated failure in add_synonym")

        monkeypatch.setattr(store, "add_synonym", _boom_synonym)
        bad_record = {
            "source_key": "fact:rollback_target",
            "text": "Eksperymentalny zapis z wymuszonym błędem w add_synonym.",
            "entities": ["encja_rollback_1", "encja_rollback_2"],
            "synonyms": [("encja_rollback_1", "encja_rollback_2", 0.95)],
        }
        with pytest.raises(KuzuWriteError):
            await wp.write_async(bad_record, replay_at=datetime(2026, 8, 2))

        p_cnt_c = int(
            store.conn.execute(
                "MATCH (p:Passage {passage_id: 'fact:rollback_target'}) RETURN count(p)"
            ).get_next()[0]
        )
        c_cnt_c = int(
            store.conn.execute(
                "MATCH (p:Passage {passage_id: 'fact:rollback_target'})-[c:CONTAINS]->() RETURN count(c)"
            ).get_next()[0]
        )
        assert p_cnt_c == 0, "Scenario (c): Passage must be rolled back on add_synonym failure"
        assert c_cnt_c == 0, "Scenario (c): CONTAINS must be rolled back on add_synonym failure"
    finally:
        store.close()
        await kv.close()
