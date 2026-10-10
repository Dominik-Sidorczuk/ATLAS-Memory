"""
LIVE HERMES DAEMON SOAK & IPC PROTOCOL VALIDATOR
Wektor 3: Rzeczywista integracja w runtime na żywym gnieździe ~/.hermes/atlas.sock.
Realizuje 5-etapowy scenariusz testowy:
1. Phatic Skip (Fast-exit gate dla czystej konwersacji fatycznej)
2. Assert (Zapis nowego ugruntowanego faktu ze spójnością multi-store)
3. Contradict (Ochrona przed inwersją pewności i zaszumioną sprzecznością)
4. What-If (Merytoryczny prefetch z rankingiem veracity i PPR)
5. Active Sensing (Telemetria sensoryczna i weryfikacja stabilności)
"""

import asyncio
import sys
import time
from pathlib import Path

from atlas_memory.ipc.uds_client import AtlasDaemonClient


async def run_live_soak():
    sock_path = Path.home() / ".hermes" / "atlas.sock"
    if not sock_path.exists():
        print(f"BŁĄD: Gniazdo UDS {sock_path} nie istnieje!")
        sys.exit(1)

    client = AtlasDaemonClient(str(sock_path), timeout=5.0)

    # 0. Ping & Status
    ping_res = await client.ping()
    print(f"[LIVE:STATUS] Ping: {ping_res}")
    assert ping_res is True, "Demon nie odpowiada na ping!"

    # =========================================================================
    # ETAP 1: PHATIC SKIP
    # =========================================================================
    print("\n--- ETAP 1: PHATIC SKIP ---")
    phatic_res = await client.call("prefetch", {"query": "Cześć Hermes, jak się dzisiaj miewasz?"})
    print(f"[ETAP 1] Query fatyczne -> skipped={phatic_res.get('skipped')} | records={len(phatic_res.get('records', []))}")
    assert phatic_res.get("skipped") is True, "Zapytanie fatyczne musi zostać pominięte przez prefetch gate!"

    # =========================================================================
    # ETAP 2: ASSERT
    # =========================================================================
    run_id = int(time.time()) % 10000
    assert_key = f"service:metrics_collector_{run_id}:listen_port"
    assert_val = "9090"
    set_res = await client.call("set", {
        "key": assert_key,
        "value": assert_val,
        "confidence": 0.98,
        "source": "user_explicit",
    })
    print(f"[ETAP 2] Assert -> {set_res}")
    assert set_res.get("status") in ("persisted", "ok", "success"), f"Zapis faktu nie powiódł się: {set_res}"

    # =========================================================================
    # ETAP 3: CONTRADICT
    # =========================================================================
    print("\n--- ETAP 3: CONTRADICT (Próba sprzecznego zapisu o niższej pewności) ---")
    contra_res = await client.call("set", {
        "key": assert_key,
        "value": "9091",
        "confidence": 0.70,
        "source": "tool_output",
    })
    print(f"[ETAP 3] Contradict -> {contra_res}")
    # Powinien zostać zablokowany lub wywołać konflikt epistemiczny
    status_contra = contra_res.get("status")
    assert status_contra in ("rejected", "conflict", "warn_conflict", "skipped") or contra_res.get("warning") is not None, (
        f"Zaszumiony zapis nie został zablokowany! {contra_res}"
    )

    # =========================================================================
    # ETAP 4: WHAT-IF (PREFETCH MERYTORYCZNY)
    # =========================================================================
    print("\n--- ETAP 4: WHAT-IF (Recall wiedzy operacyjnej) ---")
    query_str = f"Na jakim porcie nasłuchuje metrics_collector_{run_id}?"
    recall_res = await client.call("prefetch", {"query": query_str})
    recs = recall_res.get("records", [])
    print(f"[ETAP 4] Recall -> skipped={recall_res.get('skipped')} | count={len(recs)}")
    found_target = False
    for r in recs:
        subj = r.get("subject", "")
        obj = r.get("object", "") or r.get("value", "")
        print(f"  • Winner: [{subj}] -> {obj} (conf={r.get('confidence')}, score={r.get('score', r.get('graph_score'))})")
        if f"metrics_collector_{run_id}" in subj and "9090" in str(obj):
            found_target = True

    assert found_target is True, "Recall nie zwrócił ugruntowanego faktu portu 9090!"

    # =========================================================================
    # ETAP 5: ACTIVE SENSING
    # =========================================================================
    print("\n--- ETAP 5: ACTIVE SENSING & TELEMETRIA ---")
    obs_res = await client.call("commit_observation", {
        "subject": f"service:metrics_collector_{run_id}",
        "predicate": "telemetry_health",
        "object": "healthy_200_ok",
        "confidence": 1.0,
        "task_success": True,
    })
    print(f"[ETAP 5] Commit Observation -> {obs_res}")
    assert obs_res.get("status") in ("committed", "ok", "success"), f"Błąd obserwacji: {obs_res}"

    print("\n=======================================================")
    print("  LIVE HERMES DAEMON SOAK: ALL 5 PHASES PASSED (100% OK)")
    print("=======================================================\n")


if __name__ == "__main__":
    asyncio.run(run_live_soak())
