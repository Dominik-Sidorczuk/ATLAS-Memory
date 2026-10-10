#!/usr/bin/env python3
"""ATLAS Memory Cleanup — usuwa śmieci testowe z produkcyjnego KV przez RPC daemona.

Dlaczego przez RPC (nie raw SQL): (1) delete guard ADR-005 chroni prefiksy
hermes:/fact:/prompt-/mnemosyne: — raw SQL by go ominął; (2) każdy delete jest
tombstone'em rejestrowanym w SHA-256 audit chain — weryfikowalny po fakcie
(atlas_verify_chain). Skrypt niczego nie zgaduje: klasyfikacja = twarde wzorce
kluczy z audytu (load:/test:/probe:usability/secrets/artefakty bin/unicode/...).

Uruchomienie:
    python scripts/atlas_memory_cleanup.py --dry-run     # pokaż plan (domyślne)
    python scripts/atlas_memory_cleanup.py --apply       # wykonaj deletes

Kategorie usuwane (tylko testowe artefakty, NIGDY fakty użytkownika):
  load:*            — load-test "v0" × 100
  test:*, test.rpc.*, T*/usability/*, v42:*, v57:*, test_e2e_v41
  probe:usability:* — writebacki z baterii (db_port/tolerance itd.)
  session_*/ses_*/secret:*  — "TAJNE_Z_SESJI" izolacja testów
  budget:*, cache:*, fresh:*, chain:sep04:*, veracity:*, unicode:*, tamper:*
  graph:A/B/C, conflict:key, normal_*, key\x00, key\n, xxxx..., :observation
  config:worker_pool:size, service:db, task:pipeline:step_v47, canary_token_v32
  user:alice_pref*, user:rule:persistent*, user:preference:test*
"""
from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas_memory.ipc.uds_client import send_uds_request_sync  # noqa: E402

# Prefiksy/klucze, których NIGDY nie dotykamy (fakty użytkownika + systemowe).
KEEP_PREFIXES = (
    "mnemosyne:", "fact:", "hermes:", "prompt-", "turn:", "rule:", "atlas:",
    "user:active_context", "user:preference:editor", "agent:preference:language",
    "GHK-Cu:", "OptimizationStrategy:", "SystemArchitecture:",
    "hardware:", "system:boot", "service:crawl4ai", "sop::", "incident:",
    "config:worker_pool", "config:llm", "service:llm",
)

# Wzorce USUWANE (fnmatch). Świadomie wąskie — tylko udokumentowane artefakty testów.
DELETE_PATTERNS = (
    "load:*", "test:*", "test.rpc.write", "test_e2e_v41",
    "T0*/usability/*", "T1*/usability/*", "T06/*", "T12/*", "T13/*", "T14/*",
    "probe:usability:*", "session_*", "ses_*", "secret:*",
    "budget:*", "cache:*", "fresh:*", "chain:sep04:*", "v42:*", "v57:*",
    "veracity:*", "unicode:*", "tamper:*", "graph:A", "graph:B", "graph:C",
    "decay:*", "conflict:key", "normal_key", "normal_key2", "normal_flow:*",
    "key\x00*", "key\n*", "xxxxxxxx*", ":observation", "*:observation",
    "config:worker_pool:size", "service:db", "task:pipeline:step_v47:*",
    "project:atlas:canary_token_v32", "user:alice_pref*",
    "user:rule:persistent*", "user:preference:test*", "zero.trust.check",
    "result:pytest-v45", "usability:*", "agent:session:test_usability_v41",
)


def classify(key: str) -> str:
    if key.startswith(KEEP_PREFIXES):
        return "keep"
    for pat in DELETE_PATTERNS:
        if fnmatch.fnmatch(key, pat):
            return "delete"
    return "UNMATCHED"  # nie ruszamy — wymaga ręcznej decyzji


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="wykonaj deletes (domyślnie dry-run)")
    args = ap.parse_args()

    import sqlite3

    src = Path.home() / ".hermes" / "atlas" / "atlas.db"
    con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    keys = [r[0] for r in con.execute("SELECT key FROM state_variables")]
    con.close()

    plan: dict[str, list[str]] = {"delete": [], "keep": [], "UNMATCHED": []}
    for k in keys:
        plan[classify(k)].append(k)

    print(f"Razem: {len(keys)} | usunąć: {len(plan['delete'])} | "
          f"zostać: {len(plan['keep'])} | niezaklasyfikowane: {len(plan['UNMATCHED'])}")

    if plan["UNMATCHED"]:
        print("\nNIEZAKLASYFIKOWANE (nie ruszam, decyzja ręczna):")
        for k in sorted(plan["UNMATCHED"])[:25]:
            print(f"  ? {k[:80]}")

    if plan["keep"]:
        print(f"\nZOSTAJE ({len(plan['keep'])}): prefiksy chronione/fakty")
    print("\nUSUWANE przykłady:")
    for k in sorted(plan["delete"])[:12]:
        print(f"  - {k[:80]}")

    if not args.apply:
        print("\n[dry-run] Nic nie usunięto. Uruchom z --apply, aby wykonać.")
        return 0

    ok = rej = err = 0
    for k in sorted(plan["delete"]):
        resp = send_uds_request_sync(method="delete", params={"key": k, "reason": "cleanup_test_artifacts_2026-09-09"})
        if resp is None:
            err += 1
            print(f"  ! brak odpowiedzi: {k[:60]}")
        elif resp.get("status") in ("ok",) or resp.get("deleted"):
            ok += 1
        else:
            rej += 1
            print(f"  ! odrzucony: {k[:60]} -> {resp.get('status')} {resp.get('reason')}")
    print(f"\nWynik: usunięto={ok} odrzucono={rej} brak_odp={err}")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
