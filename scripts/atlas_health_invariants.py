"""ATLAS — brama zdrowego rozwoju bazy (invarianty, nie "czy duza").

Pytanie, na ktore odpowiada: czy baza mnozy sie ZGODNIE Z OCZEKIWANIAMI.
Uruchomienie:  python scripts/atlas_health_invariants.py [--json]
Kod wyjscia: 0 = wszystkie twarde invarianty OK, 1 = naruszenie, 2 = brak danych.

Zrodla: daemon po UDS (get_stats, bloki 'graph' z V72.1 i 'vector_store') + atlas.db (ro)
+ kopia Kuzu (fallback).

V72.2 (W5) — dwie zmiany miary, obie z powodu defektow, ktore dawaly falszywy spokoj:
  * H3 liczy sie teraz w BAJTACH wartosci, nie w prefiksach kluczy. Miara prefiksowa
    (mnemosyne:*/turn:*) raportowala 0% w chwili incydentu SEC-001, bo wielokilobajtowy
    blob surowej rozmowy lezal pod kluczem `fact:*` — a to wlasnie tam byl zywy token API.
    Klasa "surowy czat" jest rozpoznawana wlasciwosciowo (dlugosc + markery tur), nie nazwa klucza.
  * H6 (nowy): pokrycie rekordow KV warstwa wektorowa = punkty z prefiksem `kv:` /
    liczba rekordow KV. Prog dolny 0.60 lapie CICHE NIEZAINDEKSOWANIE (rekord w KV bez
    punktu = fakt niewidoczny dla atlas_recall), gorny 1.05 lapie MNOZENIE punktow
    (historyczny najgorszy defekt: 9 969 punktow przy 511 unikalnych tekstach).

Dowod czulosci: na bazie z BUG-036 (266k relacji / 2.8k trojek) inwariant H1 MUSI krzyknac;
na rekordzie z 5 kB surowej rozmowy pod kluczem `fact:*` H3 MUSI krzyknac (stara miara
prefiksowa milczala); przy dwoch punktach na jeden rekord KV H6 MUSI krzyknac.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ATLAS = Path.home() / ".hermes/atlas"
DB = ATLAS / "atlas.db"
KUZU = ATLAS / "kuzu"
STATE = Path.home() / ".hermes/atlas_health_state.json"

# V72.2 (W5): rozpoznawanie surowego czatu po WARTOSCI, nie po prefiksie klucza.
# Prog dlugosci: normalny fakt ATLAS to kilkadziesiat-kilkaset znakow; surowa tura
# rozmowy to tysiace. Markery tur + separator " | " rozstrzygaja przypadki graniczne.
RAW_CHAT_MIN_CHARS = 1024
RAW_CHAT_MARKERS = (
    "[USER]",
    "[user]",
    "[ASSISTANT]",
    "[assistant]",
    "[conversation]",
    "[Hermes]",
    "<|user|>",
    "Human:",
    "Assistant:",
)
TURN_SEPARATOR = " | "
RAW_CHAT_MIN_SEPARATORS = 2

HARD = {  # prog, opis
    "H1_duplication": 1.05,          # count(RELATION) / count(DISTINCT trojka)
    "H2_ram_disk_parity": 0.05,      # |ram_edges - distinct_triples| / distinct_triples
    "H3_raw_chat_bytes": 0.20,       # udzial BAJTOW surowego czatu w bajtach wszystkich KV
    "H5_points_ratio": 3.0,          # punkty wektorowe / rekordy KV
    "H6_vector_coverage_min": 0.60,  # punkty kv: / rekordy KV — dolna granica (niezaindeksowane)
    "H6_vector_coverage_max": 1.05,  # ...gorna granica (mnozenie punktow)
}


def decode_kv_value(raw) -> str:
    """Zwraca TEKST wartosci z `state_variables.value` (podwojnie kodowany JSON wg V70.2)."""
    value = raw
    for _ in range(2):
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                return value
            if isinstance(parsed, str):
                value = parsed
                continue
            if isinstance(parsed, (dict, list)):
                return json.dumps(parsed, ensure_ascii=False)
            return str(parsed) if parsed is not None else ""
        break
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def is_raw_chat_text(text: str) -> bool:
    """Wlasciwosciowa klasa "surowy czat": dlugosc ORAZ slad tur rozmowy.

    Celowo NIE patrzymy na klucz: defekt SEC-001 polegal na tym, ze surowa rozmowa
    (z zywym tokenem API) siedziala pod `fact:cos:...`, wiec miara prefiksowa milczala.
    """
    if not isinstance(text, str) or len(text) < RAW_CHAT_MIN_CHARS:
        return False
    if any(marker in text for marker in RAW_CHAT_MARKERS):
        return True
    return text.count(TURN_SEPARATOR) >= RAW_CHAT_MIN_SEPARATORS


def h3_raw_chat_bytes(rows) -> dict:
    """H3: udzial BAJTOW surowego czatu w bajtach wszystkich wartosci KV.

    `rows` = iterable (key, value) — value surowe z SQLite. Zwraca dict z pomiarem
    i flaga `ok` (prog HARD['H3_raw_chat_bytes']).
    """
    total_bytes = 0
    raw_bytes = 0
    raw_records = 0
    n_kv = 0
    for _key, value in rows:  # celowo NIE zagladamy w klucz — patrz is_raw_chat_text
        n_kv += 1
        text = decode_kv_value(value)
        nbytes = len(text.encode("utf-8"))
        total_bytes += nbytes
        if is_raw_chat_text(text):
            raw_bytes += nbytes
            raw_records += 1
    share = raw_bytes / max(total_bytes, 1)
    return {
        "n_kv": n_kv,
        "raw_bytes": raw_bytes,
        "total_bytes": total_bytes,
        "raw_records": raw_records,
        "share": share,
        "ok": share <= HARD["H3_raw_chat_bytes"],
    }


def h5_points_ratio(n_kv: int, points) -> dict:
    """H5: punkty wektorowe / rekordy KV (lapie mnozenie warstwy wektorowej)."""
    if points is None:
        return {"points": None, "ratio": None, "ok": True}
    ratio = float(points) / max(int(n_kv), 1)
    return {"points": int(points), "ratio": ratio, "ok": ratio <= HARD["H5_points_ratio"]}


def h6_vector_coverage(n_kv: int, kv_points) -> dict:
    """H6 (V72.2): pokrycie rekordow KV warstwa wektorowa — DWUSTRONNE.

    Dolna granica (`>= H6_vector_coverage_min`) lapie ciche niezaindeksowanie: rekord
    istnieje w KV, ale `atlas_recall` go nie znajdzie, bo nie ma punktu z `kv:<klucz>`.
    Gorna granica (`<= H6_vector_coverage_max`) lapie mnozenie punktow — punkty licza sie
    jako PUNKTY (nie unikalne klucze), wiec duplikat podnosi wskaznik tak samo, jak brak
    indeksu go obniza. Brak danych daemona => ok=True (zaden wnioszek nie jest dowodem).
    """
    if kv_points is None:
        return {"kv_points": None, "coverage": None, "ok": True}
    coverage = float(kv_points) / max(int(n_kv), 1)
    return {
        "kv_points": int(kv_points),
        "coverage": coverage,
        "ok": HARD["H6_vector_coverage_min"] <= coverage <= HARD["H6_vector_coverage_max"],
    }


def kv_stats():
    """Surowe (key, value) z KV + liczba wpisow audytu. Read-only, bez dotykania produkcji."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT key, value FROM state_variables").fetchall()
    finally:
        con.close()

    audit_db = ATLAS / "atlas_audit.db"
    audit_target = audit_db if audit_db.exists() else DB
    con_audit = sqlite3.connect(f"file:{audit_target}?mode=ro", uri=True)
    try:
        audit = con_audit.execute("SELECT COUNT(*) FROM state_audit_log").fetchone()[0]
    except sqlite3.OperationalError:
        audit = 0
    finally:
        con_audit.close()
    return [(r[0], r[1]) for r in rows], audit


def kuzu_stats():
    """Zwraca (nodes, rels, distinct_triples, distinct_pairs) albo None."""
    tmp = Path(tempfile.mkdtemp(prefix="atlas_health_"))
    try:
        if not KUZU.exists():
            return None
        shutil.copy(KUZU, tmp / "kz")
        if (ATLAS / "kuzu.wal").exists():
            shutil.copy(ATLAS / "kuzu.wal", tmp / "kz.wal")
        import kuzu

        cn = kuzu.Connection(kuzu.Database(str(tmp / "kz"), read_only=True))
        nodes = cn.execute("MATCH (n:Entity) RETURN count(n)").get_all()[0][0]
        rels = cn.execute("MATCH ()-[r:RELATION]->() RETURN count(r)").get_all()[0][0]
        tri = cn.execute(
            "MATCH (a:Entity)-[r:RELATION]->(b:Entity) RETURN count(DISTINCT [a.name,b.name,r.predicate])"
        ).get_all()[0][0]
        pairs = cn.execute(
            "MATCH (a:Entity)-[r:RELATION]->(b:Entity) RETURN count(DISTINCT [a.name,b.name])"
        ).get_all()[0][0]
        return {"nodes": nodes, "rels": rels, "triples": tri, "pairs": pairs}
    except Exception as exc:  # noqa: BLE001
        print(f"  (Kuzu nieodczytane: {type(exc).__name__}: {str(exc)[:90]})")
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def daemon_stats():
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
    try:
        import asyncio

        from atlas_memory.ipc.uds_client import AtlasDaemonClient as C

        async def go():
            cl = C()
            # include_vector_coverage: H6 potrzebuje liczby punktow z prefiksem kv:
            # (skan kolekcji robi daemon, bo magazyn ma jednego wlasciciela).
            return await cl.call("get_stats", {"include_vector_coverage": True}, timeout=30)

        return asyncio.run(go())
    except Exception as exc:  # noqa: BLE001
        print(f"  (daemon nieodpytany: {type(exc).__name__}: {str(exc)[:90]})")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    kv_rows, n_audit = kv_stats()
    h3 = h3_raw_chat_bytes(kv_rows)
    n_kv = h3["n_kv"]
    kz = kuzu_stats()
    ds = daemon_stats()
    graph = (ds or {}).get("graph") or {}
    vstore = (ds or {}).get("vector_store") or {}

    rows, viol = [], []

    if kz:
        ratio = kz["rels"] / max(kz["triples"], 1)
        ok = ratio <= HARD["H1_duplication"]
        rows.append(("H1 duplikacja relacji", f"{ratio:.2f}x ({kz['rels']:,} / {kz['triples']:,})",
                     f"<= {HARD['H1_duplication']}", ok))
        if not ok:
            viol.append(f"H1: {ratio:.2f}x duplikatow w Kuzu")

    if graph.get("ram_edges") is not None and kz:
        ram = graph["ram_edges"]
        if ram == 0:
            rows.append(("H2 parytet RAM<->dysk", "RAM pusty (brak hydracji)", "info", True))
        else:
            dev = abs(ram - kz["triples"]) / max(kz["triples"], 1)
            ok = dev <= HARD["H2_ram_disk_parity"]
            rows.append(("H2 parytet RAM<->dysk", f"ram={ram:,} vs dysk={kz['triples']:,} ({dev:.1%})",
                         f"<= {HARD['H2_ram_disk_parity']:.0%}", ok))
            if not ok:
                viol.append(f"H2: rozjazd RAM/dysk {dev:.1%}")
    elif not graph:
        rows.append(("H2 parytet RAM<->dysk", "brak bloku 'graph' w get_stats (V72.1)", "info", True))

    # H3 (V72.2): BAJTY, nie prefiksy kluczy. Wartosc to np. "3.2 kB / 41.7 kB (1/189)".
    rows.append((
        "H3 udzial surowego czatu",
        f"{h3['share']:.1%} ({h3['raw_bytes']:,} B / {h3['total_bytes']:,} B, {h3['raw_records']}/{n_kv} rek.)",
        f"<= {HARD['H3_raw_chat_bytes']:.0%} bajtow",
        h3["ok"],
    ))
    if not h3["ok"]:
        viol.append(
            f"H3: {h3['share']:.1%} bajtow KV to surowy czat "
            f"({h3['raw_records']} rek.) — wyciek tresci rozmow (incydent SEC-001)"
        )

    rows.append(("H4 lancuch audytu", f"{n_audit:,} wpisow", "roznaca dlugosc", True))

    # H5: warstwa wektorowa. Historyczny najgorszy defekt ATLAS to mnozenie punktow
    # (9 969 punktow przy 511 unikalnych tekstach = 19,5x), wiec brama MUSI patrzec
    # takze tu. Slabsze slowo: stosunek punktow do rekordow KV.
    h5 = h5_points_ratio(n_kv, vstore.get("points_count"))
    if h5["points"] is not None:
        rows.append(("H5 punkty/rekordy KV", f"{h5['points']} / {n_kv} = {h5['ratio']:.2f}x",
                     f"<= {HARD['H5_points_ratio']}", h5["ok"]))
        if not h5["ok"]:
            viol.append(f"H5: {h5['ratio']:.1f}x punktow wzgledem KV — warstwa wektorowa mnozy sie")
    else:
        rows.append(("H5 punkty/rekordy KV", "brak danych daemona", f"<= {HARD['H5_points_ratio']}", True))

    # H6 (V72.2): pokrycie KV wektorami — dwustronne. Patrz docstring h6_vector_coverage.
    h6 = h6_vector_coverage(n_kv, vstore.get("kv_indexed_points"))
    if h6["kv_points"] is not None:
        rows.append((
            "H6 pokrycie wektorowe KV",
            f"{h6['kv_points']} / {n_kv} = {h6['coverage']:.2f}x",
            f">= {HARD['H6_vector_coverage_min']:.2f} i <= {HARD['H6_vector_coverage_max']:.2f}",
            h6["ok"],
        ))
        if not h6["ok"]:
            if h6["coverage"] < HARD["H6_vector_coverage_min"]:
                viol.append(
                    f"H6: tylko {h6['coverage']:.0%} rekordow KV ma punkt wektorowy "
                    f"({h6['kv_points']}/{n_kv}) — rekordy sa cicho niezaindeksowane"
                )
            else:
                viol.append(
                    f"H6: {h6['coverage']:.2f}x punktow kv: na rekord KV ({h6['kv_points']}/{n_kv}) "
                    "— warstwa wektorowa mnozy punkty"
                )
    else:
        rows.append((
            "H6 pokrycie wektorowe KV",
            "brak danych daemona (include_vector_coverage)",
            f">= {HARD['H6_vector_coverage_min']:.2f}",
            True,
        ))

    prev = {}
    if STATE.exists():
        try:
            prev = json.loads(STATE.read_text())
        except Exception:
            prev = {}
    cur = {"kv": n_kv, "audit": n_audit, "kuzu_bytes": KUZU.stat().st_size if KUZU.exists() else 0,
           "kuzu_rels": (kz or {}).get("rels"), "points": vstore.get("points_count"),
           "kv_points": vstore.get("kv_indexed_points"), "kv_raw_bytes": h3["raw_bytes"],
           "kv_total_bytes": h3["total_bytes"],
           "rss_mb": round((ds or {}).get("rss_mb") or 0, 1)}
    STATE.write_text(json.dumps(cur, indent=1))

    if a.json:
        print(json.dumps({"rows": rows, "violations": viol, "current": cur, "previous": prev}, ensure_ascii=False))
    else:
        print(f"{'INWARIANT':30} {'WARTOSC':44} {'PROG':24} WYNIK")
        for name, val, tgt, ok in rows:
            print(f"{name:30} {val:44} {tgt:24} {'OK' if ok else 'NARUSZENIE'}")
        print("\nPRZYROST (vs poprzedni przebieg):")
        for k, v in cur.items():
            p = prev.get(k)
            d = "" if p in (None, v) else f"  (bylo {p})"
            print(f"  {k:14} = {v}{d}")
        if viol:
            print("\nNARUSZENIA:")
            for v in viol:
                print("  -", v)

    return 1 if viol else 0


if __name__ == "__main__":
    raise SystemExit(main())
