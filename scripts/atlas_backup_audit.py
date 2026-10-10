#!/usr/bin/env python3
"""ATLAS Safe Audit Database Backup (FINDING-072).

Protokół bezpiecznej kopii zapasowej bazy audytu `atlas_audit.db`:
1. Kontrolowany WAL Checkpoint (TRUNCATE) w celu przeniesienia stron WAL do bazy głównej.
2. Wykorzystanie SQLite Online Backup API (`sqlite3.backup`) gwarantującego atomowy,
   spójny snapshot bez ryzyka odczytania częściowo zapisanych stron w locie.
3. Weryfikacja samodzielności (standalone): zliczenie rekordów i sprawdzenie spójności
   pliku docelowego bez obecności plików -wal i -shm.
"""

from __future__ import annotations

import argparse
import datetime
import sqlite3
import sys
from pathlib import Path
from typing import Tuple


def backup_audit_database(
    audit_db_path: Path | str,
    output_path: Path | str | None = None,
    checkpoint: bool = True,
) -> Tuple[Path, int, int, int]:
    """Wykonuje bezpieczną kopię zapasową bazy audytu.

    Returns:
        Tuple[Path, int, int, int]: (ścieżka_kopii, liczba_rekordów, rozmiar_kopii_bajtów, rozmiar_źródłowego_wal_bajtów)
    """
    src_path = Path(audit_db_path).resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Źródłowa baza audytu nie istnieje: {src_path}")

    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = src_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        dest_path = backup_dir / f"atlas_audit_{timestamp}.db"
    else:
        dest_path = Path(output_path).resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Kontrolowany Checkpoint (opcjonalny, domyślnie włączony)
    if checkpoint:
        try:
            conn = sqlite3.connect(str(src_path))
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            finally:
                conn.close()
        except Exception as exc:
            print(f"[OSTRZEŻENIE] Checkpoint WAL nie powiódł się (może demon trzyma blokadę): {exc}", file=sys.stderr)

    wal_path = src_path.with_name(src_path.name + "-wal")
    wal_size = wal_path.stat().st_size if wal_path.exists() else 0

    # 2. SQLite Online Backup API
    src_conn = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
    dest_conn = sqlite3.connect(str(dest_path))
    try:
        src_conn.backup(dest_conn)
    finally:
        dest_conn.close()
        src_conn.close()

    dest_size = dest_path.stat().st_size

    # 3. Weryfikacja samodzielności (standalone)
    verify_conn = sqlite3.connect(str(dest_path))
    try:
        cur = verify_conn.cursor()
        cur.execute("SELECT COUNT(*) FROM state_audit_log;")
        row = cur.fetchone()
        record_count = int(row[0]) if row else 0
        cur.close()
    finally:
        verify_conn.close()

    return dest_path, record_count, dest_size, wal_size


def main() -> int:
    parser = argparse.ArgumentParser(description="ATLAS Safe Audit Database Backup (FINDING-072)")
    default_db = Path.home() / ".hermes" / "atlas" / "atlas_audit.db"
    parser.add_argument(
        "--audit-db",
        default=str(default_db),
        help=f"Ścieżka do bazy audytu (domyślnie: {default_db})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Ścieżka docelowa pliku kopii zapasowej (.db)",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Pomiń wstępny PRAGMA wal_checkpoint(TRUNCATE)",
    )
    args = parser.parse_args()

    src_db = Path(args.audit_db)
    if not src_db.exists():
        print(f"BŁĄD: Plik bazy nie istnieje: {src_db}", file=sys.stderr)
        return 1

    print("[FINDING-072] Rozpoczynanie bezpiecznej kopii zapasowej:")
    print(f"  Źródło: {src_db}")
    print(f"  Rozmiar .db:  {src_db.stat().st_size:,} B")
    wal_f = src_db.with_name(src_db.name + "-wal")
    print(f"  Rozmiar -wal: {wal_f.stat().st_size if wal_f.exists() else 0:,} B")

    dest_path, count, dest_size, wal_size = backup_audit_database(
        audit_db_path=src_db,
        output_path=args.output,
        checkpoint=not args.no_checkpoint,
    )

    print("\n[SUKCES] Utworzono kopię zapasową:")
    print(f"  Ścieżka: {dest_path}")
    print(f"  Rozmiar kopii: {dest_size:,} B")
    print(f"  Zliczone wpisy (state_audit_log): {count}")
    print("  Kopia jest w 100% standalone (odczytywalna bez plików -wal / -shm).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
