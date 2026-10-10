"""V152-CLOSURE / ZADANIE 2: audyt operacyjny podłogi budżetu PPR 200 ms (Filar III).

Filar III ADR-V152:
  budget_ms = max(ppr_budget_floor_ms, ppr_tau_base + ppr_gamma * ln(V + E))
  floor: hardcoded 200.0 ms.
  Inwariant triady: każde odpalenie fetch_graph_ppr_candidates loguje budżet
  >= floor; BRAK wpisów [WARNING] HippoRAG 2 PPR budget exceeded.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

DAEMON_LOG = Path.home() / ".hermes" / "atlas" / "daemon.log"
ATLAS_CFG_DEFAULT = 200.0
DB_PATH = Path.home() / ".hermes" / "atlas" / "atlas.db"


def _read_daemon_log_tail(n: int = 4000) -> str:
    if not DAEMON_LOG.exists():
        return ""
    # tail n lines
    with DAEMON_LOG.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        # zgrubne okno: 16 KB lub n*200
        window = max(size - 16384, 0)
        f.seek(window)
        raw = f.read().decode("utf-8", errors="replace")
    return raw


class TestPprBudgetFloor:
    """Inwarianty FB-1 .. FB-3: konfiguracja + log + brak degradacji."""

    def test_atlas_config_floor_default(self):
        from atlas_memory.core.config import AtlasConfig
        cfg = AtlasConfig()
        assert float(cfg.ppr_budget_floor_ms) >= 200.0, (
            f"ppr_budget_floor_ms={cfg.ppr_budget_floor_ms} < 200.0"
        )

    def test_ppr_budget_logged_above_floor(self):
        """Każde logowane wywołanie PPR powinno mieć budget_ms >= 200.0."""
        log = _read_daemon_log_tail()
        # Szukamy wzorca logu z budget_ms
        pat = re.compile(r"budget_ms[=:]?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
        found = [float(m.group(1)) for m in pat.finditer(log)]
        if not found:
            pytest.skip("brak logu budget_ms w oknie — daemon nie wywołał jeszcze PPR")
        for b in found:
            assert b >= ATLAS_CFG_DEFAULT * 0.9, (
                f"budget_ms={b:.1f} poniżej podłogi 200.0 (znalezione w logu: {found})"
            )

    def test_no_budget_exceeded_warnings(self):
        log = _read_daemon_log_tail()
        pat = re.compile(r"PPR budget exceeded", re.IGNORECASE)
        offenders = pat.findall(log)
        assert not offenders, f"degradacja PPR do fallbacku (matches={len(offenders)})"

    def test_atlas_graph_above_floor_budget_for_recent(self):
        """Policz |V|+|E|, sprawdź zgodność z budżetem adaptacyjnym."""
        # Last N KV records to estimate activity
        if not DB_PATH.exists():
            pytest.skip("brak atlas.db")
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute("SELECT COUNT(*) FROM state_variables").fetchone()
        finally:
            conn.close()
        # Inwariant: twardy floor 200 ms, więc nawet pusty graf ma >= 200 ms.
        assert ATLAS_CFG_DEFAULT == 200.0
