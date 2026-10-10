#!/usr/bin/env python3
"""V98 F9 (BUG-CAL35): policz próg σ² unstable z REALNEGO pomiaru i zapisz proweniencję.

Liczy rozkład sigma2 self-consistency (K=3 independent, latenty w
`self_consistency.latents`) z pliku cache sędziego, wypisuje statystyki
(N, P50/P90/P95/max, % unstable) i zapisuje plik proweniencji
`docs/calib/sigma2_threshold_provenance.json` (source_file, n, p95, computed_at,
sha256 pliku źródłowego). Runner (`build_retrieval_calibration.py`) czyta próg
Z TEGO pliku (`resolve_unstable_sigma2_threshold`) — nie z ręcznie wpisanej stałej.

Zero LLM, zero GPU — tylko odczyt cache.

Użycie:
    python scripts/recompute_sigma2_threshold.py [--cache docs/calib/cache_v93_sc.jsonl]
        [--out docs/calib/sigma2_threshold_provenance.json]
        [--percentile 95] [--distribution]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_retrieval_calibration import (  # noqa: E402
    DEFAULT_JUDGE_CACHE,
    DOCS_CALIB_DIR,
    JUDGE_PROMPT_VERSION,
    SIGMA2_THRESHOLD_PROVENANCE_FIELDS,
    resolve_unstable_sigma2_threshold,
    sha256_file,
)


def collect_sigma2(cache_path: Path) -> List[float]:
    """Wszystkie zapisane sigma2 z pliku cache (agregaty self-consistency)."""
    values: List[float] = []
    with cache_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            sc = rec.get("self_consistency") if isinstance(rec, dict) else None
            if isinstance(sc, dict) and sc.get("sigma2") is not None:
                values.append(float(sc["sigma2"]))
    return values


def percentile(values: List[float], q: float) -> float:
    """Percentyl bez numpy — interpolacja liniowa jak np.percentile."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (q / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def main() -> int:
    ap = argparse.ArgumentParser(description="Przelicz próg sigma2 z realnego pomiaru (BUG-CAL35)")
    ap.add_argument("--cache", type=Path, default=DEFAULT_JUDGE_CACHE,
                    help="Plik cache sędziego z rekordami self_consistency (default: docs/calib/cache_v93_sc.jsonl)")
    ap.add_argument("--out", type=Path, default=DOCS_CALIB_DIR / "sigma2_threshold_provenance.json",
                    help="Plik proweniencji progu (default: docs/calib/sigma2_threshold_provenance.json)")
    ap.add_argument("--percentile", type=float, default=95.0,
                    help="Percentyl rozkładu zapisywany jako próg (domyślnie 95; pole `p95`)")
    ap.add_argument("--distribution", action="store_true",
                    help="Zapisz też pełny rozkład sigma2 w pliku proweniencji")
    ap.add_argument("--unstable-threshold", type=float, default=None,
                    help="Próg do raportu proc. unstable (default: P95 z tego samego rozkładu)")
    ap.add_argument("--require-version-match", action=argparse.BooleanOptionalAction, default=True,
                    help="V100 F1 (CAL37): przy zapisie zawsze stempluj aktualną wersję promptu "
                         "i zwaliduj odczyt proweniencji przez resolver (domyślnie ON; "
                         "--no-require-version-match pomija read-back walidację)")
    args = ap.parse_args()

    cache_path = args.cache if args.cache.is_absolute() else (REPO_ROOT / args.cache)
    if not cache_path.is_file():
        print(f"ERROR: brak pliku cache: {cache_path}", file=sys.stderr)
        return 1

    values = collect_sigma2(cache_path)
    if not values:
        print(f"ERROR: plik {cache_path} nie zawiera żadnych sigma2 (self_consistency)", file=sys.stderr)
        return 1

    n = len(values)
    p50 = percentile(values, 50.0)
    p90 = percentile(values, 90.0)
    p95 = percentile(values, 95.0)
    vmax = max(values)
    rep_thr = args.unstable_threshold if args.unstable_threshold is not None else p95
    n_unstable = sum(1 for v in values if v > rep_thr)

    def _display(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:  # cache poza repo (np. tmp_path w teście) — zapisujemy jak jest
            return str(p)

    doc: Dict[str, Any] = {
        "source_file": _display(cache_path),
        "n": n,
        "p95": round(p95, 6),
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": sha256_file(cache_path),
    }
    # V100 F1 (CAL37): próg σ² wiążemy z wersją promptu sędziego — zawsze zapisujemy
    # aktualną stałą; czytający (resolve_unstable_sigma2_threshold) odrzuci proweniencję
    # bez tego pola albo z niezgodną wersją (fallback głośny, na stałą awaryjną).
    doc["judge_prompt_version"] = JUDGE_PROMPT_VERSION
    doc["percentile"] = args.percentile
    doc["p50"] = round(p50, 6)
    doc["p90"] = round(p90, 6)
    doc["max"] = round(vmax, 6)
    doc["unstable_fraction"] = round(n_unstable / n, 6)
    doc["unstable_threshold_reported"] = round(rep_thr, 6)
    if args.distribution:
        doc["distribution"] = values

    missing = [k for k in SIGMA2_THRESHOLD_PROVENANCE_FIELDS if k not in doc or doc[k] is None]
    if missing:
        print(f"ERROR: proweniencja niekompletna (brak pól: {missing})", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.out)

    # V100 F1 (CAL37): read-back — odczyt przez TE SAME semantyki, które obowiązują
    # konsumenta (resolver + walidacja wersji promptu). Bez tego zapis mógłby "udać"
    # proweniencję, której nikt ważny nie przyjmie.
    if args.require_version_match:
        resolved = resolve_unstable_sigma2_threshold(args.out)
        if resolved is None:
            print(
                "ERROR: read-back odrzucony — resolver nie przyjął świeżo zapisanej "
                f"proweniencji (wersja promptu: {doc['judge_prompt_version']})",
                file=sys.stderr,
            )
            return 1
        if abs(resolved - doc["p95"]) > 1e-9:
            print(
                f"ERROR: read-back niezgodny: resolver={resolved} != zapisany p95={doc['p95']}",
                file=sys.stderr,
            )
            return 1

    print(f"source_file : {doc['source_file']}")
    print(f"sha256      : {doc['sha256']}")
    print(f"N           : {n}")
    print(f"P50         : {p50:.5f}")
    print(f"P90         : {p90:.5f}")
    print(f"P95         : {p95:.5f}")
    print(f"max         : {vmax:.5f}")
    print(f"unstable    : {n_unstable}/{n} = {100.0 * n_unstable / n:.2f}%  (thr={rep_thr:.5f})")
    print(f"provenance  : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
