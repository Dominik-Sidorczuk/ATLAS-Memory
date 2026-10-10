#!/usr/bin/env python3
"""Measure empirical self-consistency variance (sigma^2) on 40 probe pairs (T13a/T13b).

Dataset: 20 positives + 20 negatives from docs/baselines/calibration_dataset.jsonl (seed=42).
Pairs 1-10 hit docs/calib/cache_v94_smoke.jsonl; remaining pairs run live against local judge.
Output artifact: docs/calib/sc_sigma_smoke40.json with full indep_latents per pair.
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project src and root are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.append(str(REPO_ROOT / "src"))
sys.path.append(str(REPO_ROOT))

from scripts.build_retrieval_calibration import (
    DEFAULT_DB,
    DEFAULT_JUDGE_TIMEOUT,
    AsyncRetrievalJudge,
    load_facts,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Measure Self-Consistency Variance on 40 Pairs")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB), help="Path to atlas.db (read-only)")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(REPO_ROOT / "docs" / "baselines" / "calibration_dataset.jsonl"),
        help="Path to calibration dataset",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(REPO_ROOT / "docs" / "calib" / "sc_sigma_smoke40.json"),
        help="Output JSON artifact path",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default=str(REPO_ROOT / "docs" / "calib" / "cache_v94_smoke40.jsonl"),
        help="Target cache file for new evaluations",
    )
    parser.add_argument(
        "--smoke-cache",
        type=str,
        default=str(REPO_ROOT / "docs" / "calib" / "cache_v94_smoke.jsonl"),
        help="Warmup cache containing the first 10 smoke pairs",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.getenv("ATLAS_JUDGE_URL", "http://127.0.0.1:20122/v1/chat/completions"),
        help="LLM Judge endpoint",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("ATLAS_JUDGE_MODEL", "local"),
        help="Judge model identifier",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("ATLAS_JUDGE_TIMEOUT", str(DEFAULT_JUDGE_TIMEOUT))),
        help="Timeout per request in seconds",
    )
    parser.add_argument("--k", type=int, default=3, help="Number of self-consistency samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")
    parser.add_argument("--n-pos", type=int, default=20, help="Number of positive pairs")
    parser.add_argument("--n-neg", type=int, default=20, help="Number of negative pairs")
    return parser.parse_args()


def load_pairs(dataset_path: Path, facts: Dict[str, Any], n_pos: int, n_neg: int, seed: int):
    pos_candidates = []
    neg_candidates = []

    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("query_type") != "prose":
                continue
            ck = r.get("cand_key")
            if not ck or ck not in facts:
                continue
            text = facts[ck].get("text", "")
            if len(text.split()) < 8:
                continue
            item = {
                "q": r["query"],
                "ck": ck,
                "ds_label": r["label"],
                "neg_type": r.get("neg_type", "positive" if r["label"] == 1 else "negative"),
                "target_key": r.get("target_key"),
            }
            if r["label"] == 1 and ck == r.get("target_key"):
                pos_candidates.append(item)
            elif r["label"] == 0:
                neg_candidates.append(item)

    # Pairs 1-10 must match smoke 10: first 5 pos + first 5 neg
    smoke_pos = pos_candidates[:5]
    smoke_neg = neg_candidates[:5]

    rem_pos = pos_candidates[5:]
    rem_neg = neg_candidates[5:]

    # Sample remaining if needed
    req_pos = n_pos - len(smoke_pos)
    req_neg = n_neg - len(smoke_neg)

    chosen_rem_pos = rem_pos[:req_pos]
    chosen_rem_neg = rem_neg[:req_neg]

    # Structure 40 pairs:
    # 1..5: smoke positives
    # 6..10: smoke negatives
    # 11..11+req_pos-1: remaining positives
    # 11+req_pos..40: remaining negatives
    selected = smoke_pos + smoke_neg + chosen_rem_pos + chosen_rem_neg
    return selected


def compute_threshold_crosstabs(records: List[Dict[str, Any]]):
    valid_sigmas = [r["sigma2"] for r in records if r.get("sigma2") is not None]
    if not valid_sigmas:
        return {}

    sorted_s = sorted(valid_sigmas)
    n = len(sorted_s)

    def _quantile(p):
        idx = max(0, min(int(p * n) - 1, n - 1))
        return sorted_s[idx]

    p90 = round(float(_quantile(0.90)), 5)
    p95 = round(float(_quantile(0.95)), 5)
    p99 = round(float(_quantile(0.99)), 5)

    candidates = {
        "P90": p90,
        "P95": p95,
        "P99": p99,
        "const_0.02": 0.02,
        "const_0.05": 0.05,
        "const_0.10": 0.10,
    }

    crosstab = {}
    for name, thresh in candidates.items():
        unstable_recs = [r for r in records if r.get("sigma2") is not None and r["sigma2"] > thresh]
        pos_unstable = [r for r in unstable_recs if r.get("ds_label") == 1]
        neg_unstable = [r for r in unstable_recs if r.get("ds_label") == 0]

        neg_breakdown = {}
        for r in neg_unstable:
            ntype = r.get("neg_type", "unknown")
            neg_breakdown[ntype] = neg_breakdown.get(ntype, 0) + 1

        crosstab[name] = {
            "threshold": thresh,
            "total_unstable": len(unstable_recs),
            "unstable_pct": round(len(unstable_recs) / len(records) * 100, 2),
            "pos_unstable": len(pos_unstable),
            "neg_unstable": len(neg_unstable),
            "neg_breakdown": neg_breakdown,
            "unstable_pairs": [r["cand_key"] for r in unstable_recs],
        }

    return {
        "n_evaluated": len(records),
        "mean_sigma2": round(float(statistics.mean(valid_sigmas)), 5),
        "median_sigma2": round(float(statistics.median(valid_sigmas)), 5),
        "max_sigma2": round(float(max(valid_sigmas)), 5),
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "crosstab": crosstab,
    }


async def async_main():
    args = parse_args()
    print("=== MEASURE SC SIGMA40 (T13a/T13b) ===")
    print(f"DB: {args.db_path} (mode=ro)")
    print(f"Dataset: {args.dataset}")
    print(f"Out: {args.out_json}")
    print(f"Cache: {args.cache_file}")
    print(f"Smoke cache: {args.smoke_cache}")
    print(f"Endpoint: {args.endpoint} | Model: {args.model} | Timeout: {args.timeout}s")

    facts = {f["key"]: f for f in load_facts(args.db_path)}
    print(f"Loaded {len(facts)} facts from DB")

    pairs = load_pairs(Path(args.dataset), facts, args.n_pos, args.n_neg, args.seed)
    print(f"Selected {len(pairs)} probe pairs (pos={args.n_pos}, neg={args.n_neg})")

    out_json_path = Path(args.out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    judge = AsyncRetrievalJudge(
        endpoint=args.endpoint,
        enabled=True,
        sc_mode="independent",
        model=args.model,
        cache_path=cache_path,
    )

    # Pre-warm judge cache with smoke cache entries if available
    smoke_cache_path = Path(args.smoke_cache)
    prewarmed = 0
    if smoke_cache_path.exists():
        for ln in smoke_cache_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
                k = rec.get("k")
                if k:
                    judge.cache[k] = rec
                    prewarmed += 1
            except Exception:
                pass
    print(f"Prewarmed judge with {prewarmed} entries from smoke cache {smoke_cache_path.name}")

    results = []
    # Resume from existing results if already written
    if out_json_path.exists():
        try:
            existing = json.loads(out_json_path.read_text(encoding="utf-8"))
            if isinstance(existing, list) and len(existing) > 0:
                results = existing
                print(f"Resuming with {len(results)} previously completed records from {out_json_path.name}")
        except Exception:
            pass

    start_idx = len(results)
    for i in range(start_idx, len(pairs)):
        p = pairs[i]
        pair_num = i + 1
        cand_text = facts[p["ck"]]["text"]

        t0 = time.time()
        verdict = await judge.judge_batch_self_consistency(
            query=p["q"],
            cand_key=p["ck"],
            cand_text=cand_text,
            k=args.k,
            sc_mode="independent",
        )
        dt = time.time() - t0

        sc = verdict.self_consistency or {}
        rec = {
            "i": pair_num,
            "q": p["q"][:70],
            "cand_key": p["ck"],
            "ds_label": p["ds_label"],
            "neg_type": p["neg_type"],
            "label": verdict.label,
            "latent": verdict.latent_score,
            "source": verdict.source,
            "indep_latents": sc.get("latents"),
            "indep_labels": sc.get("labels"),
            "mode": sc.get("mode", "independent"),
            "sigma2": sc.get("sigma2"),
            "unstable": sc.get("unstable"),
            "thinking_chars": [len(verdict.thinking or "")] if verdict.thinking else None,
            "reason": (verdict.reasoning or "")[:130],
            "dt_s": round(dt, 1),
        }
        results.append(rec)

        # Atomic periodic save to disk
        out_json_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

        print(
            f"[{pair_num:2d}/{len(pairs)}] dt={dt:6.1f}s lbl={verdict.label} lat={verdict.latent_score:.3f} "
            f"sig2={sc.get('sigma2')} unst={sc.get('unstable')} src={verdict.source} "
            f"latents={sc.get('latents')}",
            flush=True,
        )

    # Compute final statistics and crosstab
    summary = compute_threshold_crosstabs(results)
    print("\n=== PODSUMOWANIE METROLOGICZNE T13b ===")
    print(f"Liczba par: {summary.get('n_evaluated')}")
    print(f"Średnia sigma2: {summary.get('mean_sigma2')}")
    print(f"Mediana sigma2: {summary.get('median_sigma2')}")
    print(f"Max sigma2: {summary.get('max_sigma2')}")
    print(f"Percentyle sigma2: P90={summary.get('p90')}, P95={summary.get('p95')}, P99={summary.get('p99')}")
    print("\nCross-tab wariantów progu unstable:")
    print(f"{'Wariant':<12} {'Próg':<8} {'Unstable':<10} {'%':<6} {'Pos':<5} {'Neg':<5} {'Pary niestabilne'}")
    print("-" * 75)
    for name, c in summary.get("crosstab", {}).items():
        pairs_str = ", ".join(k.split(":")[-1][:15] for k in c["unstable_pairs"])
        print(
            f"{name:<12} {c['threshold']:<8.4f} {c['total_unstable']:<10} {c['unstable_pct']:<6.1f} "
            f"{c['pos_unstable']:<5} {c['neg_unstable']:<5} {pairs_str}"
        )

    summary_file = out_json_path.with_name("sc_sigma_smoke40_summary.json")
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nZapisano podsumowanie do {summary_file}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
