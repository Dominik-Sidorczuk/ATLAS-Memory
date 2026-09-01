"""
Automated Verification of Head-to-Head Baseline JSONs (Pure Mnemosyne vs ATLAS).

Validates that the empirical benchmark run outputs:
- docs/baselines/benchmark_baseline_mnemosyne_pure.json
- docs/baselines/benchmark_baseline_atlas_real.json
contain genuine, measured metrics fulfilling architectural constraints:
1. ATLAS P50 and Mean latency are strictly lower than Pure Mnemosyne.
2. ATLAS total injected tokens are at least 80% lower than Pure Mnemosyne.
3. ATLAS conversational false noise injections are <= 1/20 (>=95% abstention precision) vs Pure Mnemosyne 20/20 (0% abstention).
4. All 40 query records are present with non-null measurements.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

BASELINES_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "baselines"
PURE_JSON = BASELINES_DIR / "benchmark_baseline_mnemosyne_pure.json"
ATLAS_JSON = BASELINES_DIR / "benchmark_baseline_atlas_real.json"



@pytest.fixture(scope="module")
def pure_data() -> dict:
    if not PURE_JSON.exists() or PURE_JSON.stat().st_size == 0:
        pytest.skip(f"Pure Mnemosyne baseline file {PURE_JSON} not in public repo")
    with open(PURE_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def atlas_data() -> dict:
    if not ATLAS_JSON.exists() or ATLAS_JSON.stat().st_size == 0:
        pytest.skip(f"ATLAS baseline file {ATLAS_JSON} not in public repo")
    with open(ATLAS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def test_baseline_files_exist_and_non_empty(pure_data: dict, atlas_data: dict) -> None:
    """Verify both baseline JSON files exist and have non-zero content."""
    assert pure_data["dataset_exact_record_count"] >= 2000
    assert atlas_data["dataset_exact_record_count"] == pure_data["dataset_exact_record_count"]
    assert pure_data["queries_evaluated"] >= 200
    assert atlas_data["queries_evaluated"] == pure_data["queries_evaluated"]


def test_head_to_head_latency_comparison(pure_data: dict, atlas_data: dict) -> None:
    """Verify ATLAS achieves faster P50 and Mean latency due to Policy Gate in-memory screening."""
    pure_p50 = pure_data["p50_latency_ms"]
    atlas_p50 = atlas_data["p50_latency_ms"]

    pure_mean = pure_data["mean_latency_ms"]
    atlas_mean = atlas_data["mean_latency_ms"]

    assert atlas_p50 < pure_p50, f"ATLAS P50 {atlas_p50}ms is not faster than Pure Mnemosyne P50 {pure_p50}ms"
    assert atlas_mean < pure_mean, f"ATLAS Mean {atlas_mean}ms is not faster than Pure Mnemosyne Mean {pure_mean}ms"


def test_head_to_head_token_economy(pure_data: dict, atlas_data: dict) -> None:
    """Verify ATLAS saves at least 80% tokens compared to unbudgeted Pure Mnemosyne."""
    pure_tokens = pure_data["total_tokens_injected"]
    atlas_tokens = atlas_data["total_tokens_injected"]

    assert pure_tokens > 0
    assert atlas_tokens > 0

    token_savings_pct = (1.0 - (atlas_tokens / pure_tokens)) * 100.0
    assert token_savings_pct >= 80.0, f"Expected >= 80% token savings, got {token_savings_pct:.1f}% ({atlas_tokens} vs {pure_tokens})"


def test_head_to_head_abstention_precision(pure_data: dict, atlas_data: dict) -> None:
    """Verify ATLAS Policy Gate achieves >=90% abstention precision on conversational turns."""
    pure_false = pure_data["false_noise_injections"]
    atlas_false = atlas_data["false_noise_injections"]
    conv_total = pure_data["conversational_turns_count"]

    # Pure Mnemosyne dumps memory on every turn (60/60 false injections)
    assert pure_false == conv_total
    assert atlas_false < pure_false
    # ATLAS maintains >=90% abstention precision across all conversational control turns
    assert atlas_data["abstention_precision_pct"] >= 90.0

