"""Test walidujący strukturę docs/benchmark_baseline.json.

Weryfikuje że baseline jest poprawny, kompletny i zgodny z bieżącym benchmarkiem.
Wprowadzony po RED FLAG V17 — baseline JSON nie był walidowany przez testy,
więc mutacje w JSON przechodziły niezauważone.
"""
import json
from pathlib import Path

import pytest

_BASELINE_PATH = Path(__file__).parent.parent.parent / "docs" / "benchmark_baseline.json"

# CI nie ma baselines (nie idą na GitHub) — skip cały moduł
if not _BASELINE_PATH.exists():
    pytest.skip("baseline JSON not in public repo", allow_module_level=True)
_REQUIRED_TOP_KEYS = {"version", "python", "total_benchmarks", "passed", "metrics"}
_REQUIRED_METRIC_KEYS = {"target_max_duration_ms", "description"}
_EXPECTED_BENCHMARK_COUNT = 15


def test_baseline_json_exists() -> None:
    """Baseline JSON musi istnieć w docs/."""
    assert _BASELINE_PATH.exists(), f"Brak pliku: {_BASELINE_PATH}"
    assert _BASELINE_PATH.stat().st_size > 0, "Baseline JSON jest pusty"


def test_baseline_json_valid_syntax() -> None:
    """Baseline musi być poprawnym JSON."""
    with open(_BASELINE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_baseline_json_top_level_keys() -> None:
    """Wymagane klucze najwyższego poziomu."""
    with open(_BASELINE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    missing = _REQUIRED_TOP_KEYS - set(data.keys())
    assert not missing, f"Brak kluczy: {missing}"


def test_baseline_json_passed_matches_total() -> None:
    """'passed' musi równać się 'total_benchmarks' (baseline = wszystko przeszło)."""
    with open(_BASELINE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert data["total_benchmarks"] == _EXPECTED_BENCHMARK_COUNT, (
        f"total_benchmarks = {data['total_benchmarks']}, oczekiwano {_EXPECTED_BENCHMARK_COUNT}"
    )
    assert data["passed"] == data["total_benchmarks"], (
        f"passed ({data['passed']}) != total ({data['total_benchmarks']})"
    )
    assert data["passed"] > 0, "passed musi być > 0"


def test_baseline_json_version_string() -> None:
    """version musi być niepustym stringiem."""
    with open(_BASELINE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data["version"], str) and data["version"], "version musi być niepustym stringiem"


def test_baseline_json_python_version() -> None:
    """python musi być 3.12 lub 3.14 (obsługiwane wersje)."""
    with open(_BASELINE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert data["python"] in ("3.12", "3.14"), f"python = {data['python']}, oczekiwano 3.12 lub 3.14"


def test_baseline_json_metrics_count() -> None:
    """metrics musi zawierać dokładnie _EXPECTED_BENCHMARK_COUNT wpisów."""
    with open(_BASELINE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["metrics"]) == _EXPECTED_BENCHMARK_COUNT, (
        f"metrics ma {len(data['metrics'])} wpisów, oczekiwano {_EXPECTED_BENCHMARK_COUNT}"
    )


def test_baseline_json_metric_structure() -> None:
    """Każdy metric musi mieć target_max_duration_ms (positive number) + description."""
    with open(_BASELINE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for name, metric in data["metrics"].items():
        missing_keys = _REQUIRED_METRIC_KEYS - set(metric.keys())
        assert not missing_keys, f"Metric {name} brak kluczy: {missing_keys}"
        target = metric["target_max_duration_ms"]
        assert isinstance(target, (int, float)) and target > 0, (
            f"Metric {name}: target_max_duration_ms={target} musi być > 0"
        )
        assert isinstance(metric["description"], str) and metric["description"], (
            f"Metric {name}: description musi być niepustym stringiem"
        )
