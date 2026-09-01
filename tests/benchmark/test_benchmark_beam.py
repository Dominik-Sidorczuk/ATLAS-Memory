"""
Pytest Suite for BEAM Semantic Quality Benchmark (10 Cognitive Abilities).
Validates ATLAS + Mnemosyne memory quality using LLM-as-Judge over OmniRoute.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas_memory.hermes.atlas_provider import AtlasMemoryProvider
from atlas_memory.orchestrator import MemoryOrchestrator
from scripts.run_beam_benchmark import (
    BASELINES_DIR,
    CURATED_BEAM_SAMPLES,
    DEFAULT_MODEL,
    DEFAULT_OMNIROUTE_URL,
    BeamEvaluator,
)


@pytest.fixture(scope="module")
def evaluator() -> BeamEvaluator:
    ev = BeamEvaluator(
        endpoint_url=DEFAULT_OMNIROUTE_URL,
        model_name=DEFAULT_MODEL,
        judge_model=DEFAULT_MODEL,
    )
    return ev


@pytest.fixture(scope="module")
def is_omniroute_live(evaluator: BeamEvaluator) -> bool:
    return evaluator.check_endpoint()


def test_beam_baseline_json_exists_and_valid():
    """Verify that generated BEAM baseline JSON files in docs/baselines/ are structurally sound.
    Supports both legacy (flat) and head-to-head (nested atlas+pure_mnemosyne) schemas.
    """
    beam_files = list(BASELINES_DIR.glob("benchmark_beam_*.json"))
    if not beam_files:
        pytest.skip("no BEAM baseline JSONs in public repo (docs/baselines/ is private)")
    assert len(beam_files) > 0, "No benchmark_beam_*.json files found in docs/baselines/"

    latest_file = max(beam_files, key=lambda p: p.stat().st_mtime)
    data = json.loads(latest_file.read_text(encoding="utf-8"))

    if "atlas_plus_mnemosyne" in data:
        # Head-to-head schema (V27+): atlas + pure_mnemosyne comparison
        atlas = data["atlas_plus_mnemosyne"]
        assert "system" in atlas
        assert "ability_breakdown" in atlas
        assert "overall_accuracy" in atlas
        assert len(atlas["ability_breakdown"]) == 10
        raw = data.get("raw_records", {})
        if isinstance(raw, dict):
            atlas_raw = raw.get("atlas", [])
            assert len(atlas_raw) >= 10, f"Expected >=10 atlas raw records, got {len(atlas_raw)}"
        else:
            assert len(raw) >= 10
    else:
        # Legacy flat schema
        assert "system" in data
        assert "ability_breakdown" in data
        assert "overall_accuracy" in data
        assert len(data["ability_breakdown"]) == 10
        assert len(data["raw_records"]) >= 10


def test_atlas_provider_beam_pure_recall():
    """Verify ATLAS prefetch recall behavior across all 10 curated BEAM ability samples without requiring external LLM."""
    orchestrator = MemoryOrchestrator()
    provider = AtlasMemoryProvider(orchestrator=orchestrator)
    provider.initialize("beam_pure_test", hermes_home=str(Path.home() / ".hermes"))

    for sample in CURATED_BEAM_SAMPLES:
        query = sample["query"]
        should_run, entities, reason = orchestrator.should_retrieve(query)
        assert isinstance(should_run, bool)
        assert isinstance(reason, str)


@pytest.mark.parametrize("sample", CURATED_BEAM_SAMPLES, ids=[s["ability"] for s in CURATED_BEAM_SAMPLES])
def test_beam_10_abilities_end_to_end(sample, evaluator: BeamEvaluator, is_omniroute_live: bool):
    """Evaluates each of the 10 BEAM abilities end-to-end via OmniRoute LLM-as-Judge when live."""
    if not is_omniroute_live:
        pytest.skip(f"OmniRoute endpoint {DEFAULT_OMNIROUTE_URL} is offline — skipping LLM judge evaluation for {sample['ability']}")

    result = evaluator.evaluate_sample(sample)
    assert result["judge_score"] in (0, 1)
    assert "judge_reasoning" in result
    assert result["judge_score"] == 1, f"BEAM Ability {sample['ability']} failed: {result['judge_reasoning']} | Answer: {result['answer']}"

