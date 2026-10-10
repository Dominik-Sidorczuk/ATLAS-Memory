"""Unit tests for Metrological LLM Judge (V93 SOTA) & Active Uncertainty Sampling."""
from __future__ import annotations

from atlas_memory.cognitive.retrieval.score import score_candidates
from scripts.build_retrieval_calibration import (
    _extract_json_block,
    active_sampling_prior,
    compute_latent_score,
)


def test_compute_latent_score_perfect_match():
    rubric = {
        "semantic_entailment": {"score": 3, "rationale": "Idealna zgodność"},
        "operational_actionability": {"score": 3, "rationale": "Ścisła reguła"},
        "entity_scope_grounding": {"score": 2, "rationale": "Dokładna encja"},
        "polarity_fidelity": {"score": 2, "rationale": "Spójność modalna"},
        "epistemic_freshness": {"score": 2, "rationale": "Świeży stan"},
        "hard_veto": {"detected": False, "reason": "brak sprzeczności"},
    }
    y_latent, label, reason = compute_latent_score(rubric)
    assert y_latent == 1.0
    assert label == 1
    assert "y=1.000" in reason


def test_compute_latent_score_hard_veto_detected():
    rubric = {
        "semantic_entailment": 3,
        "operational_actionability": 3,
        "entity_scope_grounding": 2,
        "polarity_fidelity": 2,
        "epistemic_freshness": 2,
        "hard_veto": {"detected": True, "reason": "Zakaz modyfikacji Hermes core naruszony"},
    }
    y_latent, label, reason = compute_latent_score(rubric)
    assert y_latent == 0.0
    assert label == 0
    assert "hard_veto" in reason


def test_compute_latent_score_polarity_inversion_veto():
    # Odwrócenie polaryzacji (nakaz zamiast zakazu) p_pol == 0 natychmiast zeruje wynik
    rubric = {
        "semantic_entailment": {"score": 3},
        "operational_actionability": {"score": 3},
        "entity_scope_grounding": {"score": 2},
        "polarity_fidelity": {"score": 0},
        "epistemic_freshness": {"score": 2},
        "hard_veto": {"detected": False},
    }
    y_latent, label, reason = compute_latent_score(rubric)
    assert y_latent == 0.0
    assert label == 0
    assert "polarity_inversion_veto" in reason


def test_compute_latent_score_partial_match():
    rubric = {
        "semantic_entailment": 1,
        "operational_actionability": 1,
        "entity_scope_grounding": 1,
        "polarity_fidelity": 2,
        "epistemic_freshness": 1,
        "hard_veto": {"detected": False},
    }
    y_latent, label, reason = compute_latent_score(rubric)
    # y_latent = 0.3*(1/3) + 0.25*(1/3) + 0.2*(1/2) + 0.15*(2/2) + 0.10*(1/2)
    #          = 0.10 + 0.0833 + 0.10 + 0.15 + 0.05 = 0.4833
    assert 0.40 <= y_latent <= 0.50
    assert label == 0  # y < 0.50 and s_ent < 1.8


def test_extract_json_block_with_thinking_and_markdown():
    text = """
<think>
Analizuję zgodność semantyczną...
Fakt jest częściowo zgodny, ale encja się różni.
</think>
```json
{
  "semantic_entailment": {"score": 2, "rationale": "częściowa"},
  "operational_actionability": {"score": 2, "rationale": "parametry są"},
  "entity_scope_grounding": {"score": 1, "rationale": "pokrewny moduł"},
  "polarity_fidelity": {"score": 2, "rationale": "brak inwersji"},
  "epistemic_freshness": {"score": 1, "rationale": "brak daty"},
  "hard_veto": {"detected": false, "reason": "brak sprzeczności"}
}
```
"""
    obj = _extract_json_block(text)
    assert obj is not None
    assert obj["semantic_entailment"]["score"] == 2
    assert obj["hard_veto"]["detected"] is False


def test_active_sampling_prior_rules():
    # 1. Symbolic match
    v = active_sampling_prior(0.9, 0.2, "key:a", "key:a", "symbolic")
    assert v is not None
    assert v.label == 1
    assert v.source == "symbolic_exact_match"

    # 2. Symbolic negative
    v = active_sampling_prior(0.9, 0.2, "key:b", "key:a", "symbolic")
    assert v is not None
    assert v.label == 0
    assert v.source == "symbolic_structural_negative"

    # 3. V93.1 (T2): Target prose MUST NOT be bypassed by score heuristics (must return None to call LLM judge)
    v = active_sampling_prior(0.92, 0.25, "fact:1", "fact:1", "prose", vector_score=0.85, store_keys={"fact:1"})
    assert v is None, "T2: Target positive must never bypass LLM judge via score heuristic"

    # 4. Structural absent negative (cand_key not in store_keys)
    v = active_sampling_prior(0.20, -0.30, "fact:distractor", "fact:1", "prose", vector_score=0.15, store_keys={"fact:1"})
    assert v is not None
    assert v.label == 0
    assert v.source == "structural_absent_negative"
    assert v.prior_used is True

    # 5. Existing rival in store_keys -> must call LLM judge (returns None)
    v = active_sampling_prior(0.65, 0.05, "fact:near_miss", "fact:1", "prose", vector_score=0.60, store_keys={"fact:1", "fact:near_miss"})
    assert v is None


def test_score_candidates_multidimensional_fields():
    cands = [
        {
            "key": "test:k1",
            "subject": "test:k1",
            "object": "To jest przykładowa reguła konfiguracyjna do testów",
            "vector_score": 0.85,
            "confidence": 1.0,
            "source_type": "user_explicit",
        }
    ]
    ranked = score_candidates("reguła konfiguracyjna", cands, fusion_strategy="dbsf")
    assert len(ranked) == 1
    top = ranked[0]
    assert "dense_score" in top
    assert "sparse_score" in top
    assert top["dense_score"] > 0.0
    assert top["metadata"]["dense_score"] == top["dense_score"]
    assert top["metadata"]["sparse_score"] == top["sparse_score"]


def test_extract_json_block_truncated_repair():
    truncated = '{\n  "semantic_entailment": {\n    "score": 0,\n    "rationale": "Fakt dotyczy limitu timeout"\n  },\n  "operational_actionability": {\n    "score": 1,\n    "rationale": "Ogólna reguła timeout"'
    res = _extract_json_block(truncated)
    assert res is not None
    assert res["semantic_entailment"]["score"] == 0
    assert res["operational_actionability"]["score"] == 1


def test_sc_probe_specs_orthogonal_dimensions():
    from scripts.build_retrieval_calibration import DEFAULT_SC_MODE, SC_PROBE_SPECS

    assert DEFAULT_SC_MODE == "batch"
    assert len(SC_PROBE_SPECS) == 3
    all_dims = []
    for spec in SC_PROBE_SPECS:
        dims = spec["measured_dimensions"]
        assert len(dims) >= 1
        all_dims.extend(dims)
    assert set(all_dims) == {
        "semantic_entailment",
        "operational_actionability",
        "entity_scope_grounding",
        "polarity_fidelity",
        "epistemic_freshness",
    }
    assert len(all_dims) == 5, "Probe dimensions must form an exact disjoint partition of the 5 rubric dimensions"


def test_batch_mode_orthogonal_probes_produce_nonzero_sigma2(tmp_path):
    import asyncio
    import json
    from unittest.mock import patch

    import scripts.build_retrieval_calibration as brc

    cache_file = tmp_path / "cache_batch_sc.jsonl"
    with patch.object(brc, "JUDGE_CACHE_FILE", cache_file):
        judge = brc.RetrievalJudge(enabled=True, sc_mode="batch", cache_path=cache_file)
        try:
            # Case 1: Cross-entity false positive where LLM fills identical 5D scores across all 3 batch objects:
            # P1 (ent=0.5, act=0.0) -> l_1 = 0.0909
            # P2 (scope=0.0) -> l_2 = 0.0
            # P3 (pol=2.0, fresh=2.0) -> l_3 = 1.0
            identical_cross_entity_obj = {
                "semantic_entailment": {"score": 0.5, "rationale": "Inny serwis"},
                "operational_actionability": {"score": 0.0, "rationale": "Brak"},
                "entity_scope_grounding": {"score": 0.0, "rationale": "Obca encja"},
                "polarity_fidelity": {"score": 2.0, "rationale": "Spójny tryb oznajmujący"},
                "epistemic_freshness": {"score": 2.0, "rationale": "Aktualny"},
                "hard_veto": {"detected": False, "reason": "brak"},
            }
            fake_batch_resp = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                [
                                    identical_cross_entity_obj,
                                    identical_cross_entity_obj,
                                    identical_cross_entity_obj,
                                ]
                            )
                        }
                    }
                ]
            }
            with patch.object(judge, "_blocking_http_call", return_value=fake_batch_resp) as mock_http:
                verdict = asyncio.run(
                    judge.judge_batch_self_consistency(
                        query="Jaki jest port serwera produkcyjnego Y?",
                        cand_key="fact:server_x",
                        cand_text="Serwer produkcyjny X stoi na porcie 8080.",
                        k=3,
                    )
                )
                assert mock_http.call_count == 1, "Batch mode must execute in 1 HTTP call (ADR-015 / ADR-020)"
                assert verdict.label == 0
                assert verdict.self_consistency is not None
                assert verdict.self_consistency["mode"] == "batch"
                assert verdict.self_consistency["latents"] == [0.0909, 0.0, 1.0]
                assert verdict.self_consistency["sigma2"] > judge.unstable_threshold
                assert verdict.self_consistency["unstable"] is True

            # Case 2: Unanimous true positive -> latents = [1.0, 1.0, 1.0], sigma2 = 0.0, unstable = False
            perfect_obj = {
                "semantic_entailment": {"score": 3.0, "rationale": "Pełne wynikania"},
                "operational_actionability": {"score": 3.0, "rationale": "Kompletna komenda"},
                "entity_scope_grounding": {"score": 2.0, "rationale": "Ta sama encja"},
                "polarity_fidelity": {"score": 2.0, "rationale": "Zgodny znak"},
                "epistemic_freshness": {"score": 2.0, "rationale": "Aktualne"},
                "hard_veto": {"detected": False, "reason": "brak"},
            }
            fake_tp_resp = {
                "choices": [
                    {"message": {"content": json.dumps([perfect_obj, perfect_obj, perfect_obj])}}
                ]
            }
            with patch.object(judge, "_blocking_http_call", return_value=fake_tp_resp):
                verdict_tp = asyncio.run(
                    judge.judge_batch_self_consistency(
                        query="Jaki jest port serwera produkcyjnego X?",
                        cand_key="fact:server_x",
                        cand_text="Serwer produkcyjny X stoi na porcie 8080.",
                        k=3,
                    )
                )
                assert verdict_tp.label == 1
                assert verdict_tp.latent_score == 1.0
                assert verdict_tp.self_consistency["latents"] == [1.0, 1.0, 1.0]
                assert verdict_tp.self_consistency["sigma2"] == 0.0
                assert verdict_tp.self_consistency["unstable"] is False
        finally:
            judge.close()


def test_independent_mode_neutral_unfocused_dimensions_preserve_true_positive(tmp_path):
    import asyncio
    import json
    from unittest.mock import patch

    import scripts.build_retrieval_calibration as brc

    cache_file = tmp_path / "cache_indep_sc.jsonl"
    with patch.object(brc, "JUDGE_CACHE_FILE", cache_file):
        judge = brc.RetrievalJudge(enabled=True, sc_mode="independent", cache_path=cache_file)
        try:
            # Under ADR-020 rule (b), P2 and P3 assign neutral midpoint 1.5 to unfocused semantic_entailment,
            # while scoring full marks on their own measured_dimensions.
            p1_obj = {
                "semantic_entailment": {"score": 3.0},
                "operational_actionability": {"score": 3.0},
                "entity_scope_grounding": {"score": 1.0},
                "polarity_fidelity": {"score": 1.0},
                "epistemic_freshness": {"score": 1.0},
                "hard_veto": {"detected": False},
            }
            p2_obj = {
                "semantic_entailment": {"score": 1.5},
                "operational_actionability": {"score": 1.5},
                "entity_scope_grounding": {"score": 2.0},
                "polarity_fidelity": {"score": 1.0},
                "epistemic_freshness": {"score": 1.0},
                "hard_veto": {"detected": False},
            }
            p3_obj = {
                "semantic_entailment": {"score": 1.5},
                "operational_actionability": {"score": 1.5},
                "entity_scope_grounding": {"score": 1.0},
                "polarity_fidelity": {"score": 2.0},
                "epistemic_freshness": {"score": 2.0},
                "hard_veto": {"detected": False},
            }
            responses = [
                {"choices": [{"message": {"content": json.dumps(p1_obj)}}]},
                {"choices": [{"message": {"content": json.dumps(p2_obj)}}]},
                {"choices": [{"message": {"content": json.dumps(p3_obj)}}]},
            ]
            with patch.object(judge, "_blocking_http_call", side_effect=responses):
                verdict = asyncio.run(
                    judge.judge_batch_self_consistency(
                        query="Jak uruchomić doktor pętli?",
                        cand_key="fact:doctor",
                        cand_text="Uruchom pixi run doctor w katalogu repozytorium.",
                        k=3,
                    )
                )
                assert verdict.label == 1
                assert verdict.latent_score == 1.0
                assert verdict.self_consistency["latents"] == [1.0, 1.0, 1.0]
                assert verdict.self_consistency["sigma2"] == 0.0
        finally:
            judge.close()


def test_sigma2_provenance_version_binding_and_hot_path_isolation(tmp_path):
    import json
    from pathlib import Path

    from scripts.build_retrieval_calibration import (
        JUDGE_PROMPT_VERSION,
        resolve_unstable_sigma2_threshold,
        sha256_file,
    )

    src_cache = tmp_path / "cache_v93_sc.jsonl"
    src_cache.write_text('{"self_consistency": {"sigma2": 0.05}}\n', encoding="utf-8")
    prov_file = tmp_path / "sigma2_threshold_provenance.json"
    prov_doc = {
        "source_file": str(src_cache),
        "n": 30,
        "p95": 0.1425,
        "computed_at": "2026-09-24T12:00:00+00:00",
        "sha256": sha256_file(src_cache),
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
    }
    prov_file.write_text(json.dumps(prov_doc), encoding="utf-8")
    assert resolve_unstable_sigma2_threshold(prov_file) == 0.1425

    # Version mismatch must reject provenance and return None
    prov_doc["judge_prompt_version"] = "stale_v0"
    prov_file.write_text(json.dumps(prov_doc), encoding="utf-8")
    assert resolve_unstable_sigma2_threshold(prov_file) is None

    # Hot-path isolation guard (V100 F6 / CAL40): src/atlas_memory must not reference sigma2_threshold_provenance
    src_root = Path(__file__).resolve().parents[2] / "src" / "atlas_memory"
    for py_file in src_root.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        assert "sigma2_threshold_provenance" not in text, f"Hot-path leak in {py_file}"


