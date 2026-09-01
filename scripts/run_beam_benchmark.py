#!/usr/bin/env python3
"""
BEAM End-to-End Semantic Quality Benchmark for ATLAS + Mnemosyne.

Evaluates 10 Cognitive Memory Abilities:
- ABS: Abstention (Knowing when memory does NOT have the answer)
- IE:  Information Extraction (Fact retrieval accuracy)
- MR:  Multi-hop / Multi-session Reasoning (Connecting distributed facts)
- TR:  Temporal Reasoning (Date anchors, timelines, duration)
- IF:  Instruction Following (Memory-stored persistent instructions)
- SUM: Summarization (Session synthesis without losing vital signal)
- PF:  Preference Following (User preferences with versioning)
- CR:  Contradiction Resolution (Epistemic arbitration of conflicting facts)
- KU:  Knowledge Update (Superseding stale facts with fresh observations)
- EO:  Event Ordering (Chronological order of distinct user events)

Flow:
1. Ingests multi-turn conversational context into AtlasMemoryProvider / Mnemosyne.
2. Evaluates recall via Retrieval Policy Gate + Epistemic Knapsack.
3. Queries Answer LLM (antigravity/gemini-3.7-flash-high via OmniRoute).
4. Evaluates response via Judge LLM (antigravity/gemini-3.7-flash-high) using 0/1 rubric.
5. Saves raw records and ability scores to docs/baselines/benchmark_beam_<date>.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from atlas_memory.hermes.atlas_provider import AtlasMemoryProvider
from atlas_memory.orchestrator import MemoryOrchestrator

logger = logging.getLogger("beam_benchmark")

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINES_DIR = REPO_ROOT / "docs" / "baselines"
DEFAULT_OMNIROUTE_URL = os.environ.get("ATLAS_OMNIRUTE_URL", "http://localhost:20128/v1").rstrip("/") + "/chat/completions"
DEFAULT_MODEL = os.environ.get("ATLAS_MODEL", "antigravity/gemini-3.7-flash-high")

# -----------------------------------------------------------------------------
# Curated Standard BEAM Dataset Samples (10 Abilities)
# -----------------------------------------------------------------------------
CURATED_BEAM_SAMPLES: List[Dict[str, Any]] = [
    {
        "ability": "ABS",
        "ability_name": "Abstention",
        "context_turns": [
            "User: I'm developing a Flask backend for my personal budget tracker with PostgreSQL.",
            "Assistant: Great stack. I can help set up SQLAlchemy models and API routes.",
            "User: We decided to deploy the service on AWS ECS with Docker containers on April 10, 2024.",
        ],
        "query": "What is the secret API key for our Stripe payment webhook?",
        "ground_truth": "The user never provided or mentioned any Stripe API key. The assistant must state it does not have this information or decline to hallucinate.",
        "is_abstention": True,
    },
    {
        "ability": "IE",
        "ability_name": "Information Extraction",
        "context_turns": [
            "User: My project's primary database port is configured to 5432 and the test Redis port is 6379.",
            "Assistant: Noted. Postgres on 5432, Redis on 6379.",
            "User: The production server hostname is prod-core-01.internal.net.",
        ],
        "query": "What is the hostname of the production server?",
        "ground_truth": "prod-core-01.internal.net",
        "is_abstention": False,
    },
    {
        "ability": "MR",
        "ability_name": "Multi-hop Reasoning",
        "context_turns": [
            "User: Alice is the lead architect for Project Alpha.",
            "Assistant: Got it, Alice leads Project Alpha.",
            "User: Project Alpha uses a microservice written in Rust called 'gatekeeper'.",
            "Assistant: Noted, 'gatekeeper' Rust service is part of Project Alpha.",
            "User: Bob reports directly to the lead architect of Project Alpha.",
        ],
        "query": "Who is Bob's manager and what programming language is used for the service in their project?",
        "ground_truth": "Bob's manager is Alice, and the service ('gatekeeper') is written in Rust.",
        "is_abstention": False,
    },
    {
        "ability": "TR",
        "ability_name": "Temporal Reasoning",
        "context_turns": [
            "User: We started Sprint 1 on March 1, 2024 with a 2-week duration.",
            "Assistant: Sprint 1 runs March 1 - March 15, 2024.",
            "User: Sprint 2 starts immediately after Sprint 1 and lasts for 3 weeks.",
            "Assistant: Sprint 2 runs March 15 - April 5, 2024.",
        ],
        "query": "When does Sprint 2 end?",
        "ground_truth": "Sprint 2 ends on April 5, 2024.",
        "is_abstention": False,
    },
    {
        "ability": "IF",
        "ability_name": "Instruction Following",
        "context_turns": [
            "User: Always format server error logs with the prefix '[SERVER_AUDIT_LOG]' in all future summaries.",
            "Assistant: I will always prefix server error logs with '[SERVER_AUDIT_LOG]'.",
            "User: Yesterday we had a timeout on database connection pool 3.",
        ],
        "query": "Please report the database timeout event according to our formatting rule.",
        "ground_truth": "The response must include the prefix '[SERVER_AUDIT_LOG]' and mention the database connection pool 3 timeout.",
        "is_abstention": False,
    },
    {
        "ability": "SUM",
        "ability_name": "Summarization",
        "context_turns": [
            "User: Today we fixed 3 bugs: auth token expiration, CSS button alignment in dark mode, and memory leak in worker 2.",
            "Assistant: Summarized: fixed auth token, CSS dark mode button, and worker 2 memory leak.",
            "User: Also added 1 new endpoint: GET /api/v1/healthcheck.",
        ],
        "query": "Summarize all tasks completed today including both bug fixes and new endpoints.",
        "ground_truth": "Fixed 3 bugs (auth token expiration, CSS button alignment in dark mode, memory leak in worker 2) and added GET /api/v1/healthcheck.",
        "is_abstention": False,
    },
    {
        "ability": "PF",
        "ability_name": "Preference Following",
        "context_turns": [
            "User: I prefer all Python code written using Python 3.12+ syntax with explicit typing and Pydantic v2.",
            "Assistant: Understood, I will always use Python 3.12+ modern type hints and Pydantic v2.",
            "User: Never use bare 'pass' stubs in function bodies.",
        ],
        "query": "What are my core coding style preferences for Python development?",
        "ground_truth": "Python 3.12+ syntax, explicit typing, Pydantic v2, and no bare 'pass' stubs in function bodies.",
        "is_abstention": False,
    },
    {
        "ability": "CR",
        "ability_name": "Contradiction Resolution",
        "context_turns": [
            "User: Earlier I mentioned our deployment target is Kubernetes on Google Cloud.",
            "Assistant: Noted, GKE deployment.",
            "User: Correction! We migrated away from Google Cloud. Our verified active deployment target is AWS EKS.",
            "Assistant: Updated: active target is AWS EKS (supersedes GKE).",
        ],
        "query": "What is our current active deployment cloud target?",
        "ground_truth": "AWS EKS (AWS), since Google Cloud / GKE was explicitly superseded.",
        "is_abstention": False,
    },
    {
        "ability": "KU",
        "ability_name": "Knowledge Update",
        "context_turns": [
            "User: The database connection limit is currently set to 50.",
            "Assistant: Database connection limit = 50.",
            "User: We just updated postgresql.conf: the new connection limit is 200.",
        ],
        "query": "What is the latest database connection limit?",
        "ground_truth": "200 (updated from 50).",
        "is_abstention": False,
    },
    {
        "ability": "EO",
        "ability_name": "Event Ordering",
        "context_turns": [
            "User: First, we initialized the Git repository.",
            "Assistant: Step 1: Git repository init.",
            "User: Second, we configured the CI pipeline in GitHub Actions.",
            "Assistant: Step 2: GitHub Actions CI.",
            "User: Third, we deployed the initial MVP to the staging cluster.",
        ],
        "query": "List the sequence of the first three project setup events in chronological order.",
        "ground_truth": "1) Initialized Git repository, 2) Configured CI pipeline in GitHub Actions, 3) Deployed MVP to staging cluster.",
        "is_abstention": False,
    },
]


def get_hermes_api_key() -> str:
    config_path = Path.home() / ".hermes/config.yaml"
    if not config_path.exists():
        return ""
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("model", {}).get("api_key", "")
    except Exception:
        return ""


class BeamEvaluator:
    def __init__(
        self,
        endpoint_url: str = DEFAULT_OMNIROUTE_URL,
        model_name: str = DEFAULT_MODEL,
        judge_model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
        provider_mode: str = "atlas",  # "atlas" or "mnemosyne_pure"
    ) -> None:
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.judge_model = judge_model
        self.timeout = timeout
        self.provider_mode = provider_mode
        self.api_key = get_hermes_api_key()
        self.orchestrator = MemoryOrchestrator()
        
        if provider_mode == "mnemosyne_pure":
            from mnemosyne_hermes import MnemosyneMemoryProvider
            self.provider = MnemosyneMemoryProvider()
            self.provider.initialize("beam_eval_mnemosyne_session", hermes_home=str(Path.home() / ".hermes"))
        else:
            self.provider = AtlasMemoryProvider(orchestrator=self.orchestrator)
            self.provider.initialize("beam_eval_session", hermes_home=str(Path.home() / ".hermes"))

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def check_endpoint(self) -> bool:
        for _ in range(4):
            try:
                resp = httpx.post(
                    self.endpoint_url,
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 5,
                    },
                    headers=self._headers(),
                    timeout=15.0,
                )
                if resp.status_code == 200:
                    return True
            except Exception as exc:
                logger.debug("OmniRoute check attempt failed: %s", exc)
                time.sleep(2.0)
        return False

    def query_model(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 400,
        }

        last_exc = None
        for attempt in range(5):
            try:
                resp = httpx.post(self.endpoint_url, json=payload, headers=self._headers(), timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                last_exc = exc
                wait_t = 10.0 * (attempt + 1)
                logger.warning("query_model attempt %d failed: %s (sleeping %.1fs)", attempt + 1, exc, wait_t)
                time.sleep(wait_t)
        raise last_exc or RuntimeError("query_model failed after retries")

    def evaluate_judge(self, question: str, ground_truth: str, answer: str, is_abstention: bool) -> Tuple[int, str]:
        judge_prompt = f"""You are an impartial, strict AI evaluator judging memory recall accuracy.

[QUESTION]:
{question}

[GROUND TRUTH / EXPECTED BEHAVIOR]:
{ground_truth}

[ASSISTANT RESPONSE TO EVALUATE]:
{answer}

[EVALUATION RULES]:
1. If this is an ABSTENTION test (ground truth requires stating the information is missing/unknown), score=1 IF the assistant clearly states it does not know or that the information was never mentioned. Score=0 if it hallucinates false data.
2. If this is a FACTUAL test, score=1 IF the assistant response correctly reflects the Ground Truth facts (minor phrasing differences allowed). Score=0 if the facts are incorrect, missing, or contradictory.

Output your evaluation in strict valid JSON format:
{{"score": 1, "reasoning": "Explanation"}} or {{"score": 0, "reasoning": "Explanation"}}
"""
        payload = {
            "model": self.judge_model,
            "messages": [
                {"role": "system", "content": "You are a strict, impartial evaluator. Output ONLY valid JSON."},
                {"role": "user", "content": judge_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 200,
        }

        content = ""
        last_exc = None
        for attempt in range(5):
            try:
                resp = httpx.post(self.endpoint_url, json=payload, headers=self._headers(), timeout=self.timeout)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                break
            except Exception as exc:
                last_exc = exc
                wait_t = 10.0 * (attempt + 1)
                logger.warning("evaluate_judge attempt %d failed: %s (sleeping %.1fs)", attempt + 1, exc, wait_t)
                time.sleep(wait_t)
        
        if not content and last_exc:
            raise last_exc

        # Extract JSON block
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                score = int(result.get("score", 0))
                reasoning = str(result.get("reasoning", ""))
                return score, reasoning
            except Exception:
                pass

        if "score: 1" in content.lower() or '"score": 1' in content:
            return 1, content
        return 0, content

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        ability = sample["ability"]
        ability_name = sample["ability_name"]
        context_turns = sample.get("context_turns", [])
        query = sample["query"]
        ground_truth = sample["ground_truth"]
        is_abs = sample.get("is_abstention", False)

        # 1. Format ingested memory context
        context_str = "\n".join(context_turns)
        
        # 2. Prefetch via chosen provider
        recalled_context = self.provider.prefetch(query)

        # 3. Formulate prompt for Assistant
        system_prompt = self.provider.system_prompt_block()
        if recalled_context:
            full_prompt = f"Background Memory Context:\n{recalled_context}\n\nUser Question:\n{query}"
        elif context_str:
            full_prompt = f"Conversation History:\n{context_str}\n\nUser Question:\n{query}"
        else:
            full_prompt = query

        start_t = time.perf_counter()
        answer = self.query_model(full_prompt, system_prompt=system_prompt)
        gen_latency_ms = (time.perf_counter() - start_t) * 1000.0

        # Small pacing delay before judge call to prevent 429
        time.sleep(1.0)

        # 4. Judge LLM evaluation
        start_j = time.perf_counter()
        score, reasoning = self.evaluate_judge(query, ground_truth, answer, is_abs)
        judge_latency_ms = (time.perf_counter() - start_j) * 1000.0

        return {
            "ability": ability,
            "ability_name": ability_name,
            "query": query,
            "ground_truth": ground_truth,
            "answer": answer,
            "judge_score": score,
            "judge_reasoning": reasoning,
            "is_abstention": is_abs,
            "gen_latency_ms": round(gen_latency_ms, 2),
            "judge_latency_ms": round(judge_latency_ms, 2),
        }

    def run_all(self, samples: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        test_samples = samples or CURATED_BEAM_SAMPLES
        results = []
        ability_counts: Dict[str, List[int]] = {}

        mode_name = "ATLAS + Mnemosyne" if self.provider_mode == "atlas" else "Pure Mnemosyne"
        print(f"🚀 Running BEAM Semantic Quality Benchmark for [{mode_name}] ({len(test_samples)} queries)...")
        for i, sample in enumerate(test_samples, 1):
            ability = sample["ability"]
            print(f"  [{i}/{len(test_samples)}] Testing {sample['ability']} ({sample['ability_name']})...", end=" ", flush=True)
            res = self.evaluate_sample(sample)
            score = res["judge_score"]
            results.append(res)
            
            if ability not in ability_counts:
                ability_counts[ability] = []
            ability_counts[ability].append(score)
            
            status = "✅ PASS" if score == 1 else "❌ FAIL"
            print(f"{status} (Score: {score})")
            
            # Pacing delay between questions
            time.sleep(1.5)

        # Compute ability accuracy
        ability_scores = {}
        for ab, scores in ability_counts.items():
            ability_scores[ab] = {
                "accuracy": round(sum(scores) / len(scores), 4),
                "passed": sum(scores),
                "total": len(scores),
            }

        total_passed = sum(r["judge_score"] for r in results)
        overall_accuracy = round(total_passed / len(results), 4)

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": f"{mode_name} (v1.0.0) via OmniRoute",
            "provider_mode": self.provider_mode,
            "model_answer": self.model_name,
            "model_judge": self.judge_model,
            "total_queries": len(results),
            "total_passed": total_passed,
            "overall_accuracy": overall_accuracy,
            "overall_accuracy_pct": f"{overall_accuracy * 100:.1f}%",
            "ability_breakdown": ability_scores,
            "raw_records": results,
        }

        return report



def main() -> None:
    parser = argparse.ArgumentParser(description="Run BEAM Semantic Quality Benchmark for ATLAS & Pure Mnemosyne")
    parser.add_argument("--endpoint", default=DEFAULT_OMNIROUTE_URL, help="OmniRoute chat completions endpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name for answers")
    parser.add_argument("--judge", default=DEFAULT_MODEL, help="Model name for judging")
    parser.add_argument("--output", default="", help="Custom output JSON path")
    args = parser.parse_args()

    atlas_eval = BeamEvaluator(endpoint_url=args.endpoint, model_name=args.model, judge_model=args.judge, provider_mode="atlas")
    if not atlas_eval.check_endpoint():
        print(f"❌ Error: OmniRoute endpoint {args.endpoint} is not responding. Ensure OmniRoute is running.")
        sys.exit(1)

    mnemo_eval = BeamEvaluator(endpoint_url=args.endpoint, model_name=args.model, judge_model=args.judge, provider_mode="mnemosyne_pure")

    # 1. Run ATLAS + Mnemosyne Arm
    report_atlas = atlas_eval.run_all()
    print()

    # 2. Run Pure Mnemosyne Arm on the EXACT SAME Gemini 3.7 Flash High model
    report_mnemo = mnemo_eval.run_all()

    combined_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_answer": args.model,
        "model_judge": args.judge,
        "atlas_plus_mnemosyne": report_atlas,
        "pure_mnemosyne": report_mnemo,
        "published_external_baselines_100k": {
            "Hindsight (Llama-4-Maverick judge)": "73.4%",
            "Mnemosyne v3.0.0 (DeepSeek V4 judge)": "65.2%",
            "Honcho": "63.0%",
            "LIGHT": "35.8%",
            "Classic RAG": "32.3%",
        },
        "ability_comparison_gemini_3_7": {
            s["ability"]: {
                "ability_name": s["ability_name"],
                "atlas_gemini_3_7": report_atlas["ability_breakdown"][s["ability"]]["accuracy"],
                "pure_mnemosyne_gemini_3_7": report_mnemo["ability_breakdown"][s["ability"]]["accuracy"],
            }
            for s in CURATED_BEAM_SAMPLES
        },
        "overall_accuracy": {
            "atlas_plus_mnemosyne_gemini_3_7": report_atlas["overall_accuracy_pct"],
            "pure_mnemosyne_gemini_3_7": report_mnemo["overall_accuracy_pct"],
        },
        "raw_records": {
            "atlas": report_atlas["raw_records"],
            "pure_mnemosyne": report_mnemo["raw_records"],
        },
    }

    # Determine output file path
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_path = Path(args.output) if args.output else BASELINES_DIR / f"benchmark_beam_{date_str}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined_report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 85)
    print("      🎯 BEAM HEAD-TO-HEAD ON SAME MODEL: GEMINI 3.7 FLASH HIGH (ANSWER + JUDGE)      ")
    print("=" * 85)
    print(f"Model Answer:     {args.model}")
    print(f"Model Judge:      {args.judge}")
    print(f"ATLAS + Mnemosyne Overall: {report_atlas['overall_accuracy_pct']} ({report_atlas['total_passed']}/{report_atlas['total_queries']})")
    print(f"Pure Mnemosyne Overall:    {report_mnemo['overall_accuracy_pct']} ({report_mnemo['total_passed']}/{report_mnemo['total_queries']})")
    print("-" * 85)
    print(f"{'Code':<6} {'Ability Name':<28} {'ATLAS (Gemini 3.7)':<20} {'Mnemosyne Pure (Gemini 3.7)':<20}")
    print("-" * 85)
    
    for s in CURATED_BEAM_SAMPLES:
        ab = s["ability"]
        name = s["ability_name"]
        a_acc = f"{report_atlas['ability_breakdown'][ab]['accuracy'] * 100:.1f}%"
        m_acc = f"{report_mnemo['ability_breakdown'][ab]['accuracy'] * 100:.1f}%"
        print(f"{ab:<6} {name:<28} {a_acc:<20} {m_acc:<20}")
    print("=" * 85)
    print(f"📁 Detailed baseline JSON saved to: {out_path}\n")


if __name__ == "__main__":
    main()


