from __future__ import annotations

import collections
import logging
import time
from typing import Any, Dict, List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


try:
    from peft import LoraConfig, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


class BakedProcedure(BaseModel):
    """
    Skompresowana procedura z mikro-adapterem LoRA (PEFT) wyekstrahowana z trajektorii.
    """
    procedure_id: str
    pattern_signature: str
    action_sequence: List[str]
    frequency: int
    success_rate: float
    lora_config: Dict[str, Any] = Field(default_factory=dict)
    distilled_weights_meta: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class WeightBaker:
    """
    L3: Pamięć Proceduralna & Meta-Uczenie (Weight Baking / PEFT LoRA Distillation).
    
    Identyfikuje powtarzające się skuteczne sekwencje akcji i kompiluje je
    w konfiguracje mikro-adapterów LoRA (PEFT) gotowe do dołączenia do modelu bazowego.
    """

    def __init__(
        self,
        min_occurrence_threshold: int = 3,
        min_success_rate: float = 0.8,
        default_lora_rank: int = 8,
        default_lora_alpha: int = 16,
    ):
        self.min_threshold = min_occurrence_threshold
        self.min_success_rate = min_success_rate
        self.lora_rank = default_lora_rank
        self.lora_alpha = default_lora_alpha
        self.trajectory_log: List[Dict[str, Any]] = []
        self.baked_procedures: Dict[str, BakedProcedure] = {}

    def log_episode_trajectory(
        self,
        action_names: List[str],
        success: bool,
        context_tag: str = "general",
    ) -> None:
        self.trajectory_log.append({
            "actions": action_names,
            "success": success,
            "context_tag": context_tag,
            "timestamp": time.time(),
        })

    def _generate_peft_lora_config(self, signature: str) -> Dict[str, Any]:
        """Generuje konfigurację LoRA przy użyciu biblioteki PEFT."""
        if HAS_PEFT:
            try:
                cfg = LoraConfig(
                    r=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                return cfg.to_dict()
            except Exception as exc:
                logger.debug("Failed to export PEFT LoraConfig to dict, using fallback: %s", exc)


        return {
            "r": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "peft_type": "LORA",
            "signature": signature,
        }

    def bake_procedures(self) -> List[BakedProcedure]:
        """Analizuje logi i destyluje powtarzające się wzorce do mikro-adapterów LoRA."""
        pattern_counts: Dict[str, Dict[str, Any]] = collections.defaultdict(lambda: {"count": 0, "successes": 0, "actions": []})

        for entry in self.trajectory_log:
            actions = entry["actions"]
            if not actions:
                continue
            sig = f"{entry['context_tag']}::" + "->".join(actions)
            pattern_counts[sig]["count"] += 1
            pattern_counts[sig]["actions"] = actions
            if entry["success"]:
                pattern_counts[sig]["successes"] += 1

        newly_baked: List[BakedProcedure] = []

        for sig, stats in pattern_counts.items():
            count = stats["count"]
            successes = stats["successes"]
            rate = successes / max(1, count)

            if count >= self.min_threshold and rate >= self.min_success_rate:
                proc_id = f"proc_{abs(hash(sig)) % 1000000:06d}"
                lora_cfg = self._generate_peft_lora_config(sig)

                baked = BakedProcedure(
                    procedure_id=proc_id,
                    pattern_signature=sig,
                    action_sequence=stats["actions"],
                    frequency=count,
                    success_rate=rate,
                    lora_config=lora_cfg,
                    distilled_weights_meta={
                        "adapter_name": f"lora_proc_{proc_id}",
                        "rank": self.lora_rank,
                        "alpha": self.lora_alpha,
                        "status": "ready_for_weight_baking",
                    },
                )
                self.baked_procedures[proc_id] = baked
                newly_baked.append(baked)

        return newly_baked

    def get_baked_procedure_for_context(self, context_tag: str) -> List[BakedProcedure]:
        return [
            proc for proc in self.baked_procedures.values()
            if proc.pattern_signature.startswith(f"{context_tag}::")
        ]
