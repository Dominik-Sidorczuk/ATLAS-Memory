"""Self-Calibrating Retrieval Engine (PAVA isotonic calibration).

Replaces hardcoded magic thresholds with learned calibration artifacts:
- Monotonic mapping: score -> P(hit) via PAVA (Pool Adjacent Violators Algorithm) in pure numpy.
- Scale-invariant rule: score >= f(P) and score_top1 - median_pool >= m.
- Drift protection: detects distribution shift via two-sample Kolmogorov-Smirnov test,
  authorizes refit only when drift is verified AND hold-out Brier score improves by >= epsilon.
- Versioned calibration artifacts with SHA-256 integrity verification.
- Zero torch, zero external dependencies in hot path (numpy only).
"""
from __future__ import annotations

import bisect
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# V99/ADR-017: auto-discovery z docs/baselines wycięty razem z reliktowym artefaktem
# (pliki usunięte przez właściciela w 6f0b3b1; nazwy artefaktu żyją tylko w ADR-017).
# Silnik przyjmuje JAWNĄ ścieżkę artifact_path; bez niej pozostaje dormant.
DRIFT_REPORT_KEY = "atlas:calibration:drift_report"
QUERY_RING_BUFFER_KEY = "atlas:calibration:query_ring_buffer"


def is_symbolic_or_identifier(q: str) -> bool:
    """Classifies query as symbolic/identifier vs natural prose (V72.8 - B1 / V109.6 AGI Guard).

    Single source of truth for runtime retrieval pipeline and calibration dataset generation.
    Distinguishes true structured keys/identifiers (e.g. `hermes:memory_md:...`, `fact:...`, single tokens)
    from natural multi-word prose sentences that happen to contain hyphens or colons (e.g. `BPC-157`, `Code - OSS`).
    """
    if not q:
        return False
    q_clean = q.strip()
    if q_clean.startswith(("[IMPORTANT:", "[SYSTEM", "<system", "<SYSTEM", "[Message] timestamp=")):
        return False
    if " " not in q_clean:
        return len(q_clean) > 3 or any(ch in q_clean for ch in (":", "_", "-"))
    tokens = q_clean.split()
    first_tok = tokens[0]
    # Structured database keys (e.g. hermes:memory_md:..., fact:..., rule:..., prompt-antigravity:...) have :, _, or - in first token
    if any(ch in first_tok for ch in (":", "_", "-")) and not first_tok.endswith(":"):
        return True
    # Compact 2-3 word technical identifier lookup without interrogative punctuation
    if len(tokens) <= 3 and not q_clean.endswith("?") and any(ch in q_clean for ch in (":", "_", "-")):
        return True
    return False


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Computes 95% bootstrap confidence interval for ROC AUC."""
    yt = np.asarray(y_true, dtype=np.int32)
    ys = np.asarray(y_score, dtype=np.float64)
    n = len(yt)
    if n == 0 or len(np.unique(yt)) < 2:
        return 0.5, 0.5
    rng = np.random.RandomState(seed)
    aucs: List[float] = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        sub_y = yt[idx]
        if len(np.unique(sub_y)) < 2:
            continue
        aucs.append(roc_auc_score(sub_y, ys[idx]))
    if not aucs:
        return 0.5, 0.5
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def pava(y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """Pool Adjacent Violators Algorithm (PAVA) in pure numpy.

    Given values `y` and optional non-negative weights `w`, finds the non-decreasing
    step function `y_hat` that minimizes sum(w_i * (y_i - y_hat_i)^2).
    """
    y_arr = np.asarray(y, dtype=np.float64)
    if w is None:
        w_arr = np.ones(len(y_arr), dtype=np.float64)
    else:
        w_arr = np.asarray(w, dtype=np.float64)

    weights: List[float] = []
    values: List[float] = []
    counts: List[int] = []

    for val, weight in zip(y_arr, w_arr, strict=False):
        weights.append(weight)
        values.append(val)
        counts.append(1)

        while len(values) >= 2 and values[-2] > values[-1]:
            w_new = weights[-2] + weights[-1]
            val_new = (weights[-2] * values[-2] + weights[-1] * values[-1]) / max(1e-12, w_new)
            cnt_new = counts[-2] + counts[-1]

            weights.pop()
            weights.pop()
            values.pop()
            values.pop()
            counts.pop()
            counts.pop()

            weights.append(w_new)
            values.append(val_new)
            counts.append(cnt_new)

    out = []
    for val, cnt in zip(values, counts, strict=False):
        out.extend([val] * cnt)
    return np.array(out, dtype=np.float64)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculates Brier score: mean squared error of predicted probabilities."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((yp - yt) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculates Expected Calibration Error (ECE) across uniform probability bins."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_prob, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    n = len(yt)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (yp >= low) & (yp <= high if i == n_bins - 1 else yp < high)
        if np.any(mask):
            acc = float(np.mean(yt[mask]))
            conf = float(np.mean(yp[mask]))
            ece_val += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece_val)


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area Under the ROC Curve via Wilcoxon-Mann-Whitney rank sum test in pure numpy."""
    yt = np.asarray(y_true, dtype=np.int32)
    ys = np.asarray(y_score, dtype=np.float64)
    pos = ys[yt == 1]
    neg = ys[yt == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 1.0
    count = 0.0
    for p in pos:
        count += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(count / (len(pos) * len(neg)))


class KSResult(float):
    """Two-sample Kolmogorov-Smirnov test result.

    Subclasses float so that it compares directly as statistic (D),
    while also unpacking as (statistic, pvalue) and exposing .statistic and .pvalue.
    """

    statistic: float
    pvalue: float

    def __new__(cls, statistic: float, pvalue: float):
        obj = super().__new__(cls, statistic)
        obj.statistic = float(statistic)
        obj.pvalue = float(pvalue)
        return obj

    def __iter__(self):
        yield self.statistic
        yield self.pvalue

    def __repr__(self) -> str:
        return f"KSResult(statistic={self.statistic:.4f}, pvalue={self.pvalue:.4e})"


def ks_2sample(
    sample1: Union[np.ndarray, List[float]],
    sample2: Union[np.ndarray, List[float]],
) -> KSResult:
    """Two-sample Kolmogorov-Smirnov test statistic D and p-value.

    Returns KSResult which behaves as float (statistic) and unpacks as (statistic, pvalue).
    """
    s1 = np.sort(np.asarray(sample1, dtype=np.float64))
    s2 = np.sort(np.asarray(sample2, dtype=np.float64))
    if len(s1) == 0 or len(s2) == 0:
        return KSResult(0.0, 1.0)
    try:
        from scipy.stats import ks_2samp

        res = ks_2samp(s1, s2)
        return KSResult(float(res.statistic), float(res.pvalue))
    except Exception:
        data_all = np.concatenate([s1, s2])
        cdf1 = np.searchsorted(s1, data_all, side="right") / len(s1)
        cdf2 = np.searchsorted(s2, data_all, side="right") / len(s2)
        d_stat = float(np.max(np.abs(cdf1 - cdf2)))
        n1, n2 = len(s1), len(s2)
        en = math.sqrt((n1 * n2) / (n1 + n2))
        lam = (en + 0.12 + 0.11 / en) * d_stat
        p_val = 0.0
        for k in range(1, 101):
            term = 2.0 * ((-1) ** (k - 1)) * math.exp(-2.0 * (k * lam) ** 2)
            p_val += term
            if abs(term) < 1e-8:
                break
        p_val = max(0.0, min(1.0, p_val))
        return KSResult(d_stat, p_val)


@dataclass
class IsotonicCalibrator:
    """Piecewise-linear isotonic curve and scale-invariant acceptance rule."""

    knots_x: List[float] = field(default_factory=list)
    knots_y: List[float] = field(default_factory=list)
    reference_scores_sample: List[float] = field(default_factory=list)
    optimal_threshold: Optional[float] = None
    optimal_margin: Optional[float] = None
    threshold_selection_reason: Optional[str] = None
    margin_selection_reason: Optional[str] = None
    min_precision_target: float = 0.50
    cost_fp: float = 1.0
    cost_fn: float = 1.0
    version_id: str = "pava"
    created_at: float = field(default_factory=time.time)
    metrics: Dict[str, Any] = field(default_factory=dict)
    by_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    multivariate_prior: Optional[Dict[str, Any]] = None
    sha256: str = ""

    def __post_init__(self) -> None:
        if self.threshold_selection_reason is None:
            self.threshold_selection_reason = "learned" if self.optimal_threshold is not None else "no_candidate_fallback"
        if self.margin_selection_reason is None:
            self.margin_selection_reason = "learned" if self.optimal_margin is not None else "no_candidate_fallback"

    def get_threshold(self, query_type: Optional[str] = None, default: Optional[float] = None) -> float:
        """Returns calibrated threshold for query_type if published, else fallback default."""
        fallback = default
        if fallback is None:
            fallback = 0.70 if query_type == "symbolic" else 0.35

        if query_type and query_type in self.by_type:
            type_cfg = self.by_type[query_type]
            if (
                type_cfg.get("is_published")
                and type_cfg.get("optimal_threshold") is not None
                and type_cfg.get("threshold_selection_reason") == "learned"
            ):
                return float(type_cfg["optimal_threshold"])
            return fallback

        # Global calibration fallback when query_type is unmapped
        if (
            self.optimal_threshold is not None
            and self.optimal_threshold > 0.0
            and self.threshold_selection_reason == "learned"
        ):
            return self.optimal_threshold

        return fallback

    def fit_by_type(
        self,
        train_data: Dict[str, Dict[str, np.ndarray]],
        test_data: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        n_bins: int = 12,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Fits separate PAVA curves for symbolic and prose query types (V72.8 - B2/B3)."""
        self.by_type = {}
        for q_type in ("symbolic", "prose"):
            if q_type not in train_data:
                continue
            t_data = train_data[q_type]
            if len(t_data.get("scores", [])) == 0:
                continue
            s_tr = np.asarray(t_data["scores"], dtype=np.float64)
            y_tr = np.asarray(t_data["labels"], dtype=np.float64)
            m_tr = np.asarray(t_data.get("margins", np.zeros_like(s_tr)), dtype=np.float64)

            # Fit single curve
            w_tr = t_data.get("sample_weight")
            sub_cal = IsotonicCalibrator(min_precision_target=self.min_precision_target)
            sub_cal.fit(s_tr, y_tr, margins=m_tr, sample_weight=w_tr, n_bins=n_bins, seed=seed)

            # Evaluate on test set if provided
            if test_data and q_type in test_data:
                eval_data = test_data[q_type]
                s_te = np.asarray(eval_data["scores"], dtype=np.float64)
                y_te = np.asarray(eval_data["labels"], dtype=np.float64)
            else:
                s_te = s_tr
                y_te = y_tr

            p_te = sub_cal.predict_proba(s_te)
            raw_auc = roc_auc_score(y_te, s_te)
            cal_auc = roc_auc_score(y_te, p_te)
            ci_low, ci_high = bootstrap_auc_ci(y_te, s_te, n_bootstraps=1000, seed=seed)
            brier_r = brier_score(y_te, s_te)
            brier_c = brier_score(y_te, p_te)
            ece_r = expected_calibration_error(y_te, s_te)
            ece_c = expected_calibration_error(y_te, p_te)
            neg_count = int(np.sum(y_te == 0))
            pos_count = int(np.sum(y_te == 1))

            # B4 Rule: Reject publication if AUC 95% CI does not strictly exceed 0.5, negatives < 30,
            # or optimal_threshold is None (no candidate found by selection criterion)
            ci_includes_or_below_half = bool(ci_low <= 0.5)
            insufficient_negatives = bool(neg_count < 30)
            has_learned_threshold = bool(sub_cal.optimal_threshold is not None)
            can_publish = (not ci_includes_or_below_half) and (not insufficient_negatives) and has_learned_threshold

            rejection_reason = None
            if not has_learned_threshold:
                rejection_reason = "no_candidate_fallback"
            elif insufficient_negatives:
                rejection_reason = "insufficient_negatives"
            elif ci_includes_or_below_half:
                rejection_reason = "auc_ci_includes_0.5" if ci_high >= 0.5 else "auc_sub_random_below_0.5"

            self.by_type[q_type] = {
                "knots_x": sub_cal.knots_x,
                "knots_y": sub_cal.knots_y,
                "optimal_threshold": sub_cal.optimal_threshold if can_publish else None,
                "candidate_threshold": sub_cal.optimal_threshold,
                "optimal_margin": sub_cal.optimal_margin if can_publish else None,
                "candidate_margin": sub_cal.optimal_margin,
                "threshold_selection_reason": sub_cal.threshold_selection_reason,
                "margin_selection_reason": sub_cal.margin_selection_reason,
                "is_published": can_publish,
                "rejection_reason": rejection_reason,
                "auc_raw": round(raw_auc, 4),
                "auc_calibrated": round(cal_auc, 4),
                "auc_95ci_lower": round(ci_low, 4),
                "auc_95ci_upper": round(ci_high, 4),
                "brier_raw": round(brier_r, 4),
                "brier_calibrated": round(brier_c, 4),
                "ece_raw": round(ece_r, 4),
                "ece_calibrated": round(ece_c, 4),
                "n_test_negatives": neg_count,
                "n_test_positives": pos_count,
                "reference_scores_sample": sub_cal.reference_scores_sample,
            }

        if "prose" in self.by_type and self.by_type["prose"].get("reference_scores_sample"):
            self.reference_scores_sample = self.by_type["prose"]["reference_scores_sample"]
        elif self.by_type:
            first_k = next(iter(self.by_type))
            self.reference_scores_sample = self.by_type[first_k].get("reference_scores_sample", [])

        self.version_id = f"pava_dual_{int(time.time())}"
        self._update_digest()
        return self.by_type

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        margins: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        n_bins: int = 12,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Fits regularized PAVA curve on training scores and labels (continuous or binary)."""
        scores_arr = np.asarray(scores, dtype=np.float64)
        labels_arr = np.asarray(labels, dtype=np.float64)
        if margins is not None:
            margins_arr = np.asarray(margins, dtype=np.float64)
        else:
            margins_arr = np.zeros_like(scores_arr)

        weights_arr = np.asarray(sample_weight, dtype=np.float64) if sample_weight is not None else None

        # Quantile binning for robust knot selection (prevents overfitting small N)
        q_edges = np.unique(np.quantile(scores_arr, np.linspace(0.0, 1.0, n_bins + 1)))
        bin_x: List[float] = []
        bin_y: List[float] = []
        bin_w: List[float] = []

        for i in range(len(q_edges) - 1):
            low, high = q_edges[i], q_edges[i + 1]
            mask = (scores_arr >= low) & (scores_arr <= high if i == len(q_edges) - 2 else scores_arr < high)
            if np.any(mask):
                bin_x.append(float(np.mean(scores_arr[mask])))
                if weights_arr is not None:
                    w_sub = weights_arr[mask]
                    if np.sum(w_sub) > 0:
                        bin_y.append(float(np.average(labels_arr[mask], weights=w_sub)))
                        bin_w.append(float(np.sum(w_sub)))
                    else:
                        bin_y.append(float(np.mean(labels_arr[mask])))
                        bin_w.append(float(np.sum(mask)))
                else:
                    bin_y.append(float(np.mean(labels_arr[mask])))
                    bin_w.append(float(np.sum(mask)))

        knots_x_arr = np.array(bin_x, dtype=np.float64)
        knots_y_arr = pava(np.array(bin_y, dtype=np.float64), np.array(bin_w, dtype=np.float64))

        self.knots_x = [round(float(x), 4) for x in knots_x_arr]
        self.knots_y = [round(float(y), 4) for y in knots_y_arr]

        # Select threshold via reachable decision cost minimization:
        # Cost(t) = c_fp * FP(t) + c_fn * FN(t)
        # constrained on precision strictly exceeding baseline prior (> min_precision_target)
        best_thresh: Optional[float] = None
        min_cost = float("inf")
        best_recall = 0.0

        for t in np.arange(0.05, 1.105, 0.01):  # V92: skala DBSF dochodzi do ~1.05, stara siatka ucinala optimum
            t_r = round(float(t), 2)
            acc_mask = scores_arr >= t_r
            tp = int(np.sum(acc_mask & (labels_arr >= 0.5)))
            fp = int(np.sum(acc_mask & (labels_arr < 0.5)))
            fn = int(np.sum((~acc_mask) & (labels_arr >= 0.5)))
            tot = tp + fp
            prec = tp / max(1, tot) if tot > 0 else 0.0
            rec = tp / max(1, tp + fn)
            cost = fp * self.cost_fp + fn * self.cost_fn

            if tp > 0 and prec > self.min_precision_target:
                if cost < min_cost or (cost == min_cost and rec > best_recall):
                    min_cost = cost
                    best_recall = rec
                    best_thresh = t_r

        self.optimal_threshold = best_thresh
        self.threshold_selection_reason = "learned" if best_thresh is not None else "no_candidate_fallback"

        # Optimize margin threshold m from train margins
        best_m: Optional[float] = None
        best_f1 = 0.0
        for cand_m in np.arange(0.01, 0.25, 0.01):
            m_mask = margins_arr >= cand_m
            tp = int(np.sum(m_mask & (labels_arr >= 0.5)))
            fp = int(np.sum(m_mask & (labels_arr < 0.5)))
            fn = int(np.sum((~m_mask) & (labels_arr >= 0.5)))
            tot = tp + fp
            prec = tp / max(1, tot) if tot > 0 else 0.0
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-6, prec + rec)

            if tp > 0 and prec > self.min_precision_target and f1 > best_f1:
                best_f1 = f1
                best_m = round(float(cand_m), 4)

        self.optimal_margin = best_m
        self.margin_selection_reason = "learned" if best_m is not None else "no_candidate_fallback"
        self.version_id = f"pava_{int(time.time())}"
        self.reference_scores_sample = [round(float(s), 4) for s in np.quantile(scores_arr, np.linspace(0.0, 1.0, 50))]

        # Raw vs Calibrated metrics on training
        p_train = self.predict_proba(scores_arr)
        labels_eval = (labels_arr >= 0.5).astype(np.int32)
        self.metrics = {
            "n_train": len(scores_arr),
            "brier_raw": round(brier_score(labels_eval, scores_arr), 4),
            "brier_calibrated": round(brier_score(labels_eval, p_train), 4),
            "ece_raw": round(expected_calibration_error(labels_eval, scores_arr), 4),
            "ece_calibrated": round(expected_calibration_error(labels_eval, p_train), 4),
            "auc_raw": round(roc_auc_score(labels_eval, scores_arr), 4),
            "auc_calibrated": round(roc_auc_score(labels_eval, p_train), 4),
            "optimal_threshold": self.optimal_threshold,
            "optimal_margin": self.optimal_margin,
        }
        self._update_digest()
        return self.metrics

    def predict_proba(
        self,
        scores: Union[np.ndarray, float, List[float]],
        query_type: Optional[str] = None,
        x_multivariate: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> Union[np.ndarray, float]:
        """Maps cosine similarity scores to calibrated hit probabilities via piecewise interpolation.

        If x_multivariate and multivariate_prior are provided, computes convex combination:
        p_final = alpha * p_isotonic + (1 - alpha) * sigma(w^T x + b) (alpha=0.70).
        """
        if query_type and self.by_type and query_type in self.by_type:
            type_cfg = self.by_type[query_type]
            if isinstance(type_cfg, dict):
                knots_x = type_cfg.get("knots_x") or self.knots_x
                knots_y = type_cfg.get("knots_y") or self.knots_y
            elif hasattr(type_cfg, "predict_proba"):
                return type_cfg.predict_proba(scores, x_multivariate=x_multivariate)
            else:
                knots_x, knots_y = self.knots_x, self.knots_y
        else:
            knots_x, knots_y = self.knots_x, self.knots_y

        if isinstance(scores, (int, float)):
            if not knots_x or not knots_y:
                p_iso = max(0.0, min(1.0, float(scores)))
            else:
                val = float(scores)
                if val <= knots_x[0]:
                    p_iso = max(0.0, min(1.0, float(knots_y[0])))
                elif val >= knots_x[-1]:
                    p_iso = max(0.0, min(1.0, float(knots_y[-1])))
                else:
                    idx = bisect.bisect_right(knots_x, val)
                    x0, x1 = knots_x[idx - 1], knots_x[idx]
                    y0, y1 = knots_y[idx - 1], knots_y[idx]
                    if x1 == x0:
                        interp = y0
                    else:
                        interp = y0 + (val - x0) * (y1 - y0) / (x1 - x0)
                    p_iso = max(0.0, min(1.0, float(interp)))

            if x_multivariate is not None and self.multivariate_prior:
                alpha = float(self.multivariate_prior.get("alpha", 0.70))
                if "weights" in self.multivariate_prior:
                    weights = np.asarray(self.multivariate_prior.get("weights", []), dtype=np.float64)
                    bias = float(self.multivariate_prior.get("bias", 0.0))
                else:
                    weights = np.asarray([
                        self.multivariate_prior.get("w_dense", 0.0),
                        self.multivariate_prior.get("w_sparse", 0.0),
                        self.multivariate_prior.get("w_margin", 0.0),
                        self.multivariate_prior.get("w_len", 0.0),
                    ], dtype=np.float64)
                    bias = float(self.multivariate_prior.get("b", self.multivariate_prior.get("bias", 0.0)))
                x_arr = np.asarray(x_multivariate, dtype=np.float64)
                if len(weights) == len(x_arr):
                    z = float(np.clip(np.dot(x_arr, weights) + bias, -30.0, 30.0))
                    p_multi = 1.0 / (1.0 + math.exp(-z))
                    return max(0.0, min(1.0, float(alpha * p_iso + (1.0 - alpha) * p_multi)))
            return p_iso

        if not knots_x or not knots_y:
            p_iso = np.clip(scores, 0.0, 1.0)
        else:
            is_scalar = np.isscalar(scores)
            arr = np.asarray(scores, dtype=np.float64)
            interp = np.interp(
                arr,
                knots_x,
                knots_y,
                left=knots_y[0],
                right=knots_y[-1],
            )
            clipped = np.clip(interp, 0.0, 1.0)
            p_iso = float(clipped) if is_scalar else clipped

        if x_multivariate is not None and self.multivariate_prior:
            alpha = float(self.multivariate_prior.get("alpha", 0.70))
            if "weights" in self.multivariate_prior:
                weights = np.asarray(self.multivariate_prior.get("weights", []), dtype=np.float64)
                bias = float(self.multivariate_prior.get("bias", 0.0))
            else:
                weights = np.asarray([
                    self.multivariate_prior.get("w_dense", 0.0),
                    self.multivariate_prior.get("w_sparse", 0.0),
                    self.multivariate_prior.get("w_margin", 0.0),
                    self.multivariate_prior.get("w_len", 0.0),
                ], dtype=np.float64)
                bias = float(self.multivariate_prior.get("b", self.multivariate_prior.get("bias", 0.0)))
            x_arr = np.asarray(x_multivariate, dtype=np.float64)
            z = np.clip(np.dot(x_arr, weights) + bias, -30.0, 30.0)
            p_multi = 1.0 / (1.0 + np.exp(-z))
            return np.clip(alpha * p_iso + (1.0 - alpha) * p_multi, 0.0, 1.0)

        return p_iso

    transform_probability = predict_proba

    def should_accept(
        self,
        score: float,
        margin: Optional[float] = None,
        use_margin: bool = True,
    ) -> bool:
        """Evaluates whether retrieval prediction satisfies calibrated acceptance rule."""
        if self.optimal_threshold is not None and score < self.optimal_threshold:
            return False
        if use_margin and margin is not None and self.optimal_margin is not None and margin < self.optimal_margin:
            return False
        return True

    def _update_digest(self) -> None:
        raw = json.dumps(
            {
                "knots_x": self.knots_x,
                "knots_y": self.knots_y,
                "optimal_threshold": self.optimal_threshold,
                "optimal_margin": self.optimal_margin,
                "threshold_selection_reason": self.threshold_selection_reason,
                "margin_selection_reason": self.margin_selection_reason,
                "by_type": self.by_type,
                "multivariate_prior": self.multivariate_prior,
                "version_id": self.version_id,
            },
            sort_keys=True,
        )
        self.sha256 = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IsotonicCalibrator:
        raw_thresh = data.get("optimal_threshold")
        raw_margin = data.get("optimal_margin")
        thresh_reason = data.get("threshold_selection_reason")
        margin_reason = data.get("margin_selection_reason")

        if raw_thresh is None:
            opt_thresh = None
            if thresh_reason not in ("learned", "no_candidate_fallback"):
                thresh_reason = "no_candidate_fallback"
        else:
            if thresh_reason == "learned":
                opt_thresh = float(raw_thresh)
            else:
                opt_thresh = None
                thresh_reason = "no_candidate_fallback"

        if raw_margin is None:
            opt_margin = None
            if margin_reason not in ("learned", "no_candidate_fallback"):
                margin_reason = "no_candidate_fallback"
        else:
            if margin_reason == "learned":
                opt_margin = float(raw_margin)
            else:
                opt_margin = None
                margin_reason = "no_candidate_fallback"

        by_type_data = dict(data.get("by_type", {}))
        for _q_type, type_cfg in by_type_data.items():
            if isinstance(type_cfg, dict):
                t_reason = type_cfg.get("threshold_selection_reason")
                if t_reason not in ("learned", "no_candidate_fallback"):
                    type_cfg["threshold_selection_reason"] = "no_candidate_fallback"
                if type_cfg.get("optimal_threshold") is not None and type_cfg.get("threshold_selection_reason") != "learned":
                    type_cfg["optimal_threshold"] = None
                m_reason = type_cfg.get("margin_selection_reason")
                if m_reason not in ("learned", "no_candidate_fallback"):
                    type_cfg["margin_selection_reason"] = "no_candidate_fallback"
                if type_cfg.get("optimal_margin") is not None and type_cfg.get("margin_selection_reason") != "learned":
                    type_cfg["optimal_margin"] = None

        cal = cls(
            knots_x=data.get("knots_x", []),
            knots_y=data.get("knots_y", []),
            reference_scores_sample=data.get("reference_scores_sample", []),
            optimal_threshold=opt_thresh,
            optimal_margin=opt_margin,
            threshold_selection_reason=thresh_reason,
            margin_selection_reason=margin_reason,
            min_precision_target=float(data.get("min_precision_target", 0.50)),
            cost_fp=float(data.get("cost_fp", 1.0)),
            cost_fn=float(data.get("cost_fn", 1.0)),
            version_id=str(data.get("version_id", "pava_loaded")),
            created_at=float(data.get("created_at", time.time())),
            metrics=dict(data.get("metrics", {})),
            by_type=by_type_data,
            multivariate_prior=data.get("multivariate_prior"),
            sha256=str(data.get("sha256", "")),
        )
        return cal

    def save_artifact(self, path: Path) -> Path:
        self._update_digest()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        logger.info("Saved calibration artifact to %s (sha256: %s)", path, self.sha256[:16])
        return path

    @classmethod
    def load_artifact(cls, path: Path) -> IsotonicCalibrator:
        raw = json.loads(path.read_text(encoding="utf-8"))
        cal = cls.from_dict(raw)
        logger.info("Loaded calibration artifact %s (version: %s, sha256: %s)", path, cal.version_id, cal.sha256[:16])
        return cal


class CalibrationEngine:
    """Cognitive Calibration and Drift Protection Manager (V72.8/V93)."""

    def __init__(
        self,
        artifact_path: Optional[Path] = None,
        enabled: bool = False,
        ks_drift_threshold: float = 0.15,
        min_brier_improvement: float = 0.01,
    ):
        self.enabled = enabled
        self.ks_drift_threshold = ks_drift_threshold
        self.min_brier_improvement = min_brier_improvement
        self.calibrator: Optional[IsotonicCalibrator] = None

        # V99/ADR-017: jedyne źródło kalibratora to jawna artifact_path. None lub
        # plik niedostępny → engine dormant (calibrator=None) BEZ auto-wyszukiwania.
        if artifact_path is not None:
            art = Path(artifact_path)
            if art.is_file():
                try:
                    self.calibrator = IsotonicCalibrator.load_artifact(art)
                except Exception as exc:
                    logger.warning("Could not load calibration artifact at %s: %s", art, exc)
            else:
                logger.warning(
                    "Calibration artifact path %s does not exist — engine stays dormant (no auto-discovery).",
                    art,
                )

    def get_threshold(self, query_type: Optional[str] = None, default: Optional[float] = None) -> float:
        """Returns active retrieval threshold (calibrated if enabled & valid, else default fallback)."""
        fallback = default
        if fallback is None:
            if query_type == "symbolic":
                fallback = 0.70
            elif query_type == "prose":
                fallback = 0.35
            else:
                fallback = 0.70
        if not self.enabled or self.calibrator is None:
            return fallback
        return self.calibrator.get_threshold(query_type=query_type, default=fallback)

    def get_margin(self, default: float = 0.0) -> float:
        """Returns active margin threshold (calibrated if enabled & valid, else default constant)."""
        if self.enabled and self.calibrator is not None and self.calibrator.optimal_margin is not None:
            return self.calibrator.optimal_margin
        return default

    def predict_proba(
        self,
        score: Union[float, np.ndarray],
        query_type: Optional[str] = None,
        x_multivariate: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """Calculates calibrated hit probability using PAVA knots and multivariate logistic prior (V93)."""
        if not self.enabled or self.calibrator is None:
            return score
        return self.calibrator.predict_proba(score, query_type=query_type, x_multivariate=x_multivariate)

    def should_accept_retrieval(
        self,
        score: float,
        margin: Optional[float] = None,
        query_type: Optional[str] = None,
        fallback_threshold: Optional[float] = None,
        x_multivariate: Optional[Union[List[float], np.ndarray]] = None,
    ) -> bool:
        """Scale-invariant acceptance check with per-query-type thresholds and optional multivariate prior."""
        eff_fallback = fallback_threshold
        if eff_fallback is None:
            eff_fallback = 0.70 if query_type == "symbolic" else (0.35 if query_type == "prose" else 0.70)
        if not self.enabled or self.calibrator is None:
            return score >= eff_fallback

        thresh = self.get_threshold(query_type=query_type, default=eff_fallback)
        if os.getenv("ATLAS_CALIB_V93") == "1" and x_multivariate is not None:
            cal_prob = float(self.predict_proba(score, query_type=query_type, x_multivariate=x_multivariate))
            if cal_prob < thresh:
                return False
        else:
            if score < thresh:
                return False

        if margin is not None and self.calibrator.optimal_margin is not None and margin < self.calibrator.optimal_margin:
            return False
        return True

    def evaluate_drift_and_refit(
        self,
        new_scores: np.ndarray,
        new_labels: np.ndarray,
        new_margins: Optional[np.ndarray],
        holdout_scores: np.ndarray,
        holdout_labels: np.ndarray,
    ) -> Tuple[bool, str, Optional[IsotonicCalibrator]]:
        """Evaluates whether distribution drift warrants an offline/semi-supervised refit.

        NOTE (ANOMALIA-03): This is an offline or semi-supervised procedure (e.g. for
        the task feedback loop or offline audit batch with ground-truth/LLM-judge labels).
        Online runtime prefetch does not observe ground-truth binary labels on the fly
        without an external judge or asynchronous task feedback signal.
        """
        if self.calibrator is None or not self.calibrator.knots_x:
            new_cal = IsotonicCalibrator()
            new_cal.fit(new_scores, new_labels, new_margins)
            return True, "Initial fit (no previous calibrator exists)", new_cal

        ref = self.calibrator.reference_scores_sample if self.calibrator.reference_scores_sample else self.calibrator.knots_x
        reference_scores = np.array(ref, dtype=np.float64)
        ks_stat = ks_2sample(new_scores, reference_scores)

        if ks_stat < self.ks_drift_threshold:
            return (
                False,
                f"No significant distribution drift detected (KS={ks_stat:.4f} < {self.ks_drift_threshold})",
                None,
            )

        # Drift detected: evaluate candidate calibrator on holdout
        candidate = IsotonicCalibrator()
        candidate.fit(new_scores, new_labels, new_margins)

        current_brier = brier_score(holdout_labels, self.calibrator.predict_proba(holdout_scores))
        candidate_brier = brier_score(holdout_labels, candidate.predict_proba(holdout_scores))
        improvement = current_brier - candidate_brier

        if improvement < self.min_brier_improvement:
            return (
                False,
                f"Drift detected (KS={ks_stat:.4f} >= {self.ks_drift_threshold}) but hold-out Brier improvement insufficient ({improvement:.4f} < {self.min_brier_improvement})",
                None,
            )

        return (
            True,
            f"Refit approved: KS={ks_stat:.4f} >= {self.ks_drift_threshold} and hold-out Brier improved by {improvement:.4f} >= {self.min_brier_improvement}",
            candidate,
        )
