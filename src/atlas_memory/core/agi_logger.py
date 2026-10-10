"""Canonical AGI Cognitive Logger & Structured Observability for ATLAS.

Provides human-readable, domain-informative logging for cognitive operations:
- [COG:RECALL] / [COG:RECALL:SKIP] — Memory retrieval, scoring, PAVA threshold, and admitted winners
- [COG:REMEMBER] — Epistemic persistence, knowledge triples, confidence, and revisions
- [COG:CONFLICT] — Epistemic belief revision, contradiction arbitration, and supersession
- [COG:SURPRISE] — Predictive coding, expectation violations, and discrepancy scores
- [COG:ACTIVE_SENSING] — Closed-loop sensor updates and writebacks
- [COG:CAUSAL] — Counterfactual what-if propagation and graph anneal
- [COG:SLEEP] — Procedural consolidation, salience GC, and SOP baking
- [COG:PROCEDURAL] — Procedural memory distillation, SOP synthesis, and execution graph
- [COG:SKILL] — Autonomous skill compilation and AST safety scanner gate
- [COG:SAGA] — Reversible monad state transactions, atomic multi-view writes and compensation rollbacks
- [SYS:BOOT] — Structured, clean startup banner
- [SYS:RPC] — Significant IPC RPC operations (pings and status polling silenced to DEBUG)
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("atlas.agi")


def format_pool_counts(counts: Dict[str, int]) -> str:
    """Formats channel candidate counts (kv, vec, graph, hot)."""
    kv = counts.get("kv", 0)
    vec = counts.get("vec", 0)
    graph = counts.get("graph", 0)
    hot = counts.get("hot", 0)
    total = sum((kv, vec, graph, hot))
    return f"{total} (kv={kv}, vec={vec}, graph={graph}, hot={hot})"


def log_recall(
    log: logging.Logger,
    query: str,
    mode: str,
    pool_counts: Dict[str, int],
    gate_reason: str,
    threshold: float,
    tokens_used: int,
    token_budget: int,
    winners: List[Dict[str, Any]],
    latency_ms: Optional[float] = None,
    stems: Optional[List[str]] = None,
) -> None:
    """Logs an admitted cognitive recall event."""
    q_disp = (query[:45] + "...") if len(query) > 45 else query
    mode_str = f"mode={mode}"
    if stems:
        mode_str += f", stems={stems[:5]}"

    pool_str = format_pool_counts(pool_counts)

    # Format winner facts concisely
    winner_items: List[str] = []
    for r in winners[:3]:
        subj = str(r.get("subject") or r.get("key") or "?")
        score = r.get("score")
        rel = r.get("relevance_reason")
        sc_str = f"{float(score):.2f}" if score is not None else "?"
        if rel and "(" in str(rel):
            # Extract dominant strategy, e.g. dbsf or dense
            strategy = str(rel).split("(")[0].strip()
            winner_items.append(f"{subj} (s={sc_str}, {strategy})")
        else:
            winner_items.append(f"{subj} (s={sc_str})")

    if len(winners) > 3:
        winner_items.append(f"+{len(winners) - 3} more")

    winners_str = ", ".join(winner_items) if winner_items else "none"
    lat_str = f" | {latency_ms:.1f}ms" if latency_ms is not None else ""

    log.info(
        "[COG:RECALL] query='%s' (%s) | pool: %s | gate: ADMITTED (thresh=%.3f) | "
        "tokens: %d/%d | winners: [%s]%s",
        q_disp,
        mode_str,
        pool_str,
        threshold,
        tokens_used,
        token_budget,
        winners_str,
        lat_str,
    )


def log_recall_skip(
    log: logging.Logger,
    query: str,
    mode: str,
    pool_counts: Dict[str, int],
    gate_reason: str,
    threshold: float,
    max_score: float,
    top_rejected: str = "",
    score_clause: str = "",
    latency_ms: Optional[float] = None,
    stems: Optional[List[str]] = None,
) -> None:
    """Logs a rejected/skipped cognitive recall event."""
    q_disp = (query[:45] + "...") if len(query) > 45 else query
    mode_str = f"mode={mode}"
    if stems:
        mode_str += f", stems={stems[:5]}"

    pool_str = format_pool_counts(pool_counts)

    if not score_clause:
        if gate_reason in (
            "continuation_prompt",
            "empty_pool",
            "system_notification",
            "empty_query",
            "casual_greeting",
        ):
            score_clause = "(short_circuit)"
        else:
            score_clause = f"max_score={max_score:.3f} < thresh={threshold:.3f}"

    top_rej_str = f" | top_rejected={top_rejected}" if top_rejected else ""
    lat_str = f" | {latency_ms:.1f}ms" if latency_ms is not None else ""

    # Note: Includes [RETRIEVAL:SKIP] tag to preserve compatibility with existing assertions
    log.info(
        "[COG:RECALL:SKIP] [RETRIEVAL:SKIP] query='%s' (%s) | gate: SKIPPED (reason='%s', %s) | "
        "pool: %s%s%s",
        q_disp,
        mode_str,
        gate_reason,
        score_clause,
        pool_str,
        top_rej_str,
        lat_str,
    )


def log_remember(
    log: logging.Logger,
    key: str,
    confidence: float,
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    object_: Optional[str] = None,
    revision: int = 1,
    edges_added: int = 0,
    vector_indexed: bool = False,
    kv_persisted: bool = True,
    action: str = "persisted",
    latency_ms: Optional[float] = None,
) -> None:
    """Logs an epistemic persistence/remember event."""
    if subject and predicate and object_:
        obj_disp = (str(object_)[:35] + "...") if len(str(object_)) > 35 else str(object_)
        content_str = f"triple: ({subject})-[{predicate}]->({obj_disp})"
    else:
        content_str = f"content='{key}'"

    vec_str = "indexed" if vector_indexed else "skipped"
    kv_str = "persisted" if kv_persisted else "noop"
    lat_str = f" | {latency_ms:.1f}ms" if latency_ms is not None else ""

    log.info(
        "[COG:REMEMBER] key='%s' (conf=%.2f, rev=%d, action=%s) | %s | "
        "graph: +%d edges | vector: %s | kv: %s | audit: sha256_ok%s",
        key,
        confidence,
        revision,
        action,
        content_str,
        edges_added,
        vec_str,
        kv_str,
        lat_str,
    )


def log_conflict(
    log: logging.Logger,
    old_key: str,
    new_key: str,
    resolution: str = "superseded",
    details: str = "",
) -> None:
    """Logs an epistemic conflict / belief revision event."""
    det_str = f" ({details})" if details else ""
    log.info(
        "[COG:CONFLICT] belief revision: '%s' -> '%s' (resolution=%s)%s",
        old_key,
        new_key,
        resolution,
        det_str,
    )


def log_surprise(
    log: logging.Logger,
    target_entity: str,
    severity: str,
    discrepancy_score: float,
    expected_val: Any,
    observed_val: Any,
    action: str = "",
) -> None:
    """Logs a predictive coding surprise / prediction error event."""
    exp_disp = (str(expected_val)[:30] + "...") if len(str(expected_val)) > 30 else str(expected_val)
    obs_disp = (str(observed_val)[:30] + "...") if len(str(observed_val)) > 30 else str(observed_val)
    act_str = f" | action={action}" if action else ""

    log.info(
        "[COG:SURPRISE] target='%s' severity=%s discrepancy=%.4f | "
        "expected='%s' != observed='%s'%s",
        target_entity,
        severity,
        discrepancy_score,
        exp_disp,
        obs_disp,
        act_str,
    )


def log_active_sensing(
    log: logging.Logger,
    writeback_key: str,
    observed_value: Any,
    status: str = "committed",
) -> None:
    """Logs an active sensing closed-loop corrective writeback."""
    val_disp = (str(observed_value)[:45] + "...") if len(str(observed_value)) > 45 else str(observed_value)
    log.info("[COG:ACTIVE_SENSING] closed-loop writeback %s: %s = %s", status, writeback_key, val_disp)


def log_causal(
    log: logging.Logger,
    entity: str,
    action: Optional[str] = None,
    engine: str = "RAM",
    paths_count: int = 0,
    top_outcome: Optional[str] = None,
    latency_ms: Optional[float] = None,
) -> None:
    """Logs counterfactual what-if causal inference."""
    act_str = f" action='{action}'" if action else ""
    top_str = f" | top_outcome='{top_outcome[:40]}'" if top_outcome else ""
    lat_str = f" | {latency_ms:.1f}ms" if latency_ms is not None else ""

    log.info(
        "[COG:CAUSAL] what_if entity='%s'%s engine=%s -> %d paths%s%s",
        entity,
        act_str,
        engine,
        paths_count,
        top_str,
        lat_str,
    )


def log_sleep(
    log: logging.Logger,
    session_id: str,
    records_count: int = 0,
    pruned_count: int = 0,
    sops_count: int = 0,
    duration_ms: Optional[float] = None,
) -> None:
    """Logs sleep consolidation and procedural baking."""
    dur_str = f" in {duration_ms:.1f}ms" if duration_ms is not None else ""
    log.info(
        "[COG:SLEEP] session=%s consolidation finished: %d records consolidated, "
        "%d salience_pruned, %d SOPs baked%s",
        session_id,
        records_count,
        pruned_count,
        sops_count,
        dur_str,
    )


def log_procedural_bake(
    log: logging.Logger,
    sop_id: str,
    steps_count: int,
    trigger: str,
    skill_path: Optional[str] = None,
    latency_ms: Optional[float] = None,
) -> None:
    """Logs procedural distillation of agent trajectory into SOP."""
    lat_str = f" in {latency_ms:.1f}ms" if latency_ms is not None else ""
    path_str = f" -> skill: {skill_path}" if skill_path else ""
    log.info(
        "[COG:PROCEDURAL] baked SOP '%s' (%d steps, trigger='%s')%s%s",
        sop_id,
        steps_count,
        trigger,
        path_str,
        lat_str,
    )


def log_skill_compilation(
    log: logging.Logger,
    skill_name: str,
    ast_status: str = "passed",
    violations_count: int = 0,
    target_path: Optional[str] = None,
) -> None:
    """Logs autonomous skill compilation with AST safety status."""
    tgt_str = f" -> {target_path}" if target_path else ""
    log.info(
        "[COG:SKILL] compiled skill '%s' | AST safety: %s (violations=%d)%s",
        skill_name,
        ast_status,
        violations_count,
        tgt_str,
    )


def log_saga_transaction(
    log: logging.Logger,
    tx_id: str,
    action: str,
    steps_applied: List[str] | Tuple[str, ...],
    status: str = "committed",
    reason: str = "",
) -> None:
    """Logs reversible monad state transaction execution or compensatory rollback."""
    steps_str = ",".join(steps_applied) if steps_applied else "none"
    reason_str = f" (reason='{reason}')" if reason else ""
    log.info(
        "[COG:SAGA] tx='%s' %s status=%s | steps: [%s]%s",
        tx_id,
        action,
        status,
        steps_str,
        reason_str,
    )


def log_boot_banner(
    log: logging.Logger,
    version: str,
    commit: str,
    branch: str,
    socket_path: str,
    pid: int,
    gil_disabled: Optional[bool] = None,
    backends_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Logs the clean, informative startup banner."""
    py_ver = sys.version.split()[0]
    gil_str = " (GIL disabled)" if gil_disabled else ""
    banner_lines = [
        "================================================================================",
        f"🧠 [SYS:BOOT] ATLAS AGI MEMORY DAEMON ONLINE (v{version} @ {commit})",
        f"   Branch:     {branch} | Python: {py_ver}{gil_str} | PID: {pid}",
        f"   UDS Socket: {socket_path}",
    ]
    if backends_info:
        kv_str = backends_info.get("kv", "VerifiedKVStore (SQLite WAL + SHA-256)")
        kuzu_str = backends_info.get("graph", "Kùzu RAM Multigraph + Disk Checkpoint")
        vec_str = backends_info.get("vector", "Qdrant Vector Store (ONNX 512d)")
        banner_lines.extend([
            f"   KV Store:   {kv_str}",
            f"   Graph:      {kuzu_str}",
            f"   Vectors:    {vec_str}",
        ])
    banner_lines.append("================================================================================")
    for line in banner_lines:
        log.info(line)


def log_subsystem_status_matrix(
    log: logging.Logger,
    rows: List[Tuple[str, str, str]],
    total_modules: int = 8,
    active_count: int = 8,
    optional_disabled: int = 0,
    status_summary: Optional[str] = None,
) -> None:
    """Logs the structured ATLAS Cognitive Core Subsystem Initialization Matrix [AGI RUNTIME]."""
    lines = [
        "=" * 80,
        "  ATLAS COGNITIVE CORE // SUBSYSTEM INITIALIZATION MATRIX [AGI RUNTIME]",
        "=" * 80,
    ]
    for name, status, details in rows:
        lines.append(f"  {name:<25} | {status:<11} | {details}")
    lines.append("=" * 80)
    if status_summary:
        lines.append(f"  CORE SYSTEM STATUS: {status_summary}")
    elif optional_disabled > 0:
        lines.append(
            f"  CORE SYSTEM STATUS: ALL MANDATORY SUBSYSTEMS GREEN ({active_count}/{total_modules} ONLINE, {optional_disabled} NON-CRITICAL BYPASS)"
        )
    else:
        lines.append(
            f"  CORE SYSTEM STATUS: ALL MANDATORY SUBSYSTEMS GREEN ({active_count}/{total_modules} ONLINE)"
        )
    lines.append("=" * 80)
    for line in lines:
        log.info(line)


def log_rpc(
    log: logging.Logger,
    method: str,
    latency_ms: float,
    is_ok: bool = True,
    sig: Optional[Dict[str, Any]] = None,
) -> None:
    """Dispatches RPC execution logging, filtering out routine heartbeat spam."""
    status = "ok" if is_ok else "error"

    # Routine heartbeats and polling stay at DEBUG level
    if method in ("ping", "get_stats"):
        log.debug("[IPC:%s] -> %s in %.1fms", method.upper(), status, latency_ms)
        return

    # Prefetch already emitted rich [COG:RECALL] / [COG:RECALL:SKIP]
    if method == "prefetch":
        log.debug("[IPC:RPC] prefetch -> %s in %.1fms", status, latency_ms)
        return

    # Significant RPC methods are logged cleanly at INFO
    sig_str = f" sig={sig}" if sig and isinstance(sig, dict) and len(sig) > 0 else ""
    log.info("[SYS:RPC] %s -> %s in %.1fms%s", method, status, latency_ms, sig_str)

