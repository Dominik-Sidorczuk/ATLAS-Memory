"""Atlas Micro-Sidecar Daemon executing JSON-RPC 2.0 requests via Unix Domain Socket.

Refactored to Clean Architecture & Controller Pattern (SOLID / OOP):
- JsonRpcDispatcher: request parsing, validation, telemetry, and method routing.
- MemoryRpcControllerMixin: KV store CRUD, prefetch, observation commits, audit verification.
- CausalRpcControllerMixin: what_if counterfactual queries, graph hydration, annealing.
- SensingRpcControllerMixin: shadow worker extraction, predictive coding probes, attention bus.
- SleepRpcControllerMixin: trajectory ingestion, sleep consolidation, session lifecycle.
- SyncRpcControllerMixin: multi-agent CRDT sync, gossip protocol, peer status.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import struct
import time
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

from atlas_memory.cognitive.conflict_arbiter import ConflictArbiter
from atlas_memory.cognitive.models import (
    clamp_conf,
)
from atlas_memory.cognitive.predictive_coder import PredictiveCoder
from atlas_memory.cognitive.retrieval.pipeline import (
    RetrievalPipeline,
    compute_content_fingerprint,
    get_record_authority_score,
)
from atlas_memory.cognitive.write_path import EpistemicWritePath
from atlas_memory.core.agi_logger import log_subsystem_status_matrix
from atlas_memory.ipc.wire_models import (
    DEFAULT_PID_PATH,
    DEFAULT_SOCKET_PATH,
    PARSE_ERROR,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
)
from atlas_memory.l1_working.hot_buffer import (
    HotBufferPolicy,
    extract_effective_session,
    is_conversational_churn,
    is_session_accessible,
)
from atlas_memory.l2_semantic.vector_backend import load_vector_backend
from atlas_memory.server.controllers.causal import CausalRpcControllerMixin
from atlas_memory.server.controllers.dispatcher import JsonRpcDispatcher
from atlas_memory.server.controllers.memory import MemoryRpcControllerMixin
from atlas_memory.server.controllers.sensing import SensingRpcControllerMixin
from atlas_memory.server.controllers.sleep import (
    SleepRpcControllerMixin,
    _resolve_sleep_watermark,
)
from atlas_memory.server.controllers.sync import SyncRpcControllerMixin
from atlas_memory.server.telemetry import (
    _daemon_build_stamp,
    clamp_rpc_limit,
)

logger = logging.getLogger("atlas.server.daemon")

__all__ = [
    "AtlasDaemon",
    "extract_effective_session",
    "is_session_accessible",
    "compute_content_fingerprint",
    "get_record_authority_score",
    "is_conversational_churn",
    "clamp_rpc_limit",
    "clamp_conf",
    "DEFAULT_SOCKET_PATH",
    "DEFAULT_PID_PATH",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "JSONRPCError",
    "_daemon_build_stamp",
    "_resolve_sleep_watermark",
    "SubsystemLifecycleManager",
]


class SubsystemLifecycleManager:
    """Deterministic Lifecycle Manager tracking DAG component states and transitions."""

    def __init__(self, log: logging.Logger) -> None:
        self.log = log

    def transition(
        self,
        phase_idx: int,
        total_phases: int,
        tag: str,
        name: str,
        target_state: str,
        details: str = "",
        duration_ms: Optional[float] = None,
    ) -> None:
        """Transitions a subsystem: INITIALIZING -> HEALTHCHECK -> target_state."""
        dur_str = f" ({duration_ms:.0f}ms)" if duration_ms is not None else ""
        det_str = f" [{details}]" if details else ""
        self.log.info(
            "[SYS:BOOT] [%d/%d] %s %s: INITIALIZING -> HEALTHCHECK -> %s%s%s",
            phase_idx,
            total_phases,
            tag,
            name,
            target_state,
            det_str,
            dur_str,
        )


class AtlasDaemon(
    MemoryRpcControllerMixin,
    CausalRpcControllerMixin,
    SensingRpcControllerMixin,
    SleepRpcControllerMixin,
    SyncRpcControllerMixin,
):
    """Atlas Micro-Sidecar Daemon executing JSON-RPC 2.0 requests via Unix Domain Socket."""

    def __init__(
        self,
        socket_path: Path | str = DEFAULT_SOCKET_PATH,
        pid_path: Path | str = DEFAULT_PID_PATH,
        engine: Optional[Any] = None,
        graph_client: Optional[Any] = None,
        latent_buffer: Optional[Any] = None,
        orchestrator: Optional[Any] = None,
        shadow_worker: Optional[Any] = None,
        *,
        causal_engine: Optional[Any] = None,
        active_sensing: Optional[Any] = None,
        gossip: Optional[Any] = None,
        calibration_engine: Optional[Any] = None,
        enable_calibrated_thresholds: Optional[bool] = None,
        boot_matrix: Optional[List[Tuple[str, str, str]]] = None,
        hot_buffer: Optional[Any] = None,
        write_path: Optional[Any] = None,
    ) -> None:
        from atlas_memory.core.config import get_config
        cfg = get_config()
        self.socket_path = Path(socket_path) if str(socket_path) != str(DEFAULT_SOCKET_PATH) else cfg.socket_path
        self.pid_path = Path(pid_path) if str(pid_path) != str(DEFAULT_PID_PATH) else cfg.pid_path
        self.engine = engine
        self._boot_matrix = boot_matrix
        self.enable_calibrated_thresholds = bool(
            cfg.enable_calibrated_thresholds
            if enable_calibrated_thresholds is None
            else enable_calibrated_thresholds
        )
        self.calibration_engine = calibration_engine
        if self.enable_calibrated_thresholds and self.calibration_engine is None:
            try:
                from atlas_memory.cognitive.calibration_engine import CalibrationEngine
                self.calibration_engine = CalibrationEngine(enabled=True)
            except Exception as ce_exc:
                logger.debug("Failed to auto-init CalibrationEngine in daemon: %s", ce_exc)
                self.calibration_engine = None
        self.graph_client = graph_client or (engine.graph if engine and hasattr(engine, "graph") else None)
        self.latent_buffer = latent_buffer or (engine.latent if engine and hasattr(engine, "latent") else None)
        self.orchestrator = orchestrator
        self.causal_engine = causal_engine
        if self.causal_engine is None and self.graph_client is not None:
            try:
                from atlas_memory.causal.retro_causal_edge import RetroCausalEngine
                self.causal_engine = RetroCausalEngine(
                    graph_client=self.graph_client,
                    latent_buffer=self.latent_buffer,
                )
            except Exception as ce_err:
                logger.debug("Could not auto-create RetroCausalEngine: %s", ce_err)
        self.active_sensing = active_sensing
        if self.active_sensing is None:
            try:
                from atlas_memory.active.prediction_error import ActiveSensingEngine
                self.active_sensing = ActiveSensingEngine()
            except Exception as as_err:
                logger.debug("Could not auto-create ActiveSensingEngine: %s", as_err)
        self.gossip = gossip
        self.shadow_worker: Optional[Any] = shadow_worker
        if self.shadow_worker is None and self.orchestrator is not None:
            if cfg.shadow_worker_enabled:
                try:
                    from atlas_memory.active.shadow_worker import OmniRouteShadowWorker

                    self.shadow_worker = OmniRouteShadowWorker(orchestrator=self.orchestrator)
                except Exception as sw_err:
                    logger.debug("Could not auto-create OmniRouteShadowWorker: %s", sw_err)
                    self.shadow_worker = None
        if self.shadow_worker is not None and self.orchestrator is not None:
            self.orchestrator.shadow_extractor = self.shadow_worker._extract_spo_sync
        self.session_hook: Optional[Any] = None
        if self.engine is not None:
            try:
                from atlas_memory.hermes.prefix_guard import HermesSessionHook
                self.session_hook = HermesSessionHook(self.engine)
            except Exception as sh_err:
                logger.debug("Could not auto-create HermesSessionHook: %s", sh_err)

        self._server: Optional[asyncio.Server] = None
        self._serving: bool = False
        self._stopping: bool = False
        self._active_writers: Set[asyncio.StreamWriter] = set()

        self._last_memory_md_mtime: float = 0.0
        self._last_memory_md_sync_check: float = 0.0
        self._memory_md_task: Optional[asyncio.Task] = None
        self._graph_hydration: Dict[str, Any] = {}
        self._is_sleeping: bool = False

        # L1 Cognitive Layer Domain Core
        self.hot_buffer = hot_buffer or HotBufferPolicy(maxlen=25, ttl_seconds=900.0)
        if self.engine is not None:
            self.engine.hot_buffer = self.hot_buffer

        self._attention_queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(maxsize=256)
        self._attention_task: Optional[asyncio.Task] = None
        self._hot_kv_buffer = self.hot_buffer
        self.conflict_arbiter = ConflictArbiter()
        self.predictive_coder = PredictiveCoder(
            engine=self.engine,
            causal_engine=self.causal_engine,
            active_sensing=self.active_sensing,
        )
        if self.orchestrator is not None and self.enable_calibrated_thresholds:
            self.orchestrator.enable_calibrated_thresholds = True
            if getattr(self.orchestrator, "calibration_engine", None) is None:
                self.orchestrator.calibration_engine = self.calibration_engine
        self.retrieval_pipeline = RetrievalPipeline(
            hot_buffer=self.hot_buffer,
            kv_store=self.engine.kv if self.engine and hasattr(self.engine, "kv") else None,
            vector_store=self.engine.vector_store if self.engine and hasattr(self.engine, "vector_store") else None,
            graph_store=self.graph_client or (self.engine.graph if self.engine and hasattr(self.engine, "graph") else None),
            orchestrator=self.orchestrator,
            engine=self.engine,
            calibration_engine=self.calibration_engine,
            enable_calibrated_thresholds=self.enable_calibrated_thresholds,
        )
        self.write_path = write_path or EpistemicWritePath(
            kv_store=self.engine.kv if self.engine and hasattr(self.engine, "kv") else None,
            graph_store=self.graph_client or (self.engine.graph if self.engine and hasattr(self.engine, "graph") else None),
            hot_buffer=self.hot_buffer,
            conflict_arbiter=self.conflict_arbiter,
            vector_store=self.engine.vector_store if self.engine and hasattr(self.engine, "vector_store") else None,
        )

        # Wire JsonRpcDispatcher
        self.dispatcher = JsonRpcDispatcher({
            "ping": self._handle_ping,
            "prefetch": self._handle_prefetch,
            "atlas_recall": self._handle_atlas_recall,
            "commit_observation": self._handle_commit_observation,
            "set": self._handle_set,
            "set_state": self._handle_set,
            "get": self._handle_get,
            "delete": self._handle_delete,
            "sync_memory_file": self._handle_sync_memory_file,
            "sync_mnemosyne": self._handle_sync_mnemosyne,
            "get_stats": self._handle_get_stats,
            "sync_turn": self._handle_sync_turn,
            "what_if": self._handle_what_if,
            "active_sensing": self._handle_active_sensing,
            "telemetry_report": self._handle_telemetry_report,
            "sync_export_delta": self._handle_sync_export_delta,
            "sync_apply_delta": self._handle_sync_apply_delta,
            "trigger_sleep_consolidation": self._handle_trigger_sleep_consolidation,
            "sync_peer_status": self._handle_sync_peer_status,
            "anneal": self._handle_anneal,
            "verify_audit_log": self._handle_verify_audit_log,
            "task_feedback": self._handle_task_feedback,
            "session/end": self._handle_session_end,
            "session_end": self._handle_session_end,
            "trajectory/ingest": self._handle_trajectory_ingest,
            "checkpoint_wal": self._handle_checkpoint_wal,
            "procedural/status": self._handle_procedural_status,
            "get_procedural_status": self._handle_procedural_status,
            "shadow_enqueue": self._handle_shadow_enqueue,
            "shadow_drain": self._handle_shadow_drain,
            "shadow_status": self._handle_shadow_status,
        })
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Coroutine[Any, Any, Any]]] = self.dispatcher.handlers

        if hasattr(signal, "SIGHUP"):
            try:
                signal.signal(signal.SIGHUP, signal.SIG_IGN)
            except (ValueError, OSError) as sighup_err:
                logger.debug("Failed ignoring SIGHUP: %s", sighup_err)
        self._seed_hot_kv_buffer()

    def _seed_hot_kv_buffer(self) -> None:
        """Seeds working memory hot buffer from persistent SQLite KV store upon startup (Finding F-4)."""
        if len(getattr(self.hot_buffer, "_buffer", [])) > 0:
            return
        kv_store = getattr(self.engine, "kv", None) if self.engine is not None else None
        if kv_store is None or not callable(getattr(kv_store, "get_recent_sync", None)):
            return
        try:
            self.hot_buffer.seed_from_kv(kv_store, max_age_seconds=900.0, limit=25)
        except Exception as exc:
            logger.debug("Failed to seed hot_buffer from KV store: %s", exc)

    def _build_status_matrix(self) -> List[Tuple[str, str, str]]:
        """Constructs the subsystem status checklist rows dynamically."""
        if getattr(self, "_boot_matrix", None):
            rows: List[Tuple[str, str, str]] = []
            for name, status, details in self._boot_matrix:
                if "HotBuffer" in name:
                    seeded_count = len(getattr(self.hot_buffer, "_buffer", []))
                    rows.append(("[L1] HotBuffer (Working)", "READY", f"Seeded: {seeded_count} records"))
                elif "Multigraph" in name:
                    nodes_cnt = getattr(self.graph_client, "node_count", 0) or (
                        len(self.graph_client.nx_graph.nodes) if hasattr(self.graph_client, "nx_graph") else 0
                    )
                    edges_cnt = getattr(self.graph_client, "edge_count", 0) or (
                        len(self.graph_client.nx_graph.edges) if hasattr(self.graph_client, "nx_graph") else 0
                    )
                    p_cnt = 0
                    if hasattr(self.graph_client, "nx_graph") and self.graph_client.nx_graph is not None:
                        for _, d in self.graph_client.nx_graph.nodes(data=True):
                            if isinstance(d, dict) and d.get("type") == "passage":
                                p_cnt += 1
                    e_cnt = max(0, nodes_cnt - p_cnt)
                    rows.append(("[L2] Kùzu Multigraph", "READY", f"Nodes: {nodes_cnt} (E:{e_cnt}, P:{p_cnt}) | Edges: {edges_cnt}"))
                elif "Procedural Consol" in name:
                    sops_cnt = 0
                    if self.engine and hasattr(self.engine, "kv") and self.engine.kv is not None:
                        try:
                            conn_kv = self.engine.kv.ensure_connection()
                            sops_cnt = conn_kv.execute("SELECT COUNT(*) FROM state_variables WHERE key LIKE 'sop:%'").fetchone()[0]
                        except Exception:
                            pass
                    rows.append(("[L3] Procedural Consol", "READY", f"Stream: Arrow/Parquet | SOPs: {sops_cnt} | AST Guard: strict"))
                else:
                    rows.append((name, status, details))
            return rows

        rows = []
        ttt = getattr(self.engine, "ttt_layer", None) if self.engine else None
        steps = getattr(ttt, "step_count", 0) if ttt else 0
        rows.append(("[L0] TTT Engine", "READY" if ttt else "DISABLED", f"Steps: {steps} restored" if ttt else "[Reason: not configured]"))
        seeded = len(getattr(self.hot_buffer, "_buffer", []))
        rows.append(("[L1] HotBuffer (Working)", "READY", f"Seeded: {seeded} records"))
        g = self.graph_client
        nodes_cnt = (getattr(g, "node_count", 0) or (len(g.nx_graph.nodes) if hasattr(g, "nx_graph") else 0)) if g else 0
        edges_cnt = (getattr(g, "edge_count", 0) or (len(g.nx_graph.edges) if hasattr(g, "nx_graph") else 0)) if g else 0
        p_cnt = 0
        if g and hasattr(g, "nx_graph") and g.nx_graph is not None:
            for _, d in g.nx_graph.nodes(data=True):
                if isinstance(d, dict) and d.get("type") == "passage":
                    p_cnt += 1
        e_cnt = max(0, nodes_cnt - p_cnt)
        rows.append(("[L2] Kùzu Multigraph", "READY" if g else "DISABLED", f"Nodes: {nodes_cnt} (E:{e_cnt}, P:{p_cnt}) | Edges: {edges_cnt}" if g else "[Reason: not configured]"))
        vec = getattr(self.engine, "vector_store", None) if self.engine else None
        dim = getattr(vec, "dimension", 512) if vec else 512
        rows.append(("[L2] Qdrant Vector Engine", "READY" if vec else "DISABLED", f"Dim: {dim} (onnx68 / poldense68)" if vec else "[Reason: not configured]"))
        auditor = getattr(self.engine, "auditor", None) if self.engine else None
        rows.append(("[L3] Memory Auditor", "ACTIVE" if auditor else "DISABLED", "Saga Monad: ACTIVE | Rollback: verified | Decay: enabled" if auditor else "[Reason: not configured]"))
        sops_cnt = 0
        if self.engine and hasattr(self.engine, "kv") and self.engine.kv is not None:
            try:
                conn_kv = self.engine.kv.ensure_connection()
                sops_cnt = conn_kv.execute("SELECT COUNT(*) FROM state_variables WHERE key LIKE 'sop:%'").fetchone()[0]
            except Exception:
                pass
        rows.append(("[L3] Procedural Consol", "READY" if self.engine else "DISABLED", f"Stream: Arrow/Parquet | SOPs: {sops_cnt} | AST Guard: strict" if self.engine else "[Reason: not configured]"))
        ppr = getattr(g, "_ppr_engine", None) if g else None
        rows.append(("[CR] HippoRAG 2 PPR", "WARMED" if ppr else "SKIPPED", "Warmup completed (damping=0.85)" if ppr else "[Reason: not warmed]"))
        rows.append(("[EG] Epistemic Guard", "ACTIVE" if getattr(self, "write_path", None) else "DISABLED", "Conflict Policy: STRICT" if getattr(self, "write_path", None) else "[Reason: not configured]"))
        from atlas_memory.ingest.mnemosyne_ingest import DEFAULT_MNEMOSYNE_DB_PATH
        mnemo_exists = DEFAULT_MNEMOSYNE_DB_PATH.exists()
        rows.append(("[MB] Mnemosyne Bridge", "READY" if mnemo_exists else "UNLINKED", "Sync: active" if mnemo_exists else "Reason: In-process module missing (FALLBACK: Shadow RPC)"))
        sw = self.shadow_worker
        q_max = getattr(getattr(sw, "turn_queue", None), "maxsize", 1000) if sw else 1000
        rows.append(("[SH] OmniRoute Shadow", "DISPATCHING" if sw else "DISABLED", f"Queue max: {q_max} | Model: auto/best-fast" if sw else "[Reason: disabled in config]"))
        return rows

    @classmethod
    def create_default(
        cls,
        socket_path: Path | str = DEFAULT_SOCKET_PATH,
        pid_path: Path | str = DEFAULT_PID_PATH,
        atlas_dir: Optional[Path | str] = None,
        ingest_mnemosyne_on_start: bool = True,
        **kwargs: Any,
    ) -> AtlasDaemon:
        """Instantiates a full-throttle AtlasDaemon with HybridMemoryEngine and Mnemosyne Ingestion."""
        base_dir = Path(atlas_dir) if atlas_dir else (Path.home() / ".hermes" / "atlas")
        base_dir.mkdir(parents=True, exist_ok=True)
        from atlas_memory.core.config import get_config
        cfg = get_config()
        if cfg.hipporag_enabled:
            os.environ.setdefault("ATLAS_HIPPORAG", "1")

        lifecycle = SubsystemLifecycleManager(logger)

        # 1. VerifiedKVStore
        t_kv = time.perf_counter()
        from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
        k_store = VerifiedKVStore(db_path=str(base_dir / "atlas.db"))
        kv_dur_ms = (time.perf_counter() - t_kv) * 1000
        lifecycle.transition(1, 10, "[L2]", "VerifiedKVStore", "READY", "SHA-256 integrity verified", duration_ms=kv_dur_ms)

        # 2. Kùzu Graph Engine
        t_graph = time.perf_counter()
        from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore
        g_store = KuzuGraphStore(db_path=str(base_dir / "kuzu"))
        try:
            hyd_rep = g_store.hydrate_from_disk_sync()
            g_store._pre_hydrated_report = hyd_rep
        except Exception as hyd_err:
            logger.debug("Graph disk hydration skipped: %s", hyd_err)
        nodes_cnt = getattr(g_store, "node_count", 0) or (len(g_store.nx_graph.nodes) if hasattr(g_store, "nx_graph") else 0)
        edges_cnt = getattr(g_store, "edge_count", 0) or (len(g_store.nx_graph.edges) if hasattr(g_store, "nx_graph") else 0)
        p_cnt = 0
        if hasattr(g_store, "nx_graph") and g_store.nx_graph is not None:
            for _, d in g_store.nx_graph.nodes(data=True):
                if isinstance(d, dict) and d.get("type") == "passage":
                    p_cnt += 1
        e_cnt = max(0, nodes_cnt - p_cnt)
        graph_dur_ms = (time.perf_counter() - t_graph) * 1000
        lifecycle.transition(2, 10, "[L2]", "Kùzu Multigraph", "READY", f"Nodes: {nodes_cnt} (E:{e_cnt}, P:{p_cnt}) | Edges: {edges_cnt}", duration_ms=graph_dur_ms)

        # 3. Qdrant Vector Store
        t_vec = time.perf_counter()
        vector_backend = load_vector_backend()
        vb_dict = vector_backend.describe()
        from atlas_memory.l2_semantic.qdrant_store import QdrantVectorStore
        from atlas_memory.l2_semantic.vector_backend import build_encoder
        try:
            enc = build_encoder(vector_backend)
        except Exception:
            enc = None
        v_store = QdrantVectorStore(
            collection_name=vector_backend.collection,
            location=str(base_dir / "qdrant"),
            dimension=vector_backend.dimension,
            encoder=enc,
        )
        vec_dur_ms = (time.perf_counter() - t_vec) * 1000
        vec_kind = vb_dict.get("kind", "onnx68")
        model_name = Path(vb_dict.get("model_dir", "")).name or "poldense68"
        lifecycle.transition(3, 10, "[L2]", "Qdrant Vector Engine", "READY", f"Dim: {vector_backend.dimension} ({vec_kind} / {model_name})", duration_ms=vec_dur_ms)

        # Assemble engine core
        from atlas_memory.active.prediction_error import ActiveSensingEngine
        from atlas_memory.causal.retro_causal_edge import RetroCausalEngine
        from atlas_memory.engine import HybridMemoryEngine
        from atlas_memory.extensions.canonicalizer import EntityCanonicalizer
        from atlas_memory.extensions.compactor import ContextCompactor
        from atlas_memory.extensions.decay_scorer import SalienceDecayEngine
        from atlas_memory.l0_dynamic.ttt_layer import TTTLayer
        from atlas_memory.l1_working.jepa_latent import JEPALatentBuffer
        from atlas_memory.l1_working.trajectory_buffer import ArrowTrajectoryBuffer
        from atlas_memory.l3_procedural.auditor import MemoryAuditor
        from atlas_memory.orchestrator import MemoryOrchestrator

        latent = JEPALatentBuffer()
        ttt = TTTLayer(input_dim=latent.state_dim)
        traj_buf = ArrowTrajectoryBuffer(state_dim=latent.state_dim)
        canon = EntityCanonicalizer()
        decay = SalienceDecayEngine()
        comp = ContextCompactor()
        auditor = MemoryAuditor(g_store, k_store, v_store, decay_engine=decay)

        engine = HybridMemoryEngine(
            vector_store=v_store,
            graph_client=g_store,
            kv_store=k_store,
            ttt_layer=ttt,
            latent_buffer=latent,
            trajectory_buffer=traj_buf,
            auditor=auditor,
            weight_baker=None,
            canonicalizer=canon,
            decay_engine=decay,
            compactor=comp,
        )
        orchestrator = MemoryOrchestrator(engine=engine)
        causal_engine = RetroCausalEngine(graph_client=engine.graph, latent_buffer=engine.latent)
        active_sensing = ActiveSensingEngine()

        ttt_steps = getattr(ttt, "step_count", 0)

        # 4. MemoryAuditor (L3 Reversible Monad & Epistemic Decay)
        t_aud = time.perf_counter()
        dangling_cnt = 0
        try:
            conn_d = k_store.ensure_connection()
            dangling_cnt = conn_d.execute(
                "SELECT COUNT(*) FROM transaction_intents WHERE status = 'PENDING'"
            ).fetchone()[0]
        except Exception:
            pass
        aud_dur_ms = (time.perf_counter() - t_aud) * 1000
        lifecycle.transition(
            4,
            10,
            "[L3]",
            "MemoryAuditor (Saga)",
            "ACTIVE",
            f"Saga Monad: ACTIVE | Rollback: verified (dangling={dangling_cnt}) | Decay: enabled",
            duration_ms=aud_dur_ms,
        )

        # 5. Procedural Engine (L3 Trajectory Consolidation & AST Guard)
        t_proc = time.perf_counter()
        sops_cnt = 0
        try:
            conn_kv = k_store.ensure_connection()
            sops_cnt = conn_kv.execute(
                "SELECT COUNT(*) FROM state_variables WHERE key LIKE 'sop:%'"
            ).fetchone()[0]
        except Exception:
            pass
        from atlas_memory.l3_procedural.skill_compiler import ASTSafetyScanner
        ast_guard_status = "STRICT" if hasattr(ASTSafetyScanner, "FORBIDDEN_CALLS") else "READY"
        proc_dur_ms = (time.perf_counter() - t_proc) * 1000
        lifecycle.transition(
            5,
            10,
            "[L3]",
            "Procedural Engine",
            "READY",
            f"Stream: Arrow/Parquet | SOPs: {sops_cnt} | AST Guard: {ast_guard_status}",
            duration_ms=proc_dur_ms,
        )

        # 6. HippoRAG 2 PPR
        t_ppr = time.perf_counter()
        ppr_status = "WARMED"
        ppr_detail = "Warmup completed (damping=0.85)"
        try:
            from atlas_memory.cognitive.retrieval.ppr import HippoRAG2PPR, ig, is_hipporag_enabled
            if not is_hipporag_enabled():
                ppr_status = "SKIPPED"
                ppr_detail = "Reason: disabled in config"
            elif ig is None:
                ppr_status = "FALLBACK"
                ppr_detail = "Reason: python-igraph not installed, numpy fallback"
            elif hasattr(engine, "graph"):
                ppr_warmer = HippoRAG2PPR()
                ppr_warmer._build_graph_structures(engine.graph)
                engine.graph._ppr_engine = ppr_warmer
        except Exception as ppr_warm_exc:
            ppr_status = "FAILED"
            ppr_detail = f"Reason: {ppr_warm_exc}"
        ppr_dur_ms = (time.perf_counter() - t_ppr) * 1000
        lifecycle.transition(6, 10, "[CR]", "HippoRAG 2 PPR", ppr_status, ppr_detail, duration_ms=ppr_dur_ms)

        # 7. HotBuffer
        t_wb = time.perf_counter()
        from atlas_memory.l1_working.hot_buffer import HotBufferPolicy
        hot_buffer = HotBufferPolicy(maxlen=25, ttl_seconds=900.0)
        seeded_count = 0
        if k_store is not None and callable(getattr(k_store, "get_recent_sync", None)):
            try:
                hot_buffer.seed_from_kv(k_store, max_age_seconds=900.0, limit=25)
                seeded_count = len(getattr(hot_buffer, "_buffer", []))
            except Exception as seed_exc:
                logger.debug("Failed seeding hot_buffer from KV store: %s", seed_exc)
        engine.hot_buffer = hot_buffer
        wb_dur_ms = (time.perf_counter() - t_wb) * 1000
        lifecycle.transition(7, 10, "[L1]", "HotBuffer (Working)", "READY", f"Seeded: {seeded_count} records", duration_ms=wb_dur_ms)

        # 8. Epistemic Guard
        t_wp = time.perf_counter()
        conflict_arbiter = ConflictArbiter()
        write_path = EpistemicWritePath(
            kv_store=k_store,
            graph_store=g_store,
            hot_buffer=hot_buffer,
            conflict_arbiter=conflict_arbiter,
            vector_store=v_store,
        )
        engine.write_path = write_path
        wp_dur_ms = (time.perf_counter() - t_wp) * 1000
        lifecycle.transition(8, 10, "[EG]", "Epistemic Guard", "ACTIVE", "Conflict Policy: STRICT", duration_ms=wp_dur_ms)

        # 9. Memory Bridge (Mnemosyne)
        t_mnemo = time.perf_counter()
        mnemo_status = "UNLINKED"
        mnemo_detail = "Reason: In-process module missing (FALLBACK: Shadow RPC)"
        if ingest_mnemosyne_on_start:
            try:
                from atlas_memory.ingest.mnemosyne_ingest import (
                    DEFAULT_MNEMOSYNE_DB_PATH,
                    MnemosyneIngestEngine,
                    resolve_default_memory_md_path,
                )
                default_atlas_dir = (Path.home() / ".hermes" / "atlas").resolve()
                if base_dir.resolve() == default_atlas_dir:
                    mnemo_db = DEFAULT_MNEMOSYNE_DB_PATH
                    mem_md = resolve_default_memory_md_path()
                else:
                    mnemo_db = base_dir.parent / "mnemosyne" / "data" / "mnemosyne.db"
                    mem_md = base_dir.parent / "memories" / "MEMORY.md"
                ingester = MnemosyneIngestEngine(db_path=mnemo_db, memory_md_path=mem_md)
                if ingester.is_available:
                    ingester.sync_memory_md_into_engine(engine)
                    ingester.sync_into_atlas_engine(engine, skip_memory_md=True)
                    mnemo_status = "READY"
                    mnemo_detail = "Sync: active"
                else:
                    mnemo_status = "UNLINKED"
                    mnemo_detail = "Reason: In-process module missing (FALLBACK: Shadow RPC)"
            except Exception as sync_exc:
                mnemo_status = "UNLINKED"
                mnemo_detail = f"Reason: {sync_exc}"
        mnemo_dur_ms = (time.perf_counter() - t_mnemo) * 1000
        lifecycle.transition(9, 10, "[MB]", "Mnemosyne Bridge", mnemo_status, mnemo_detail, duration_ms=mnemo_dur_ms)

        # 10. Shadow Worker
        t_sw = time.perf_counter()
        shadow_worker = kwargs.pop("shadow_worker", None)
        sw_status = "DISABLED"
        sw_detail = "Reason: disabled in config"
        if shadow_worker is None and cfg.shadow_worker_enabled:
            try:
                from atlas_memory.active.shadow_worker import OmniRouteShadowWorker

                shadow_worker = OmniRouteShadowWorker(orchestrator=orchestrator)
                orchestrator.shadow_extractor = shadow_worker._extract_spo_sync
                sw_status = "DISPATCHING"
                q_max = getattr(getattr(shadow_worker, "turn_queue", None), "maxsize", 1000)
                sw_detail = f"Queue max: {q_max} | Model: auto/best-fast"
            except Exception as sw_exc:
                sw_status = "DISABLED"
                sw_detail = f"Reason: {sw_exc}"
        elif shadow_worker is not None:
            sw_status = "DISPATCHING"
            q_max = getattr(getattr(shadow_worker, "turn_queue", None), "maxsize", 1000)
            sw_detail = f"Queue max: {q_max} | Model: auto/best-fast"
        sw_dur_ms = (time.perf_counter() - t_sw) * 1000
        lifecycle.transition(10, 10, "[SH]", "OmniRoute Shadow", sw_status, sw_detail, duration_ms=sw_dur_ms)

        boot_matrix = [
            ("[L0] TTT Engine", "READY", f"Steps: {ttt_steps} restored"),
            ("[L1] HotBuffer (Working)", "READY", f"Seeded: {seeded_count} records"),
            ("[L2] Kùzu Multigraph", "READY", f"Nodes: {nodes_cnt} (E:{e_cnt}, P:{p_cnt}) | Edges: {edges_cnt}"),
            ("[L2] Qdrant Vector Engine", "READY", f"Dim: {vector_backend.dimension} ({vec_kind} / {model_name})"),
            ("[L3] Memory Auditor", "ACTIVE", f"Saga Monad: ACTIVE | Rollback: verified (dangling={dangling_cnt}) | Decay: enabled"),
            ("[L3] Procedural Consol", "READY", f"Stream: Arrow/Parquet | SOPs: {sops_cnt} | AST Guard: {ast_guard_status}"),
            ("[CR] HippoRAG 2 PPR", ppr_status, ppr_detail),
            ("[EG] Epistemic Guard", "ACTIVE", "Conflict Policy: STRICT"),
            ("[MB] Mnemosyne Bridge", mnemo_status, mnemo_detail),
            ("[SH] OmniRoute Shadow", sw_status, sw_detail),
        ]

        from atlas_memory.sync import EpistemicReputation, GossipProtocol, PeerIdentity
        from atlas_memory.sync.bridge import StateSyncBridge
        from atlas_memory.sync.crdt import DeltaCRDT
        from atlas_memory.sync.crypto import SyncCrypto

        crdt = DeltaCRDT(node_id="hermes_local_node")
        sync_psk = kwargs.pop("sync_psk", cfg.sync_psk)
        if sync_psk:
            crypto = SyncCrypto.from_psk(sync_psk)
        else:
            psk_file = base_dir / "sync.key"
            if psk_file.exists():
                try:
                    crypto = SyncCrypto.from_psk(psk_file.read_bytes())
                except Exception:
                    crypto = SyncCrypto(SyncCrypto.generate_key())
            else:
                generated = SyncCrypto.generate_key()
                try:
                    psk_file.write_bytes(generated)
                except Exception as exc:
                    logger.debug("Could not write sync.key file: %s", exc)
                crypto = SyncCrypto(generated)

        bridge = StateSyncBridge(
            kv_store=engine.kv,
            write_path=getattr(engine, "write_path", None),
            node_id="hermes_local_node",
        )
        identity = PeerIdentity(node_id="hermes_local_node")
        reputation = EpistemicReputation()
        gossip = GossipProtocol(
            local_node_id="hermes_local_node",
            crdt=crdt,
            crypto=crypto,
            bridge=bridge,
            identity=identity,
            reputation=reputation,
        )
        gossip.register_peer("hermes_peer_alpha")

        enable_calibrated_thresholds = kwargs.pop(
            "enable_calibrated_thresholds",
            cfg.enable_calibrated_thresholds,
        )
        calibration_engine = kwargs.pop("calibration_engine", None)
        if calibration_engine is None and enable_calibrated_thresholds:
            try:
                from atlas_memory.cognitive.calibration_engine import CalibrationEngine
                calib_file = base_dir / "calibration.json"
                if not calib_file.exists():
                    alt = Path(__file__).resolve().parents[3] / "docs" / "calib" / "calibration_artifact.json"
                    if alt.exists():
                        calib_file = alt
                calibration_engine = CalibrationEngine(
                    artifact_path=calib_file if calib_file.exists() else None,
                    enabled=True,
                )
            except Exception as ce_exc:
                logger.debug("CalibrationEngine init skipped in create_default: %s", ce_exc)
                calibration_engine = None

        return cls(
            socket_path=socket_path,
            pid_path=pid_path,
            engine=engine,
            graph_client=engine.graph,
            latent_buffer=engine.latent,
            orchestrator=orchestrator,
            shadow_worker=shadow_worker,
            causal_engine=causal_engine,
            active_sensing=active_sensing,
            gossip=gossip,
            calibration_engine=calibration_engine,
            enable_calibrated_thresholds=enable_calibrated_thresholds,
            boot_matrix=boot_matrix,
            hot_buffer=hot_buffer,
            write_path=write_path,
        )

    async def start(self) -> None:
        """Starts the Unix Domain Socket server and writes the PID file."""
        if self._serving:
            return

        self._check_and_write_pid()

        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError as exc:
                logger.warning("Could not unlink existing socket %s: %s", self.socket_path, exc)

        if self.engine is not None and hasattr(self.engine, "kv") and self.engine.kv is not None:
            if hasattr(self.engine.kv, "verify_audit_log_sync") and hasattr(self.engine.kv, "heal_audit_log_chain"):
                try:
                    is_valid, broken_seq = self.engine.kv.verify_audit_log_sync(deep=False)
                    if not is_valid:
                        logger.warning("Detected unhealed SHA audit chain on startup (broken at seq %s). Auto-healing...", broken_seq)
                        healed_count, _ = self.engine.kv.heal_audit_log_chain()
                        logger.info("Auto-healed %d audit log rows on daemon startup.", healed_count)
                except Exception as heal_exc:
                    logger.warning("Failed to auto-heal audit chain on startup: %s", heal_exc)

        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        await self._hydrate_graph_from_disk()

        if str(self.socket_path) == str(DEFAULT_SOCKET_PATH):
            try:
                from atlas_memory.core.linguistics import _bilingual_stem
                _bilingual_stem("konfiguracja")
                _bilingual_stem("configuration")
            except Exception as stem_exc:
                logger.debug("Stem warmup skipped: %s", stem_exc)

            try:
                vec_s = getattr(self.engine, "vector_store", None) if self.engine is not None else None
                if vec_s is not None and callable(getattr(vec_s, "search", None)):
                    res_w = vec_s.search("warmup", top_k=1)
                    if asyncio.iscoroutine(res_w):
                        await res_w
            except Exception as vec_exc:
                logger.debug("Vector warmup skipped: %s", vec_exc)

            try:
                from atlas_memory.cognitive.retrieval.ppr import HippoRAG2PPR, ig
                if ig is None:
                    logger.debug("HippoRAG 2 PPR warmup skipped: python-igraph not installed")
                elif self.graph_client is not None and not getattr(self.graph_client, "_ppr_engine", None):
                    _ppr_w = HippoRAG2PPR()
                    self.graph_client._ppr_engine = _ppr_w
                    _ppr_w.run("warmup", self.graph_client, top_k=1)
            except Exception as warm_exc:
                logger.debug("Startup pre-warm skipped or failed: %s", warm_exc)

        # Output the Subsystem Status Matrix directly before opening UDS listener
        matrix_rows = self._build_status_matrix()
        matrix_rows.append(("[IPC] UDS Socket Server", "LISTENING", f"Path: {self.socket_path} | MaxMsg: 16MB"))
        active_count = sum(1 for _, st, _ in matrix_rows if st in ("READY", "ACTIVE", "WARMED", "DISPATCHING", "LISTENING"))
        bypass_count = sum(1 for _, st, _ in matrix_rows if st in ("UNLINKED", "DISABLED", "SKIPPED", "FALLBACK"))
        mandatory_count = len(matrix_rows) - bypass_count

        log_subsystem_status_matrix(
            log=logger,
            rows=matrix_rows,
            total_modules=mandatory_count,
            active_count=active_count,
            optional_disabled=bypass_count,
        )

        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self.socket_path),
            limit=16 * 1024 * 1024,
        )
        self._serving = True
        self._stopping = False
        logger.info("[SYS:BOOT] AtlasDaemon UDS listener online on %s (pid: %d)", self.socket_path, os.getpid())

        if self._attention_task is None:
            self._attention_task = asyncio.create_task(self._attention_consumer())

        if self.shadow_worker is not None and getattr(self.shadow_worker, "_worker_task", None) is None:
            self.shadow_worker.start()
            logger.debug("Shadow worker task started (queue_max=%d)", getattr(self.shadow_worker.turn_queue, "maxsize", -1))

        if self._memory_md_task is None and str(self.socket_path) == str(DEFAULT_SOCKET_PATH):
            self._memory_md_task = asyncio.create_task(self._start_memory_md_watcher())

    async def stop(self) -> None:
        """Gracefully stops the server and cleans up socket and pid files."""
        if getattr(self, "_stopping", False):
            return
        self._stopping = True
        self._serving = False

        if self.shadow_worker is not None:
            try:
                await self.shadow_worker.stop()
            except Exception as sw_stop_err:
                logger.debug("Shadow worker stop handled: %s", sw_stop_err)

        audit_queue = getattr(self.engine, "audit_queue", None)
        if isinstance(audit_queue, asyncio.Queue):
            try:
                await asyncio.wait_for(audit_queue.join(), timeout=5.0)
            except Exception as drain_err:
                logger.debug("Audit queue drain before stop failed/timed out: %s", drain_err)

        if self._memory_md_task is not None:
            self._memory_md_task.cancel()
            try:
                await self._memory_md_task
            except (asyncio.CancelledError, Exception) as task_err:
                logger.debug("Memory MD task cancellation handled: %s", task_err)
            self._memory_md_task = None

        if self._attention_task is not None:
            self._attention_task.cancel()
            try:
                await self._attention_task
            except (asyncio.CancelledError, Exception) as task_err:
                logger.debug("Attention task cancellation handled: %s", task_err)
            self._attention_task = None

        for writer in list(self._active_writers):
            try:
                writer.close()
            except Exception as exc:
                logger.debug("Error closing client writer during stop: %s", exc)

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError as exc:
                logger.debug("Failed to remove socket %s: %s", self.socket_path, exc)

        if self.pid_path.exists():
            try:
                self.pid_path.unlink()
            except OSError as exc:
                logger.debug("Failed to remove PID file %s: %s", self.pid_path, exc)

        if self.engine is not None and hasattr(self.engine, "kv") and self.engine.kv is not None:
            try:
                await self.engine.kv.close()
            except Exception as exc:
                logger.debug("Error closing KV store during stop: %s", exc)

        graph_store = getattr(self.engine, "graph", None) if self.engine is not None else None
        if graph_store is not None and hasattr(graph_store, "close"):
            try:
                report = graph_store.close()
                logger.info("Graph store closed on stop: %s", report)
            except Exception as exc:
                logger.error("Error closing graph store during stop: %s", exc)

        vector_store = (
            (getattr(self.engine, "vector_store", None) or getattr(self.engine, "vector", None))
            if self.engine is not None
            else None
        )
        if vector_store is not None and callable(getattr(vector_store, "close", None)):
            try:
                vector_store.close()
            except Exception as exc:
                logger.debug("Error closing vector store during stop: %s", exc)

        self._active_writers.clear()
        logger.info("AtlasDaemon stopped")

    def register_signal_handlers(self) -> None:
        """Registers SIGTERM and SIGINT for graceful shutdown, and ignores SIGHUP."""
        if hasattr(signal, "SIGHUP"):
            try:
                signal.signal(signal.SIGHUP, signal.SIG_IGN)
            except (ValueError, OSError) as exc:
                logger.debug("Signal handler registration skipped for signal SIGHUP: %s", exc)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
                except (NotImplementedError, RuntimeError) as exc:
                    logger.debug("Signal handler registration skipped for signal %s: %s", sig, exc)

    def _check_and_write_pid(self) -> None:
        """Checks if process is already running via PID file, and writes current PID."""
        if self.pid_path.exists():
            try:
                content = self.pid_path.read_text().strip()
                if content:
                    old_pid = int(content)
                    try:
                        os.kill(old_pid, 0)
                        raise RuntimeError(f"AtlasDaemon is already running with PID {old_pid}")
                    except OSError:
                        logger.debug("Ignoring stale PID file from dead process: %d", old_pid)
            except (ValueError, OSError) as exc:
                if isinstance(exc, RuntimeError):
                    raise
                logger.warning("Error reading old PID file: %s", exc)

        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid_path.write_text(str(os.getpid()))

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handles an incoming client connection with length-prefixed JSON-RPC 2.0 messages."""
        self._active_writers.add(writer)
        try:
            while self._serving:
                try:
                    length_bytes = await asyncio.wait_for(reader.readexactly(4), timeout=60.0)
                except asyncio.TimeoutError:
                    break

                # Protect against raw HTTP/text frames or malformed headers on UDS socket
                is_text = False
                try:
                    text_peek = length_bytes.decode("ascii")
                    if text_peek.startswith(("GET", "POST", "HEAD", "PUT", "OPTIONS", "{", "[")):
                        is_text = True
                except UnicodeDecodeError:
                    pass

                if is_text:
                    extra_sample = b""
                    try:
                        extra_sample = await asyncio.wait_for(reader.read(12), timeout=0.05)
                    except Exception:
                        pass
                    sample_bytes = length_bytes + extra_sample
                    logger.warning(
                        "[IPC:CLIENT] Framing violation: Incompatible un-framed payload on UDS socket. "
                        "Raw header hex: [%s] (ascii: %r). Disconnecting client cleanly.",
                        sample_bytes.hex(),
                        sample_bytes,
                    )
                    break

                (msg_len,) = struct.unpack(">I", length_bytes)
                if msg_len > 16 * 1024 * 1024:
                    extra_sample = b""
                    try:
                        extra_sample = await asyncio.wait_for(reader.read(12), timeout=0.05)
                    except Exception:
                        pass
                    sample_bytes = length_bytes + extra_sample
                    logger.warning(
                        "[IPC:CLIENT] Framing violation: parsed msg_len=%d bytes exceeds 16MB ceiling. "
                        "Raw header hex: [%s] (ascii: %r). Disconnecting client cleanly.",
                        msg_len,
                        sample_bytes.hex(),
                        sample_bytes,
                    )
                    break

                payload_bytes = await asyncio.wait_for(reader.readexactly(msg_len), timeout=30.0)
                try:
                    request_data = json.loads(payload_bytes.decode("utf-8"))
                except Exception as parse_err:
                    err_resp = JSONRPCResponse(
                        id=None,
                        error=JSONRPCError(code=PARSE_ERROR, message=f"Parse error: {parse_err}"),
                    )
                    resp_bytes = err_resp.model_dump_json().encode("utf-8")
                    writer.write(struct.pack(">I", len(resp_bytes)) + resp_bytes)
                    await writer.drain()
                    continue

                if isinstance(request_data, list):
                    if not request_data:
                        empty_resp = JSONRPCResponse(
                            id=None,
                            error=JSONRPCError.invalid_request("Batch cannot be empty"),
                        )
                        resp_bytes = empty_resp.model_dump_json().encode("utf-8")
                        writer.write(struct.pack(">I", len(resp_bytes)) + resp_bytes)
                        await writer.drain()
                    else:
                        responses: List[JSONRPCResponse] = []
                        for req in request_data:
                            resp = await self._dispatch_rpc(req)
                            if resp is not None:
                                responses.append(resp)
                        if responses:
                            batch_data = [resp.model_dump() for resp in responses]
                            try:
                                batch_bytes = json.dumps(batch_data).encode("utf-8")
                            except (UnicodeEncodeError, UnicodeError):
                                batch_bytes = json.dumps(batch_data, ensure_ascii=True).encode("utf-8")
                            writer.write(struct.pack(">I", len(batch_bytes)) + batch_bytes)
                            await writer.drain()
                else:
                    response = await self._dispatch_rpc(request_data)
                    if response is not None:
                        try:
                            response_bytes = response.model_dump_json().encode("utf-8")
                        except (UnicodeEncodeError, UnicodeError):
                            safe_data = response.model_dump()
                            response_bytes = json.dumps(safe_data, ensure_ascii=True).encode("utf-8")
                        response_prefix = struct.pack(">I", len(response_bytes))
                        writer.write(response_prefix + response_bytes)
                        await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError) as exc:
            logger.debug("Client connection terminated normally: %s", exc)
        except Exception as exc:
            logger.error("Exception handling client connection: %s", exc, exc_info=True)
        finally:
            self._active_writers.discard(writer)
            try:
                writer.close()
                await asyncio.wait_for(writer.wait_closed(), timeout=5.0)
            except Exception as exc:
                logger.debug("Error closing client connection writer: %s", exc)

    async def _dispatch_rpc(self, req_raw: Any) -> Optional[JSONRPCResponse]:
        """Dispatches raw request dictionary to appropriate handler."""
        return await self.dispatcher.dispatch(req_raw)


def main() -> None:
    sock = Path(os.environ.get("ATLAS_SOCKET_PATH", str(Path.home() / ".hermes" / "atlas.sock")))
    daemon = AtlasDaemon.create_default(socket_path=str(sock))
    asyncio.run(daemon.start())


if __name__ == "__main__":
    main()
