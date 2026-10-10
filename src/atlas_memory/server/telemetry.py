"""Telemetry, Health Diagnostics & Build Stamp Utilities for AtlasDaemon.

Separates storage inspection (`_vector_store_stats`, `_graph_stats`, `build_daemon_stats_payload`)
and version stamping (`_daemon_build_stamp`) from the UDS server transport shell.
"""

from __future__ import annotations

import logging
import os
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

from atlas_memory.l2_semantic.vector_backend import load_vector_backend

logger = logging.getLogger(__name__)

_CACHED_BUILD_STAMP: Optional[Dict[str, Any]] = None


def _vector_store_stats(daemon: Any, include_coverage: bool = False) -> Dict[str, Any]:
    """Stan magazynu wektorowego odczytany z DZIAŁAJĄCEGO store, nie z pliku konfiguracyjnego."""
    vector_store = getattr(daemon.engine, "vector_store", None)
    info: Dict[str, Any] = {
        "collection": getattr(vector_store, "collection_name", None),
        "dimension": getattr(vector_store, "dimension", None),
        "points_count": None,
    }
    client = getattr(vector_store, "client", None)
    if client is not None and info["collection"]:
        try:
            info["points_count"] = int(client.count(collection_name=info["collection"], exact=True).count)
        except Exception as exc:  # pragma: no cover
            info["points_count_error"] = f"{type(exc).__name__}: {exc}"[:120]
    elif vector_store is not None:
        records = getattr(vector_store, "_records", None)
        if records is not None:
            try:
                info["points_count"] = len(records)
            except Exception as exc:  # pragma: no cover
                info["points_count_error"] = f"{type(exc).__name__}: {exc}"[:120]

    if include_coverage and vector_store is not None:
        counter = getattr(vector_store, "count_indexed_kv_points_sync", None) or getattr(
            vector_store, "count_with_upsert_key_prefix_sync", None
        )
        if callable(counter):
            try:
                info["kv_indexed_points"] = int(counter())
            except Exception as exc:  # pragma: no cover
                info["kv_indexed_points_error"] = f"{type(exc).__name__}: {exc}"[:120]
    return info


def _graph_stats(daemon: Any) -> Dict[str, Any]:
    """V72.1 (G2) / V83: liczniki grafu — RAM (NetworkX) oraz dysk (Kùzu)."""
    graph = getattr(daemon, "graph_client", None)
    if graph is None:
        return {
            "ram_nodes": None,
            "ram_edges": None,
            "kuzu_nodes": None,
            "kuzu_edges": None,
            "kuzu_available": False,
        }

    if hasattr(graph, "get_stats_sync"):
        return graph.get_stats_sync()

    stats: Dict[str, Any] = {
        "ram_nodes": None,
        "ram_edges": None,
        "kuzu_nodes": None,
        "kuzu_edges": None,
        "kuzu_available": False,
    }
    nx_graph = getattr(graph, "nx_graph", None)
    if nx_graph is not None:
        try:
            stats["ram_nodes"] = int(nx_graph.number_of_nodes())
            stats["ram_edges"] = int(nx_graph.number_of_edges())
        except Exception as exc:
            logger.debug("graph stats (RAM) niedostępne: %s", exc)

    conn = getattr(graph, "conn", None)
    if conn is None:
        return stats

    lock = getattr(graph, "_graph_lock", None)
    if lock is not None:
        lock.acquire()
    try:
        nodes_res = conn.execute("MATCH (e:Entity) RETURN count(e)")
        k_ent = int(nodes_res.get_next()[0]) if nodes_res.has_next() else 0
        stats["kuzu_nodes"] = k_ent
        try:
            passages_res = conn.execute("MATCH (p:Passage) RETURN count(p)")
            k_pass = int(passages_res.get_next()[0]) if passages_res.has_next() else 0
            stats["kuzu_total_nodes"] = k_ent + k_pass
        except Exception:
            stats["kuzu_total_nodes"] = k_ent

        edges_res = conn.execute("MATCH ()-[r:RELATION]->() RETURN count(r)")
        k_rel = int(edges_res.get_next()[0]) if edges_res.has_next() else 0
        stats["kuzu_edges"] = k_rel
        try:
            contains_res = conn.execute("MATCH ()-[c:CONTAINS]->() RETURN count(c)")
            k_cont = int(contains_res.get_next()[0]) if contains_res.has_next() else 0
            synonyms_res = conn.execute("MATCH ()-[s:SYNONYM]->() RETURN count(s)")
            k_syn = int(synonyms_res.get_next()[0]) if synonyms_res.has_next() else 0
            stats["kuzu_total_edges"] = k_rel + k_cont + k_syn
        except Exception:
            stats["kuzu_total_edges"] = k_rel

        stats["kuzu_available"] = True
    except Exception as exc:
        logger.debug("graph stats (Kùzu COUNT) niedostępne: %s", exc)
        stats["kuzu_nodes"] = None
        stats["kuzu_edges"] = None
        stats["kuzu_available"] = False
    finally:
        if lock is not None:
            lock.release()
    return stats


def _daemon_build_stamp() -> Dict[str, Any]:
    """Stempel budowy daemona do logu startowego (diagnostyka wersji)."""
    global _CACHED_BUILD_STAMP
    if _CACHED_BUILD_STAMP is not None:
        return _CACHED_BUILD_STAMP

    stamp: Dict[str, Any] = {"version": "unknown", "commit": "unknown", "branch": "unknown"}
    try:
        stamp["version"] = metadata.version("atlas-memory")
    except Exception:
        try:
            pyproj = Path(__file__).resolve().parents[3] / "pyproject.toml"
            for line in pyproj.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.startswith("version"):
                    stamp["version"] = s.split("=", 1)[1].strip().strip('"').strip("'")
                    break
        except Exception as pyproj_err:
            logger.debug("Failed reading pyproject version for stamp: %s", pyproj_err)
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[3]),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if out.returncode == 0:
            stamp["commit"] = out.stdout.strip()
        br = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(Path(__file__).resolve().parents[3]),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if br.returncode == 0 and br.stdout.strip():
            stamp["branch"] = br.stdout.strip()
    except Exception as git_err:
        logger.debug("Failed reading git stamp: %s", git_err)
    _CACHED_BUILD_STAMP = stamp
    return stamp


def clamp_rpc_limit(limit_val: Any, default: int = 15, min_val: int = 1, max_val: int = 100) -> int:
    """Clamps RPC limit parameter to [min_val, max_val] safely."""
    try:
        val = int(limit_val)
    except (ValueError, TypeError):
        val = default
    return max(min_val, min(val, max_val))


async def build_daemon_stats_payload(daemon: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Builds the real-time `get_stats` telemetry payload for `AtlasDaemon`."""
    kv_count = 0
    if daemon.engine is not None and hasattr(daemon.engine, "kv") and daemon.engine.kv is not None:
        try:
            if hasattr(daemon.engine.kv, "count_sync"):
                kv_count = daemon.engine.kv.count_sync()
            elif hasattr(daemon.engine.kv, "get_all_sync"):
                kv_count = len(daemon.engine.kv.get_all_sync(paged=True, limit=1000000))
            else:
                kv_count = len(daemon.engine.kv.get_all())
        except Exception:
            kv_count = 0

    trajectories_count = (
        len(daemon.engine.trajectory_buffer)
        if (daemon.engine and hasattr(daemon.engine, "trajectory_buffer") and daemon.engine.trajectory_buffer is not None)
        else 0
    )
    active_trajectories = (
        len(daemon.engine.trajectory_buffer.trajectories)
        if (
            daemon.engine
            and hasattr(daemon.engine, "trajectory_buffer")
            and daemon.engine.trajectory_buffer is not None
            and hasattr(daemon.engine.trajectory_buffer, "trajectories")
        )
        else 0
    )
    ttt_steps = (
        getattr(daemon.engine.ttt, "step_count", 0)
        if (daemon.engine and hasattr(daemon.engine, "ttt") and daemon.engine.ttt is not None)
        else 0
    )
    ttt_energy = (
        getattr(daemon.engine.ttt, "total_energy", 0.0)
        if (daemon.engine and hasattr(daemon.engine, "ttt") and daemon.engine.ttt is not None)
        else 0.0
    )
    logger.debug("[IPC:STATS] get_stats -> kv_records=%d trajectories=%d ttt_steps=%d", kv_count, trajectories_count, ttt_steps)
    graph_stats = (
        await daemon.graph_client.get_stats()
        if daemon.graph_client is not None and hasattr(daemon.graph_client, "get_stats")
        else _graph_stats(daemon)
    )
    from atlas_memory.cognitive.retrieval.ppr import get_hipporag_telemetry

    hippo_telemetry = get_hipporag_telemetry(graph_store=daemon.graph_client)
    sops_count = 0
    if daemon.engine is not None and hasattr(daemon.engine, "kv") and daemon.engine.kv is not None:
        try:
            conn_kv = daemon.engine.kv.ensure_connection()
            sops_count = conn_kv.execute("SELECT COUNT(*) FROM state_variables WHERE key LIKE 'sop:%'").fetchone()[0]
        except Exception:
            pass
    auditor = getattr(daemon.engine, "auditor", None) if daemon.engine is not None else None

    return {
        "status": "ok",
        "pid": os.getpid(),
        "socket": str(daemon.socket_path),
        "serving": daemon._serving,
        "kv_records": kv_count,
        "trajectories_count": trajectories_count,
        "active_trajectories": active_trajectories,
        "ttt_steps": ttt_steps,
        "ttt_energy": ttt_energy,
        "has_graph": daemon.graph_client is not None,
        "graph": graph_stats,
        "has_orchestrator": daemon.orchestrator is not None,
        "vector_backend": load_vector_backend().describe(),
        "vector_store": _vector_store_stats(daemon, include_coverage=bool(params.get("include_vector_coverage"))),
        "hybrid_channels_log": hippo_telemetry["hybrid_channels_log"],
        "seed_hit_ratio": hippo_telemetry["seed_hit_ratio"],
        "ppr_backend": hippo_telemetry["ppr_backend"],
        "hipporag_error": hippo_telemetry["hipporag_error"],
        "dual_node_coverage": hippo_telemetry["dual_node_coverage"],
        "l3_sops_count": sops_count,
        "l3_auditor_active": auditor is not None,
    }
