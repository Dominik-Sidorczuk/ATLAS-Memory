"""Centralized Runtime Configuration for ATLAS (Pydantic v2).

Eliminates ad-hoc, fragmented `os.getenv` calls and conflicting defaults
across daemon, provider, retrieval pipeline, and stores.

Design Principles:
1. Single Source of Truth: All runtime settings are declared in `AtlasConfig`.
2. Strict Typing & Deterministic Parsing: Boolean, integer, path conversion with explicit defaults.
3. Fallback Hierarchy: CLI kwargs > Explicit env vars > Optional .env file > Canonical defaults.
4. Zero-Trust Invariants: Compliant with Pydantic v2, no bare 'pass' stubs.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger("atlas.config")

_CONFIG_INSTANCE: Optional[AtlasConfig] = None


def parse_bool(val: Any, default: bool = False) -> bool:
    """Robust, canonical boolean parser for environment variables and CLI inputs."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "t", "on", "y", "enable", "enabled"):
        return True
    if s in ("0", "false", "no", "f", "off", "n", "disable", "disabled", "none", "stop"):
        return False
    logger.warning("Unrecognized boolean value %r for env var, defaulting to %r", val, default)
    return default


def _load_env_file_dict(path: Path) -> Dict[str, str]:
    """Reads a simple key=value .env file without external dependencies."""
    env_dict: Dict[str, str] = {}
    if not path.is_file():
        return env_dict
    try:
        content = path.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                # Usunięcie komentarzy liniowych poza cudzysłowami (zarówno ' #' jak i '#')
                if not (v.startswith('"') or v.startswith("'")) and "#" in v:
                    v = v.split("#", 1)[0].strip()
                v = v.strip("'\"")
                if k:
                    env_dict[k] = v
    except Exception as exc:
        logger.debug("Failed reading env file %s: %s", path, exc)
    return env_dict


class AtlasConfig(BaseModel):
    """Canonical ATLAS runtime configuration model."""

    ppr_tau_base: float = Field(
        default=40.0,
        description="V151.4: base PPR budget in ms — adaptive budget = max(tau_base, tau_base + gamma*ln(V+E))",
    )
    ppr_gamma: float = Field(
        default=8.0,
        description="V151.4: logarithmic PPR budget growth coefficient per graph size (V+E)",
    )
    ppr_budget_floor_ms: float = Field(
        default=200.0,
        description="V151.2 (user directive 2026-10-09): hard floor for the PPR budget in ms — "
        "budget = max(floor, tau_base + gamma*ln(V+E)). Guarantees full PageRank convergence "
        "on the current graph without degrading to the fallback matcher; scales UP with graph growth.",
    )
    axiom_theta_min: float = Field(
        default=0.05,
        description="V151.5: minimum sim(query_stems, axiom_stems) required to grant quantum-tunneling energy to an axiom",
    )
    knapsack_dedup_sim: float = Field(
        default=0.88,
        description="V151.6: semantic dedup threshold in pack_knapsack_01 for same-subject near-duplicates (<=0 or >=1 disables)",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    # --- System & IPC Paths ---
    socket_path: Path = Field(
        default_factory=lambda: Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")) / "atlas.sock",
        description="Unix Domain Socket path for JSON-RPC 2.0 daemon IPC",
    )
    pid_path: Path = Field(
        default_factory=lambda: Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")) / "atlas.pid",
        description="Path to daemon PID lock file",
    )
    hermes_home: Path = Field(
        default_factory=lambda: Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")),
        description="Root directory for Hermes data and profiles",
    )
    hermes_skills_dir: Path = Field(
        default_factory=lambda: Path(os.environ.get("HERMES_SKILLS_DIR") or (Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")) / "skills")),
        description="Directory for distilled Hermes procedural skills",
    )

    # --- Storage & Backends ---
    kuzu_buffer_mb: int = Field(
        default=1024,
        description="Buffer pool size for Kùzu graph database in MB (min 64)",
    )
    vector_backend_config: Path = Field(
        default_factory=lambda: Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")) / "atlas" / "vector_backend.json",
        description="Configuration path for active vector store backend",
    )
    embed_model_dir: Path = Field(
        default_factory=lambda: Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")) / "atlas" / "models" / "poldense68",
        description="Directory containing PolDense-68M ONNX artifacts",
    )
    swallow_ledger_path: Optional[Path] = Field(
        default=None,
        description="Optional file override for evidenced swallowed exceptions audit log",
    )

    # --- Cognitive Retrieval & Fusion ---
    hipporag_enabled: bool = Field(
        default=False,
        description="Whether HippoRAG 2 Personalized PageRank retrieval is active (default OFF per SPEC §8, enabled in production via launcher/create_default)",
    )
    hipporag_rm_enabled: bool = Field(
        default=False,
        description="HippoRAG Robust Matching (exact passage text mapping)",
    )
    ppr_backend: str = Field(
        default="igraph",
        description="Graph traversal engine for PPR (igraph or scipy)",
    )
    fusion_strategy: str = Field(
        default="dbsf",
        description="Multi-channel candidate fusion strategy (dbsf or rrf)",
    )
    enable_calibrated_thresholds: bool = Field(
        default=True,
        description="Whether PAVA isotonic regression calibrated decision gate is active",
    )
    calib_v93_enabled: bool = Field(
        default=False,
        description="V93 multivariate calibration extension flag",
    )

    # --- Active Sensing, Shadow Worker & Synchronization ---
    shadow_rpc_enabled: bool = Field(
        default=True,
        description="Whether Hermes turn hooks forward turns to AtlasDaemon over UDS",
    )
    shadow_worker_enabled: bool = Field(
        default=True,
        description="Whether background LLM shadow worker is started in AtlasDaemon",
    )
    shadow_model: str = Field(
        default="auto/best-fast",
        description="Model identifier for OmniRoute shadow worker extraction",
    )
    omniroute_url: str = Field(
        default="http://localhost:20128/v1",
        description="API endpoint URL for OmniRoute LLM service",
    )
    sync_psk: Optional[str] = Field(
        default=None,
        description="Pre-shared key for CRDT Gossip protocol encryption",
    )
    debug: bool = Field(
        default=False,
        description="Verbose telemetry and cognitive debugging mode",
    )

    @model_validator(mode="after")
    def _sync_paths_with_hermes_home(self) -> AtlasConfig:
        default_base = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
        if self.hermes_home != default_base:
            if self.socket_path == default_base / "atlas.sock":
                self.socket_path = self.hermes_home / "atlas.sock"
            if self.pid_path == default_base / "atlas.pid":
                self.pid_path = self.hermes_home / "atlas.pid"
            if self.vector_backend_config == default_base / "atlas" / "vector_backend.json":
                self.vector_backend_config = self.hermes_home / "atlas" / "vector_backend.json"
            if self.embed_model_dir == default_base / "atlas" / "models" / "poldense68":
                self.embed_model_dir = self.hermes_home / "atlas" / "models" / "poldense68"
            if self.hermes_skills_dir == default_base / "skills":
                self.hermes_skills_dir = self.hermes_home / "skills"
        return self

    @classmethod
    def from_env(
        cls,
        env_dict: Optional[Dict[str, str]] = None,
        env_file: Optional[Union[str, Path]] = None,
        **overrides: Any,
    ) -> AtlasConfig:
        """Constructs AtlasConfig merging defaults, .env file, environment, and explicit overrides."""
        source: Dict[str, str] = {}

        # 1. Base from env file if present
        hermes_home_raw = (env_dict.get("HERMES_HOME") if env_dict else None) or os.environ.get("HERMES_HOME")
        hermes_home_pre = Path(hermes_home_raw) if hermes_home_raw else Path.home() / ".hermes"

        if env_file is not None:
            source.update(_load_env_file_dict(Path(env_file)))
        else:
            default_env_file = hermes_home_pre / "atlas.env"
            if default_env_file.is_file():
                source.update(_load_env_file_dict(default_env_file))

        # 2. Overlay live os.environ
        live_env = dict(os.environ) if env_dict is None else env_dict
        source.update(live_env)

        # 3. Resolve fields with canonical keys and alias fallbacks
        hermes_home_raw = source.get("HERMES_HOME")
        hermes_home = Path(hermes_home_raw) if hermes_home_raw else Path.home() / ".hermes"

        socket_raw = source.get("ATLAS_SOCKET_PATH")
        socket_path = Path(socket_raw) if socket_raw else hermes_home / "atlas.sock"

        pid_raw = source.get("ATLAS_PID_PATH")
        pid_path = Path(pid_raw) if pid_raw else hermes_home / "atlas.pid"

        hermes_skills_raw = source.get("HERMES_SKILLS_DIR")
        hermes_skills = Path(hermes_skills_raw) if hermes_skills_raw else hermes_home / "skills"

        kuzu_raw = source.get("ATLAS_KUZU_BUFFER_MB")
        try:
            kuzu_buffer = max(64, int(kuzu_raw)) if kuzu_raw else 1024
        except ValueError:
            kuzu_buffer = 1024

        vb_raw = source.get("ATLAS_VECTOR_CONFIG")
        vb_config = Path(vb_raw) if vb_raw else hermes_home / "atlas" / "vector_backend.json"

        model_raw = source.get("ATLAS_EMBED_MODEL_DIR")
        embed_dir = Path(model_raw) if model_raw else hermes_home / "atlas" / "models" / "poldense68"

        swallow_raw = source.get("ATLAS_SWALLOW_LEDGER")
        swallow_path = Path(swallow_raw) if swallow_raw else None

        hipporag_on = parse_bool(source.get("ATLAS_HIPPORAG"), default=False)
        hipporag_rm_on = parse_bool(source.get("ATLAS_HIPPORAG_RM"), default=False)
        ppr_engine = (source.get("ATLAS_PPR_BACKEND") or "igraph").strip().lower()
        fusion = (source.get("ATLAS_FUSION_STRATEGY") or "dbsf").strip().lower()
        calib_gate_on = parse_bool(source.get("ATLAS_ENABLE_CALIBRATED_THRESHOLDS"), default=True)
        calib_v93 = parse_bool(source.get("ATLAS_CALIB_V93"), default=False)

        shadow_rpc_on = parse_bool(source.get("ATLAS_SHADOW_RPC"), default=True)
        shadow_worker_on = parse_bool(source.get("ATLAS_SHADOW_WORKER"), default=True)
        shadow_model_str = (source.get("ATLAS_SHADOW_MODEL") or "auto/best-fast").strip()

        # Typo tolerance: ATLAS_OMNIRROUTE_URL or ATLAS_OMNIRUTE_URL
        omni_url = (
            source.get("ATLAS_OMNIRROUTE_URL")
            or source.get("ATLAS_OMNIRUTE_URL")
            or "http://localhost:20128/v1"
        ).strip().rstrip("/")

        sync_key = source.get("ATLAS_SYNC_PSK")
        debug_mode = parse_bool(source.get("ATLAS_DEBUG"), default=False)

        payload: Dict[str, Any] = {
            "socket_path": socket_path,
            "pid_path": pid_path,
            "hermes_home": hermes_home,
            "hermes_skills_dir": hermes_skills,
            "kuzu_buffer_mb": kuzu_buffer,
            "vector_backend_config": vb_config,
            "embed_model_dir": embed_dir,
            "swallow_ledger_path": swallow_path,
            "hipporag_enabled": hipporag_on,
            "hipporag_rm_enabled": hipporag_rm_on,
            "ppr_backend": ppr_engine,
            "fusion_strategy": fusion,
            "enable_calibrated_thresholds": calib_gate_on,
            "calib_v93_enabled": calib_v93,
            "shadow_rpc_enabled": shadow_rpc_on,
            "shadow_worker_enabled": shadow_worker_on,
            "shadow_model": shadow_model_str,
            "omniroute_url": omni_url,
            "sync_psk": sync_key,
            "debug": debug_mode,
        }
        payload.update(overrides)
        return cls(**payload)


_CONFIG_OVERRIDE: Optional[AtlasConfig] = None
_CONFIG_CACHE: Optional[AtlasConfig] = None
_LAST_ENV_SNAPSHOT: Optional[tuple] = None


def get_config() -> AtlasConfig:
    """Returns the singleton AtlasConfig instance, auto-refreshing when ATLAS/HERMES env vars mutate."""
    global _CONFIG_OVERRIDE, _CONFIG_CACHE, _LAST_ENV_SNAPSHOT
    if _CONFIG_OVERRIDE is not None:
        return _CONFIG_OVERRIDE

    env_snapshot = tuple(
        (k, os.environ[k]) for k in sorted(os.environ) if k.startswith(("ATLAS_", "HERMES_"))
    )
    if _CONFIG_CACHE is None or _LAST_ENV_SNAPSHOT != env_snapshot:
        _LAST_ENV_SNAPSHOT = env_snapshot
        _CONFIG_CACHE = AtlasConfig.from_env()
    return _CONFIG_CACHE


def reset_config(new_config: Optional[AtlasConfig] = None) -> AtlasConfig:
    """Sets or clears an explicit configuration override (for unit test isolation)."""
    global _CONFIG_OVERRIDE, _CONFIG_CACHE, _LAST_ENV_SNAPSHOT
    _CONFIG_OVERRIDE = new_config
    if new_config is None:
        _CONFIG_CACHE = None
        _LAST_ENV_SNAPSHOT = None
    return get_config()
