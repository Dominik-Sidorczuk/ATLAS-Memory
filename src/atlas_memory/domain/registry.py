"""ATLAS Dynamic Domain Profile Registry (V85 Architecture).

Decouples all domain-specific knowledge (DevOps, healthcare, legal, finance)
from the core engine into external declarative JSON profiles.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_PROFILES_DIR = Path(__file__).resolve().parent.parent / "resources" / "profiles"


def _get_user_profiles_dir() -> Path:
    from atlas_memory.core.config import get_config
    try:
        hermes_home = get_config().hermes_home
    except Exception:
        hermes_home = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
    return hermes_home / "atlas" / "profiles"


class DomainProfileRegistry:
    """Dynamic domain schema registry and plugin manager for ATLAS."""

    _instance: Optional[DomainProfileRegistry] = None

    def __init__(self, profiles_dirs: Optional[List[Path]] = None) -> None:
        self.profiles_dirs: List[Path] = []
        if profiles_dirs:
            self.profiles_dirs.extend(profiles_dirs)
        if _DEFAULT_PROFILES_DIR.is_dir() and _DEFAULT_PROFILES_DIR not in self.profiles_dirs:
            self.profiles_dirs.append(_DEFAULT_PROFILES_DIR)

        # Also support user profiles directory
        user_profiles_dir = _get_user_profiles_dir()
        if user_profiles_dir.is_dir() and user_profiles_dir not in self.profiles_dirs:
            self.profiles_dirs.append(user_profiles_dir)

        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.reload()

    @classmethod
    def get_instance(cls) -> DomainProfileRegistry:
        if cls._instance is None:
            cls._instance = DomainProfileRegistry()
        return cls._instance

    def reload(self) -> None:
        """Reloads all profiles from configured directories."""
        self.profiles.clear()
        user_profiles_dir = _get_user_profiles_dir()
        if user_profiles_dir.is_dir() and user_profiles_dir not in self.profiles_dirs:
            self.profiles_dirs.append(user_profiles_dir)

        for pdir in self.profiles_dirs:
            if not pdir.is_dir():
                continue
            for f in pdir.glob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        pname = data.get("profile_name", f.stem)
                        self.profiles[pname] = data
                        logger.debug("Loaded domain profile '%s' from %s", pname, f)
                except Exception as exc:
                    logger.warning("Failed to load domain profile %s: %s", f, exc)

    def list_profiles(self) -> List[str]:
        return list(self.profiles.keys())

    def get_profile(self, name: str) -> Optional[Dict[str, Any]]:
        return self.profiles.get(name)

    def get_all_structural_rules(self) -> List[Tuple[str, str, str]]:
        """Returns aggregated (pattern, parent_template, rel_label) across profiles."""
        rules: List[Tuple[str, str, str]] = []
        seen_patterns: Set[str] = set()
        for p in self.profiles.values():
            for r in p.get("structural_rules", []):
                if isinstance(r, dict) and "pattern" in r and "parent_template" in r and "rel_label" in r:
                    pat = r["pattern"]
                    if pat not in seen_patterns:
                        seen_patterns.add(pat)
                        rules.append((pat, r["parent_template"], r["rel_label"]))
        return rules

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Returns list of all canonical entities declared in profiles."""
        entities: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()
        for p in self.profiles.values():
            for ent in p.get("entities", []):
                if isinstance(ent, dict) and "canonical_id" in ent:
                    cid = ent["canonical_id"]
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        entities.append(ent)
        return entities

    def get_all_developer_tools(self) -> Set[str]:
        """Returns set of developer tools configured across profiles."""
        tools: Set[str] = set()
        for p in self.profiles.values():
            tools.update(p.get("developer_tools", []))
        return tools

    def get_all_custom_predicates(self) -> Dict[str, Dict[str, Any]]:
        """Returns dictionary of custom predicates declared across profiles."""
        preds: Dict[str, Dict[str, Any]] = {}
        for p in self.profiles.values():
            custom_preds = p.get("custom_predicates")
            if isinstance(custom_preds, dict):
                preds.update(custom_preds)
        return preds

    def get_all_critical_keywords(self) -> Set[str]:
        """Returns aggregated critical keywords declared across profiles."""
        keywords: Set[str] = set()
        for p in self.profiles.values():
            for kw in p.get("critical_keywords", []):
                keywords.add(str(kw).lower())
        return keywords

