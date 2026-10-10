"""Configurable domain rules and pattern extractors for knowledge graph ingest.

Extracted from MnemosyneIngestEngine to ensure the memory engine core remains
free of application-specific and domain-specific hardcoded logic (V83 F1).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DomainRelationExtractor:
    """Configurable relation and pattern extractor for causal and dependency graphs."""

    DEFAULT_DEP_PATTERNS: List[Tuple[Any, ...]] = [
        (r"(?:protocol|protokół)\s+([A-Za-z0-9_-]{3,})", "Protokół", "specifies", 0.90),
        (r"(?:requires|wymaga)\s+([A-Za-z0-9_-]{3,})", "{subject}", "requires", 0.90),
        (r"(?:depends on|zależy od)\s+([A-Za-z0-9_-]{3,})", "{subject}", "depends_on", 0.90),
    ]

    _SPLIT_RE = re.compile(r"[,+;]")

    def __init__(
        self,
        dep_patterns: Optional[List[Tuple[str, str, str]]] = None,
        custom_rules: Optional[List[Callable[[str, str, Callable[[str, str, str, float], None]], None]]] = None,
    ) -> None:
        raw_patterns = list(dep_patterns) if dep_patterns is not None else list(self.DEFAULT_DEP_PATTERNS)
        self.dep_patterns = raw_patterns
        self._compiled_patterns: List[Tuple[re.Pattern, str, str, float]] = [
            (
                re.compile(pat[0], re.IGNORECASE) if isinstance(pat[0], str) else pat[0],
                pat[1],
                pat[2],
                float(pat[3]) if len(pat) >= 4 else 1.0,
            )
            for pat in raw_patterns
        ]
        self.custom_rules = list(custom_rules) if custom_rules is not None else []

    @classmethod
    def from_registry(cls, registry: Optional[Any] = None) -> DomainRelationExtractor:
        """Instantiates extractor using patterns aggregated from DomainProfileRegistry."""
        from atlas_memory.domain.registry import DomainProfileRegistry
        reg = registry or DomainProfileRegistry.get_instance()
        rules = reg.get_all_structural_rules()
        return cls(dep_patterns=rules or cls.DEFAULT_DEP_PATTERNS)

    def extract_and_apply(
        self,
        subject: str,
        content: str,
        add_edge_fn: Callable[[str, str, str, float], None],
        stop_words: Set[str],
    ) -> None:
        """Applies configured dependency patterns and custom rules to content."""
        for regex, parent_template, rel_label, rule_conf in self._compiled_patterns:
            parent_node = subject if parent_template == "{subject}" else parent_template
            for match in regex.finditer(content):
                matched_str = match.group(1).strip()
                if matched_str.lower() in stop_words:
                    continue
                sub_parts = [
                    p.strip() for p in self._SPLIT_RE.split(matched_str)
                    if len(p.strip()) >= 3 and p.strip().lower() not in stop_words
                ]
                if sub_parts:
                    if parent_node.lower() != subject.lower():
                        add_edge_fn(subject, "requires_condition", parent_node, rule_conf)
                    for part in sub_parts:
                        if part.lower() != parent_node.lower():
                            add_edge_fn(parent_node, rel_label, part, rule_conf)
                elif len(matched_str) >= 3 and matched_str.lower() != parent_node.lower():
                    add_edge_fn(parent_node, rel_label, matched_str, rule_conf)

        for rule in self.custom_rules:
            try:
                rule(subject, content, add_edge_fn)
            except Exception as e:
                logger.debug("Custom domain rule failed: %s", e)


DEFAULT_APP_EXTRACTOR = DomainRelationExtractor.from_registry()

