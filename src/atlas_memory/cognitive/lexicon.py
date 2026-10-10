"""ATLAS Cognitive Layer: Linguistic Lexicon & Deterministic Tokenizer (V83 Architecture).

Canonical, single-source-of-truth linguistic dictionaries, Polish morphological
heuristics, control vocabulary, and token extraction functions for epistemic
arbitration and belief revision.
"""
from __future__ import annotations

import logging
import re
from typing import Set

from atlas_memory.core.linguistics import LanguageAssetManager

logger = logging.getLogger(__name__)

# ── Operatory polarności i wyłączności epistemicznej ────────────────────────
EXCLUSIVITY_WORDS: Set[str] = {
    "wyłącznie", "tylko", "jedynie", "sole", "solely", "exclusively", "only",
}

PROHIBITION_WORDS: Set[str] = {
    "nie", "not", "brak", "never", "nigdy", "zabronione", "zabronił", "zabrania",
    "zakaz", "zakazane", "odrzucona", "bypasses", "usunięty", "forbid",
    "forbidden", "prohibited", "disallow", "disallowed", "bez", "zabraniać", "zabroniony",
}

AFFIRMATIVE_WORDS: Set[str] = {
    "uses", "używa", "używać", "używamy", "is", "jest", "requires", "enabled", "active",
    "zawsze", "always", "allow", "allowed", "wykonujemy", "zezwala", "pozwala", "zezwól",
    "stosujemy", "stosuje", "stosować", "dozwolone", "dozwolony", "dozwolona", "wykonaj",
}

CONTROL_WORDS: Set[str] = EXCLUSIVITY_WORDS | PROHIBITION_WORDS | AFFIRMATIVE_WORDS

# ── Polskie i angielskie słowa stopu (ładowane z LanguageAssetManager z ochroną polarności) ───
STOP_WORDS: Set[str] = (
    LanguageAssetManager.get_instance().get_combined_stopwords(("pl", "en"))
    - (EXCLUSIVITY_WORDS | PROHIBITION_WORDS)
)

# ── Słownictwo dziedzinowe i ogólne ──────────────────────────────────────────
GENERIC_WORDS: Set[str] = {
    "narzędzia", "narzędzie", "narzędzi", "tool", "tools", "funkcja", "funkcji",
    "metoda", "metody", "sposób", "sposobu", "kod", "kodu", "rzecz", "rzeczy",
    "rule", "state", "fact", "config", "system", "memory", "atlas", "test",
    "variable", "item", "default", "wartość", "parametr", "opcja", "użytkownika",
    "action", "task", "preference", "sensor", "architecture",
}

def _load_developer_tools() -> Set[str]:
    try:
        from atlas_memory.domain.registry import DomainProfileRegistry
        tools = DomainProfileRegistry.get_instance().get_all_developer_tools()
        if tools:
            return set(tools)
    except Exception as exc:
        logger.debug("Failed to load developer tools from registry: %s", exc)
    return {
        "git", "python", "pytest", "bash", "sh", "zsh", "npm", "cargo",
        "docker", "pip", "pixi", "node", "linux", "sqlite", "redis",
        "postgres", "postgresql", "mysql", "qdrant", "kuzu", "hermes",
        "github", "gitlab", "conda", "vscode", "react", "fastapi",
    }


DEVELOPER_TOOLS: Set[str] = _load_developer_tools()

ACTION_PREFIXES: tuple[str, ...] = (
    "wdraż", "deploy", "instal", "urucham", "commit", "push", "merge",
    "budow", "build", "test", "zapis", "sync", "read", "writ", "zarządz",
    "edycj", "logow", "pobier", "usuw", "dostęp", "access", "stor",
)

ENVIRONMENT_TOKENS: Set[str] = {
    "staging", "prod", "production", "dev", "development", "test", "testing", "local",
}

FUNCTIONAL_SLOTS: Set[str] = {
    "port", "ip", "status", "version", "email", "role", "active", "engine",
    "host", "state", "mode", "protocol", "timeout", "ttl", "temperature",
    "speed", "name", "license", "branch_strategy", "file_ops", "auth",
}

KNOWN_NAMESPACES: Set[str] = {
    "fact", "rule", "temp", "user", "system", "config", "session",
    "agent", "state", "service", "preference", "constraint", "metric",
    "observation", "incident", "probe", "sop", "atlas", "action", "architecture",
}


def polish_stem(word: str) -> str:
    """Stem Polish words by stripping common grammatical suffixes."""
    w = word.lower()
    if w.startswith(("zewnątrz", "zewnętrz")):
        return "zewnątrz"
    if len(w) <= 4:
        return w
    suffixes = (
        "owania", "owanie", "owaniu", "owaniach", "owaniami",
        "alności", "alność", "alnością",
        "ościach", "ościami", "ościom", "nością", "ności", "ność",
        "owemu", "owych", "owymi", "owego", "owej", "owym",
        "iach", "iami", "iom",
        "ach", "ami", "om",
        "cja", "cję", "cji", "cją", "cje", "cjo",
        "ego", "emu", "ych", "ymi", "ym",
        "em", "am", "om", "ie", "ej", "ze",
        "a", "e", "y", "i", "o", "u", "ę", "ą",
    )
    for suf in suffixes:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)]
    return w


def extract_tokens(text: str) -> Set[str]:
    """Extract alphanumeric tokens (length >= 3) excluding stop words."""
    return {
        t for t in re.findall(r'[a-zA-Z0-9ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]{3,}', text.lower())
        if t not in STOP_WORDS
    }


def extract_stems(tokens: Set[str]) -> Set[str]:
    """Extract Polish stems from token set."""
    return {polish_stem(t) for t in tokens}


def extract_prohibited_terms(text: str) -> Set[str]:
    """Extract target words directly governed by explicit prohibition phrasing."""
    clauses = re.split(r'[\.\;\,\n\|—–\(\)]', text)
    prohibited: Set[str] = set()
    explicit_prohib = {
        "zakaz", "zakazane", "zabronione", "zabrania", "zabronił", "nigdy", "never",
        "forbid", "forbidden", "prohibited", "disallow", "disallowed", "zabraniać", "zabroniony",
    }
    for clause in clauses:
        words = re.findall(r'[a-zA-Z0-9ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]{3,}', clause.lower())
        for idx, w in enumerate(words):
            is_prohib = (w in explicit_prohib) or (
                w in ("nie", "not") and idx + 1 < len(words) and words[idx + 1] in (
                    "wolno", "zezwala", "pozwala", "należy", "używa", "używać", "używaj", "loguj", "uruchamiaj",
                )
            )
            if is_prohib:
                start_offset = 2 if w in ("nie", "not") else 1
                for next_w in words[idx + start_offset: idx + start_offset + 5]:
                    if (
                        next_w not in STOP_WORDS
                        and next_w not in PROHIBITION_WORDS
                        and next_w not in GENERIC_WORDS
                    ):
                        prohibited.add(next_w)
    return prohibited


def extract_exclusive_target(text: str) -> Set[str]:
    """Extract target entities or tools bound to exclusivity modifiers."""
    targets: Set[str] = set()
    pattern = r'(?:wyłącznie|tylko|jedynie|solely|exclusively|only)\s+(?:przez|za\s+pomocą|via|through|in|w\s+języku|w|jako|dla)?\s*([a-zA-Z0-9ąćęłńóśźżĄĆĘŁŃÓŚŹŻ\s]{3,35})'
    context_delims = re.compile(r'\b(?:w|we|na|dla|z|ze|od|do|podczas|przed|po|oraz|i|a|ale|gdy)\b|[;,.:()"\']')
    for m in re.finditer(pattern, text.lower()):
        matched_phrase = m.group(1)
        head_part = context_delims.split(matched_phrase, maxsplit=1)[0].strip()
        sub_tokens = extract_tokens(head_part if head_part else matched_phrase) - CONTROL_WORDS - GENERIC_WORDS
        targets |= sub_tokens
    return targets


def extract_governed_action(tokens: Set[str]) -> Set[str]:
    """Identify action verbs from known stem prefixes."""
    return {t for t in tokens if any(polish_stem(t).startswith(p) for p in ACTION_PREFIXES)}

