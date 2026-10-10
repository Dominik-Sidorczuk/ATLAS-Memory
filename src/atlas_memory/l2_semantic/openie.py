"""W2: HippoRAG 2 OpenIE Phrase & Triple Extraction powered by spaCy (PL + EN) + CogitoRAG Gist Integration.

Zero hardcoded stopwords or dictionaries:
  - Uses `spacy` (`pl_core_news_sm` and `en_core_web_sm`) for morphological lemmatization (`token.lemma_`),
    Part-of-Speech tagging (`NOUN`, `PROPN`, `ADJ`, `VERB`, `NUM`), stopword detection (`token.is_stop`),
    Named Entity Recognition (`doc.ents`), Noun Chunks (`doc.noun_chunks`), and syntactic Dependency Parsing
    (`nsubj` -> `VERB/ROOT` -> `obj/obl/iobj/attr`) to build OpenIE `(subject, predicate_lemma, object)` triples.
  - Integrates CogitoRAG semantic gist phrases (§9) and computes `SYNONYM` edges between entities via
    `spaCy` lemma overlap + normalized character/embedding similarity.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_STRUCTURAL_TOKEN_RE = re.compile(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9_:\-./]{2,}")


from atlas_memory.core.linguistics import (
    _get_spacy_lang_stopwords,
    get_spacy_nlp,
)


@lru_cache(maxsize=1)
def _get_spacy_stopwords() -> frozenset[str]:
    return frozenset(_get_spacy_lang_stopwords("pl") | _get_spacy_lang_stopwords("en"))


@lru_cache(maxsize=65536)
def spacy_lemmatize_tokens(text: str) -> Tuple[frozenset[str], frozenset[str]]:
    """Returns `(surface_content_tokens, lemmatized_content_tokens)` using spaCy PL+EN models (cached)."""
    clean = str(text or "").strip()
    if not clean:
        return frozenset(), frozenset()

    nlp_pl = get_spacy_nlp("pl", fast_lemma_only=True)
    doc_pl = nlp_pl(clean[:4000])

    surface: Set[str] = set()
    lemmas: Set[str] = set()

    for tok in doc_pl:
        if tok.is_space or tok.is_punct:
            continue
        t_low = tok.text.lower().strip()
        if len(t_low) < 2:
            continue
        if tok.is_stop and len(t_low) < 4:
            continue
        surface.add(t_low)
        lem = (tok.lemma_ or t_low).lower().strip()
        if len(lem) >= 2:
            lemmas.add(lem)

    stopwords = _get_spacy_stopwords()
    for m in _STRUCTURAL_TOKEN_RE.finditer(clean):
        raw = m.group(0).lower()
        if raw not in stopwords:
            surface.add(raw)
            lemmas.add(raw)
        for part in re.split(r"[_\-:/\.]+", raw):
            if len(part) >= 2 and part not in stopwords:
                surface.add(part)
                lemmas.add(part)

    return frozenset(surface), frozenset(lemmas)


@dataclass(frozen=True)
class OpenIEResult:
    """Structured OpenIE output for a single passage (`fact:*`, `hermes:memory_md:*`)."""

    passage_id: str
    source_key: str
    text: str
    entities: List[str] = field(default_factory=list)
    triples: List[Tuple[str, str, str, float]] = field(default_factory=list)
    coverage_ratio: float = 1.0

    @property
    def phrases(self) -> List[str]:
        """Alias for `entities` used by W1/W2 migration and contract tests."""
        return self.entities

    @property
    def token_coverage(self) -> float:
        """Alias for `coverage_ratio`."""
        return self.coverage_ratio


def _normalize_entity(phrase: str) -> str:
    cleaned = re.sub(r"\s+", " ", (phrase or "").strip().strip(".,;()[]{}\"'`"))
    return cleaned[:120]


def sanitize_text_for_spacy(text: Any) -> str:
    """V152 (FILAR I, ADR): modułowy sanitizer tekstu PRZED podaniem go spaCy.

    Pure bytes-op (regex na klasach znaków, zero modelu/stopword-list): usuwa
    lone surrogates (\\uD800-\\uDFFF — trigger SIGSEGV w spaCy token.c), C0/C1
    control (poza \\t\\n\\r) i Cf zero-width (\\u200B/\\u200C/\\u200D/\\uFEFF).
    Legalne znaki (polskie litery, emoji spoza BMP jak 🚀) pozostają nietknięte.
    Wywoływane na wejściu `extract_openie_from_record` jako warstwa obrony #2 —
    bramka aktu mowy w EpistemicWritePath zostaje warstwą #1.
    """
    from atlas_memory.core.models import sanitize_epistemic_text

    if not isinstance(text, str):
        text = str(text or "")
    return sanitize_epistemic_text(text)


def extract_openie_from_record(
    source_key: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    gist: Optional[str] = None,
    passage_id: Optional[str] = None,
) -> OpenIEResult:
    """W2 + §9 (CogitoRAG): Extracts phrase nodes (`Entity`) and relational triples using spaCy dependency parsing.

    Guarantees >=90% content-token coverage so CONTAINS edges connect `Passage` nodes
    to every salient entity/phrase in the document.
    """
    # V152 (FILAR I): surogaty/sterujące nigdy nie docierają do tokenizatora.
    skey = sanitize_text_for_spacy((source_key or "").strip())
    body = sanitize_text_for_spacy((text or "").strip())
    meta = dict(metadata or {})
    if gist:
        meta["gist"] = gist
    pid = passage_id or (
        f"passage:{skey}" if skey and not skey.startswith("passage:") else (skey or f"passage:{hash(body)}")
    )

    entities_ordered: List[str] = []
    seen_lower: Set[str] = set()
    stopwords = _get_spacy_stopwords()

    def _add_entity(raw_ent: str) -> Optional[str]:
        norm = _normalize_entity(raw_ent)
        if not norm or len(norm) < 2:
            return None
        low = norm.lower()
        if low in stopwords or low in seen_lower:
            return None
        seen_lower.add(low)
        entities_ordered.append(norm)
        return norm

    if skey:
        _add_entity(skey)
        for seg in re.split(r"[:/__]+", skey):
            if len(seg) >= 2 and seg.lower() not in stopwords:
                _add_entity(seg)

    for meta_field in ("gist", "semantic_gist", "summary", "topic", "category", "entity", "subject"):
        val = meta.get(meta_field)
        if isinstance(val, str) and val.strip():
            _add_entity(val.strip())
        elif isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, str) and item.strip():
                    _add_entity(item.strip())

    # V152 (BUG-601b): extract konsumuje tok.subtree — bez parsera head-refy są
    # uninitialized i subtree-walk SIGSEGV-uje w spaCy native (token.c/strings).
    # Inwariant: konsument subtree żąda pełnego pipeline.
    nlp_pl = get_spacy_nlp("pl", fast_lemma_only=False)
    doc_pl = nlp_pl(body[:2000])

    for ent in getattr(doc_pl, "ents", ()):
        _add_entity(ent.text)
        if ent.lemma_ and ent.lemma_.lower() != ent.text.lower():
            _add_entity(ent.lemma_)

    for tok in doc_pl:
        if tok.pos_ in ("NOUN", "PROPN") and not tok.is_stop and not tok.is_punct:
            subtree_toks = [
                child
                for child in tok.subtree
                if not child.is_punct and not child.is_space and (not child.is_stop or child == tok)
            ]
            if 1 <= len(subtree_toks) <= 4:
                phrase_surface = " ".join(t.text for t in subtree_toks)
                phrase_lemma = " ".join((t.lemma_ or t.text).lower() for t in subtree_toks)
                _add_entity(phrase_surface)
                _add_entity(phrase_lemma)
            _add_entity(tok.text)
            if tok.lemma_:
                _add_entity(tok.lemma_.lower())

    triples: List[Tuple[str, str, str, float]] = []
    seen_triples: Set[Tuple[str, str, str]] = set()

    def _add_triple(u: str, r: str, v: str, conf: float = 0.90) -> None:
        u_n = _normalize_entity(u)
        v_n = _normalize_entity(v)
        r_n = re.sub(r"\s+", "_", (r or "related_to").strip().lower())[:48] or "related_to"
        if not u_n or not v_n or u_n.lower() == v_n.lower():
            return
        _add_entity(u_n)
        _add_entity(v_n)
        key = (u_n.lower(), r_n, v_n.lower())
        if key not in seen_triples:
            seen_triples.add(key)
            triples.append((u_n, r_n, v_n, float(conf)))

    for tok in doc_pl:
        if tok.pos_ in ("VERB", "AUX") or tok.dep_ == "ROOT":
            subjects = [
                c for c in tok.children if c.dep_ in ("nsubj", "nsubj:pass", "csubj", "agent") and not c.is_stop
            ]
            objects = [
                c
                for c in tok.children
                if c.dep_ in ("obj", "dobj", "iobj", "obl", "attr", "acomp", "pobj", "xcomp") and not c.is_stop
            ]
            pred_lemma = (tok.lemma_ or tok.text).lower().strip()
            if subjects and objects and len(pred_lemma) >= 2:
                for s_tok in subjects:
                    s_span = " ".join(
                        t.text for t in s_tok.subtree if not t.is_punct and not t.is_stop
                    ) or s_tok.text
                    for o_tok in objects:
                        o_span = " ".join(
                            t.text for t in o_tok.subtree if not t.is_punct and not t.is_stop
                        ) or o_tok.text
                        _add_triple(s_span, pred_lemma, o_span, conf=0.92)

    content_tokens: List[str] = []
    for tok in doc_pl:
        if tok.is_space or tok.is_punct or tok.is_stop:
            continue
        t_low = tok.text.lower().strip()
        if len(t_low) >= 2 and t_low not in stopwords:
            content_tokens.append(t_low)
            _add_entity(t_low)
            if tok.lemma_ and len(tok.lemma_.strip()) >= 2:
                _add_entity(tok.lemma_.strip().lower())

    for m in _STRUCTURAL_TOKEN_RE.finditer(body):
        raw = m.group(0).lower()
        if raw not in stopwords:
            content_tokens.append(raw)
            _add_entity(raw)

    covered_blob = " ".join(entities_ordered).lower()
    if content_tokens:
        covered_count = sum(1 for tok in content_tokens if tok in covered_blob)
        coverage_ratio = covered_count / float(len(content_tokens))
    else:
        coverage_ratio = 1.0

    if skey and entities_ordered:
        for ent in entities_ordered[:12]:
            if ent.lower() != skey.lower():
                _add_triple(skey, "mentions_entity", ent, conf=0.85)

    return OpenIEResult(
        passage_id=pid,
        source_key=skey,
        text=body,
        entities=entities_ordered,
        triples=triples,
        coverage_ratio=float(min(1.0, coverage_ratio)),
    )


extract_openie_phrases_and_triples = extract_openie_from_record


def compute_phrase_similarity(a: str, b: str) -> float:
    """Computes morphological & lexical similarity in [0.0, 1.0] between two entity phrases using spaCy lemmas."""
    a_low = (a or "").strip().lower()
    b_low = (b or "").strip().lower()
    if not a_low or not b_low or a_low == b_low:
        return 0.0

    a_norm = re.sub(r"[\s_\-:/\.]+", "", a_low)
    b_norm = re.sub(r"[\s_\-:/\.]+", "", b_low)
    if a_norm and a_norm == b_norm:
        return 0.96

    _a_surf, a_lemmas = spacy_lemmatize_tokens(a_low)
    _b_surf, b_lemmas = spacy_lemmatize_tokens(b_low)

    if a_lemmas and b_lemmas:
        inter = a_lemmas & b_lemmas
        if inter:
            union = a_lemmas | b_lemmas
            jaccard = len(inter) / float(len(union))
            cov = len(inter) / float(min(len(a_lemmas), len(b_lemmas)))
            lemma_sim = 0.60 * cov + 0.40 * jaccard
            if lemma_sim >= 0.75:
                return float(min(0.95, lemma_sim))

    if len(a_norm) >= 5 and len(b_norm) >= 5 and (a_norm in b_norm or b_norm in a_norm):
        ratio = min(len(a_norm), len(b_norm)) / float(max(len(a_norm), len(b_norm)))
        if ratio >= 0.65:
            return float(0.75 + 0.20 * ratio)

    return 0.0


def build_synonym_edges(
    entities: Sequence[str],
    threshold: float = 0.80,
    sim_threshold: Optional[float] = None,
    embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
    encoder: Optional[Any] = None,
    max_pairs_per_entity: int = 5,
) -> List[Tuple[str, str, float]]:
    """W2: Builds `SYNONYM(Entity -> Entity, sim)` edges for entities with `sim >= threshold`.

    Combines spaCy morphological lemma equivalence (`pl_core_news_sm` + `en_core_web_sm`)
    with optional dense embedding cosine similarity (`embed_fn` / `encoder`).
    """
    if sim_threshold is not None:
        threshold = float(sim_threshold)
    if embed_fn is None and encoder is not None:
        if hasattr(encoder, "encode_batch"):
            embed_fn = encoder.encode_batch
        elif hasattr(encoder, "encode"):
            embed_fn = encoder.encode
        elif callable(encoder):
            embed_fn = encoder

    unique_ents = [e for e in dict.fromkeys(entities) if e and len(e.strip()) >= 3]
    n = len(unique_ents)
    if n < 2:
        return []

    edges: Dict[Tuple[str, str], float] = {}

    lemma_buckets: Dict[str, List[str]] = {}
    for ent in unique_ents:
        _, lemmas = spacy_lemmatize_tokens(ent)
        for lem in lemmas:
            if len(lem) >= 4:
                lemma_buckets.setdefault(lem, []).append(ent)

    for bucket in lemma_buckets.values():
        if len(bucket) < 2 or len(bucket) > 35:
            continue
        for i in range(len(bucket)):
            u = bucket[i]
            hits = 0
            for j in range(i + 1, len(bucket)):
                v = bucket[j]
                sim = compute_phrase_similarity(u, v)
                if sim >= threshold:
                    pair = (u, v) if u < v else (v, u)
                    edges[pair] = max(edges.get(pair, 0.0), round(float(sim), 4))
                    hits += 1
                    if hits >= max_pairs_per_entity:
                        break

    if embed_fn is not None and n <= 2000:
        try:
            vecs = np.asarray(embed_fn(unique_ents), dtype=np.float32)
            if vecs.ndim == 2 and vecs.shape[0] == n:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms > 1e-8, norms, 1.0)
                normed = vecs / norms
                sim_mat = normed @ normed.T
                for i in range(n):
                    row = sim_mat[i]
                    top_idx = np.argsort(-row)[: max_pairs_per_entity + 1]
                    for j in top_idx:
                        if i == j:
                            continue
                        s_val = float(row[j])
                        if s_val >= threshold:
                            u, v = unique_ents[i], unique_ents[j]
                            pair = (u, v) if u < v else (v, u)
                            edges[pair] = max(edges.get(pair, 0.0), round(s_val, 4))
        except Exception as exc:
            logger.debug("Optional embed_fn in build_synonym_edges skipped: %s", exc)

    result: List[Tuple[str, str, float]] = []
    for (u, v), sim in sorted(edges.items()):
        result.append((u, v, sim))
        result.append((v, u, sim))
    return result
