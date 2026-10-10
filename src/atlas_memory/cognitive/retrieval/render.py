"""V81 Render: pojedyncze kanoniczne cialo formatowania bloku pamieci."""
from __future__ import annotations

import copy
import math
import re
import time
from collections import Counter
from typing import Any, List, Optional, Sequence, Set, Tuple

SALIENCE_DECAY_PER_DAY: float = 0.035  # λ per-day, zgodny z ttl_factor w fusion.py


def _salience_lambda() -> float:
    """V151.1 (F-032): λ per-day do rankingu konf×exp(-λ·age).

    Konfigurowalne przez AtlasConfig.salience_decay_per_day (config ma pierwszeństwo
    nad stałą modułową SALIENCE_DECAY_PER_DAY). Brak wartości w config → 0.035/dzień.
    """
    try:
        from atlas_memory.core.config import get_config

        return float(getattr(get_config(), "salience_decay_per_day", SALIENCE_DECAY_PER_DAY))
    except Exception:
        return SALIENCE_DECAY_PER_DAY


# V151.1 (F-032): kanonizator tematu encji — strip transportowych prefiksów,
# dat, wersji, hex-hash, słów szumu. NIE operuje na EAV-keys (conflict_arbiter.
# entity_root służy innemu celowi — tu chodzi o wolny subject z pamięci.md/prozy).
_TRANSPORT_PREFIX_RE = re.compile(
    r"^(?:hermes:memory_md:|memory_md:|hermes:|kv:|fact:|sop::|sop:|rule:|preference:|constraint:)+",
    re.IGNORECASE,
)
_VERSION_RE = re.compile(r"\bv\d+(?:\.\d+)*\b", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b")
_DOTTED_DATE_RE = re.compile(r"\b\d{1,2}\.\d{1,2}(?:\.\d{1,4})?\b")
_HEX_HASH_RE = re.compile(r"\b[0-9a-f]{7,40}\b", re.IGNORECASE)
_NOISE_WORDS_RE = re.compile(
    r"\b(?:final|release|tag|starego|nowy|nowa|nowe|wydanie|wydania)\b",
    re.IGNORECASE,
)
_KV_ASSIGN_RE = re.compile(r"\b[a-z_][a-z0-9_]{1,30}=([0-9a-f]{7,40})\b", re.IGNORECASE)
_SEPARATORS_RE = re.compile(r"[\s(),=:/|++]+")  # V152-CLOSURE: '+' w szumie (Release+tag, etc.)
_NOISE_SUFFIX_RE = re.compile(
    r"\b(?:backup|plus|main|branch)\b",  # V152-CLOSURE (Filar IV): 'backup'/'plus'/'main'/'branch'
    re.IGNORECASE,
)


def entity_root(subject: Any) -> str:
    """V151.1 (F-032): kanoniczna postać tematu encji z pamięci.md / recall.

    Zestandaryzowane operacje (deterministyczne, bez spaCy):
      1. casefold (bez narodowych mapowań — zachowujemy Asci-lowercase dla PL);
      2. zdjęcie transportowych prefiksów ('hermes:memory_md:', 'kv:', 'fact:'...);
      3. usunięcie hex-hash [0-9a-f]{7,40} (commit SHAs, kody), wersji v\\d+(\\.\\d+)*,
         dat ISO \\d{4}-\\d{2}-\\d{2}, dat kropkowanych dd.mm, słów szumu;
      4. ściśnięcie separatorów ([(),=:/] + whitespace) do pojedynczej spacji;
      5. strip.
    Dwa subjecty tej samej encji (różniące się tylko datą/wersją/SHA/transportem)
    dają identyczny string → dają się zgrupować przez dedup.
    """
    raw = str(subject or "").strip().lower()
    if not raw:
        return ""
    raw = _TRANSPORT_PREFIX_RE.sub("", raw)
    # Najpierw wersje (bo mogą zawierać kropkowane tokeny typu 2.0.0),
    # potem daty, potem hash, potem słowa szumu, na końcu asocjacje 'klucz=hash'.
    raw = _VERSION_RE.sub(" ", raw)
    raw = _ISO_DATE_RE.sub(" ", raw)
    raw = _DOTTED_DATE_RE.sub(" ", raw)
    raw = _NOISE_WORDS_RE.sub(" ", raw)
    # V152-CLOSURE (Filar IV): sufiksy 'backup'/'plus'/'main' z wariantów wpisów.
    raw = _NOISE_SUFFIX_RE.sub(" ", raw)
    # KV_ASSIGN (main=hex) PRZED HEX_HASH — wtedy 'main=0873d87' jako całość;
    # inaczej HEX_HASH_RE wyciąga hash i zostawia 'main=' który przeżywa SEPARATORS.
    raw = _KV_ASSIGN_RE.sub(" ", raw)
    raw = _HEX_HASH_RE.sub(" ", raw)
    raw = _SEPARATORS_RE.sub(" ", raw)
    return raw.strip()


def _record_slot(record: Any) -> Tuple[str, str]:
    """V151.1 (F-032): kanon slotu (predicate, source_type) — klasa ekwiwalentności.

    Rejestry pamięci o identycznym (root_entity, slot) są tym samym faktem w
    różnej redakcji; rejestry o różnym (predicate, source_type) NIE są łączone,
    nawet jeśli temat ten sam (np. status vs release_date).
    """
    if isinstance(record, dict):
        pred = str(record.get("predicate", "")).strip().lower() or "stated"
        st = record.get("source_type", record.get("provenance", ""))
    else:
        pred = str(getattr(record, "predicate", "")).strip().lower() or "stated"
        st = getattr(record, "source_type", None)
    if st is None:
        st = ""
    if hasattr(st, "value"):
        st = str(st.value)
    return pred, str(st or "").strip().lower()


_WORD_TOKEN_RE = re.compile(r"[a-z0-9ąćęłńóśźż]{2,}")


def _record_text_tokens(record: Any) -> Counter:
    """Kosinusowy tor identyfikacji semantycznej: żetony (stemowane) subject+object."""
    from atlas_memory.core.linguistics import _bilingual_stem

    if isinstance(record, dict):
        s_raw = record.get("subject", record.get("key", ""))
        obj = record.get("object", record.get("value", ""))
    else:
        s_raw = getattr(record, "subject", getattr(record, "key", ""))
        obj = getattr(record, "object", getattr(record, "value", ""))
    text = f"{s_raw or ''} {obj or ''}".lower()
    toks = _WORD_TOKEN_RE.findall(text)
    return Counter(_bilingual_stem(t) or t for t in toks)


def _bag_cosine(a: Counter, b: Counter) -> float:
    """Sim ∈ [0,1] (kosinus bow)."""
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = float(sum(a[t] * b[t] for t in common))
    na = math.sqrt(float(sum(v * v for v in a.values())))
    nb = math.sqrt(float(sum(v * v for v in b.values())))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


def knapsack_semantic_dedup(
    records: Sequence[Any],
    dedup_sim: Optional[float] = None,
) -> List[Any]:
    """V151.6 (WEKTOR II): Semantyczna deduplikacja przed wkładaniem do knapsacka.

    Grupa = rekordy o zgodnym podmiocie (po normalizacji: casefold; grupowane też gdy
    jeden podmiot jest prefiksem drugiego o długości >= 6) z kosinusową similarnością
    żetonów (subject+object) >= `dedup_sim` (`AtlasConfig.knapsack_dedup_sim`, def 0.88).
    W grupie zostaje wyłącznie kanoniczny rekord — najświeższy wg epistemicznego
    rankingu `compute_record_rank` (confidence × exp(-λ·age), λ = salience decay).
    Próg <= 0 lub >= 1 wyłącza dedup.
    """
    if dedup_sim is None:
        from atlas_memory.core.config import get_config

        dedup_sim = float(getattr(get_config(), "knapsack_dedup_sim", 0.88))
    docs = list(records)
    if dedup_sim <= 0.0 or dedup_sim >= 1.0 or len(docs) < 2:
        return list(docs)

    def _norm_subject(r: Any) -> str:
        if isinstance(r, dict):
            s = r.get("subject", r.get("key", ""))
        else:
            s = getattr(r, "subject", getattr(r, "key", ""))
        return str(s or "").strip().lower()

    n = len(docs)
    parents = list(range(n))

    def _find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parents[max(ri, rj)] = min(ri, rj)

    token_bags = [_record_text_tokens(r) for r in docs]
    subjects = [_norm_subject(r) for r in docs]
    # V151.1 (F-032): ekwiwalencja EntityRoot × Slot. Dwa zapisy o tym samym
    # znormalizowanym temacie encji (bez dat/wersji/SHA/transportowych prefiksów)
    # i tym samym kanonicznym slocie (predicate, source_type) to TEN SAM fakt
    # w różnej redakcji — łączą się niezależnie od kosinusa surowych tekstów.
    # Różny slot = różny fakt: nigdy nie łączymy przez granicę slotu.
    roots = [entity_root(si) if entity_root(si) else si for si in subjects]  # V151.1 F-032: EntityRoot ekwiwalencja
    slots = [_record_slot(r) for r in docs]
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = subjects[i], subjects[j]
            same_subject = (
                si == sj
                or (len(si) >= 6 and sj.startswith(si))
                or (len(sj) >= 6 and si.startswith(sj))
            )
            if same_subject and _bag_cosine(token_bags[i], token_bags[j]) >= dedup_sim:
                _union(i, j)
                continue
            ri, rj = roots[i], roots[j]
            # Ekwiwalencja root×slot dotyczy WARIANTÓW tematu (surowe si != sj):
            # identyczne surowe podmioty pozostają pod kontrolą kosinusowego
            # progu WEKTOR II (knapsack_dedup_sim ma spójną semantykę wyłączenia).
            if si != sj and ri and rj and ri == rj and slots[i] == slots[j]:
                _union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(_find(i), []).append(i)
    if len(groups) == n:
        return list(docs)

    # V151.6 (WEKTOR II): kanoniczny wariant = najświeższy wg salience
    # (`confidence × exp(-λ·age)`, λ = SALIENCE_DECAY_PER_DAY spójne z
    # ttl_factor w fusion.py); deterministyczny tiebreak nowszym timestampem.
    def _freshness_key(idx: int) -> Tuple[float, float]:
        r = docs[idx]
        if isinstance(r, dict):
            conf = float(r.get("confidence", 1.0) or 0.0)
            ts = float(r.get("timestamp", 0.0) or 0.0)
        else:
            conf = float(getattr(r, "confidence", 1.0) or 0.0)
            ts = float(getattr(r, "timestamp", 0.0) or 0.0)
        ts_seconds = ts / 1000.0 if ts > 1e12 else ts
        age_days = max(0.0, (time.time() - ts_seconds) / 86400.0) if ts_seconds > 0.0 else 0.0
        decay = math.exp(-SALIENCE_DECAY_PER_DAY * age_days)
        return (conf * decay, ts)

    dropped: set[int] = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        canonical = max(members, key=_freshness_key)
        dropped.update(idx for idx in members if idx != canonical)
    return [r for i, r in enumerate(docs) if i not in dropped]


def estimate_tokens(record: Any) -> int:
    if isinstance(record, dict):
        s_raw = record.get("subject", record.get("key", ""))
        obj = record.get("object", record.get("value", ""))
    else:
        s_raw = getattr(record, "subject", getattr(record, "key", ""))
        obj = getattr(record, "object", getattr(record, "value", ""))
    s_len = len(str(s_raw or ""))
    o_len = min(len(str(obj or "")), 1200)
    # Format: "• [{s}] {o[:1200]}" -> 5 additional formatting characters
    return max(1, math.ceil((s_len + o_len + 5) / 3.5) + 1)


def pack_knapsack_01(
    records: List[Any],
    budget: int,
    query: str = "",
    now: Optional[float] = None,
    compact: bool = False,
) -> List[Any]:
    from atlas_memory.cognitive.retrieval.score import compute_record_rank

    # V151.6 (WEKTOR II): semantyczna dedup kanonicznych wariantów przed packowaniem.
    records = knapsack_semantic_dedup(records)
    eff_records = [copy.deepcopy(r) for r in records]
    if compact and sum(estimate_tokens(r) for r in eff_records) > budget * 1.5 and len(eff_records) >= 3:
        compacted = []
        for r in eff_records:
            if isinstance(r, dict):
                obj = str(r.get("object", r.get("value", "")))
            else:
                obj = str(getattr(r, "object", getattr(r, "value", "")))
            if len(obj) > 150 and not obj.endswith("... [compacted]"):
                val = obj[:80] + "... [compacted]"
                if hasattr(r, "model_copy"):
                    r_copy = r.model_copy(deep=True)
                    if hasattr(r_copy, "object"):
                        r_copy.object = val
                    elif hasattr(r_copy, "value"):
                        r_copy.value = val
                    compacted.append(r_copy)
                else:
                    r_copy = copy.deepcopy(r)
                    if isinstance(r_copy, dict):
                        if "object" in r_copy:
                            r_copy["object"] = val
                        else:
                            r_copy["value"] = val
                    elif hasattr(r_copy, "object"):
                        r_copy.object = val
                    elif hasattr(r_copy, "value"):
                        r_copy.value = val
                    compacted.append(r_copy)
            else:
                compacted.append(copy.deepcopy(r))
        eff_records = compacted
    def _rank(r: Any) -> float:
        meta = r.get("metadata", {}) if isinstance(r, dict) else getattr(r, "metadata", {})
        is_sup = bool(r.get("is_superseded") if isinstance(r, dict) else getattr(r, "is_superseded", False)) or bool(meta.get("is_superseded") if isinstance(meta, dict) else getattr(meta, "is_superseded", False))
        if is_sup:
            return -5.0
        if isinstance(r, dict) and "score" in r:
            return float(r["score"])
        sc = getattr(r, "score", None)
        if sc is not None and not hasattr(type(r), "model_fields"):
            return float(sc)
        return compute_record_rank(r, query=query, now=now)

    valid_items = [
        (idx, r, _rank(r), estimate_tokens(r))
        for idx, r in enumerate(eff_records)
        if _rank(r) >= 0.0
    ]
    if not valid_items:
        return []

    # K_tree: Budowa relacyjnego grafu incydencji między kandydatami
    n_items = len(valid_items)
    adj = {i: set() for i in range(n_items)}
    
    def _extract_endpoints(item_tuple: Any) -> Tuple[str, str, Set[str]]:
        r = item_tuple[1]
        if isinstance(r, dict):
            s = str(r.get("subject", r.get("key", ""))).strip().lower()
            o = str(r.get("object", r.get("value", ""))).strip().lower()
            meta_dict = r.get("metadata") or {}
            steps = r.get("steps") or meta_dict.get("steps") or []
        else:
            s = str(getattr(r, "subject", getattr(r, "key", ""))).strip().lower()
            o = str(getattr(r, "object", getattr(r, "value", ""))).strip().lower()
            meta_dict = getattr(r, "metadata", {}) or {}
            steps = getattr(r, "steps", None) or meta_dict.get("steps") or []
        step_nodes = set()
        if isinstance(steps, (list, tuple)):
            for st in steps:
                if isinstance(st, dict):
                    step_nodes.add(str(st.get("source", "")).strip().lower())
                    step_nodes.add(str(st.get("target", "")).strip().lower())
                else:
                    step_nodes.add(str(getattr(st, "source", "")).strip().lower())
                    step_nodes.add(str(getattr(st, "target", "")).strip().lower())
        step_nodes.discard("")
        return s, o, step_nodes

    endpoints = [_extract_endpoints(it) for it in valid_items]

    for i in range(n_items):
        s_i, o_i, steps_i = endpoints[i]
        for j in range(i + 1, n_items):
            s_j, o_j, steps_j = endpoints[j]
            is_connected = False
            # Tranzytywne połączenie łańcucha przyczynowego: (A->B) - (B->C)
            if (o_i and s_j and (o_i == s_j or s_j in o_i or o_i in s_j)) or \
               (o_j and s_i and (o_j == s_i or s_i in o_j or o_j in s_i)):
                is_connected = True
            # Połączenie przez wspólny łańcuch kroków (causal trace overlap)
            elif steps_i and steps_j and (steps_i & steps_j):
                is_connected = True
            elif steps_i and (s_j in steps_i or o_j in steps_i):
                is_connected = True
            elif steps_j and (s_i in steps_j or o_i in steps_j):
                is_connected = True
            # Wspólny korzeń encyjny (entity-root clustering) dla niepustych encji
            elif s_i and s_j and s_i == s_j and len(s_i) >= 3:
                is_connected = True

            if is_connected:
                adj[i].add(j)
                adj[j].add(i)

    # Identyfikacja spójnych składowych poddrzew (Steiner components)
    visited = set()
    components: List[List[int]] = []
    for i in range(n_items):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for nbr in adj[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            components.append(comp)

    # Wycena poddrzew: utility(C) = sum(score) * (1.0 + 0.35 * edges_count(C))
    evaluated_comps = []
    for comp in components:
        comp_records = [valid_items[idx] for idx in comp]
        base_score = sum(it[2] for it in comp_records)
        comp_cost = sum(it[3] for it in comp_records)
        # Liczba krawędzi wewnętrznych w poddrzewie
        comp_set = set(comp)
        internal_edges = sum(len(adj[idx] & comp_set) for idx in comp) // 2
        # Filar IV-2: Obliczenie sumarycznej dotkliwości CPoF dla składowej C
        comp_cpof_severity = sum(
            float(
                (
                    it[1].get("metadata", {})
                    if isinstance(it[1], dict)
                    else getattr(it[1], "metadata", {}) or {}
                ).get("cpof_severity", 0.0)
                or 0.0
            )
            for it in comp_records
        )
        utility = base_score * (1.0 + 0.35 * float(internal_edges) + 0.50 * comp_cpof_severity)
        density = utility / max(1, comp_cost)
        evaluated_comps.append({
            "indices": comp,
            "records": comp_records,
            "base_score": base_score,
            "cost": comp_cost,
            "edges": internal_edges,
            "utility": utility,
            "density": density,
        })

    # Sortowanie poddrzew po gęstości topologicznej
    evaluated_comps.sort(key=lambda c: (c["density"], c["utility"]), reverse=True)

    packed: List[Any] = []
    curr_cost = 0

    for comp_info in evaluated_comps:
        comp_records = comp_info["records"]
        c_cost = comp_info["cost"]
        if curr_cost + c_cost <= budget:
            # Całe poddrzewo mieści się w budżecie: włączamy bez rozrywania geodezyjnej
            for it in comp_records:
                packed.append(it[1])
            curr_cost += c_cost
        else:
            # Poddrzewo nie mieści się w całości: dopakowujemy najlepsze węzły z poddrzewa
            sorted_nodes = sorted(comp_records, key=lambda it: (it[2] / max(1, it[3]), it[2]), reverse=True)
            for it in sorted_nodes:
                if curr_cost + it[3] <= budget:
                    packed.append(it[1])
                    curr_cost += it[3]

    if not packed and valid_items and budget > 0:
        best_it = max(valid_items, key=lambda it: (it[2] / max(1, it[3]), it[2]))
        packed.append(best_it[1])

    return [copy.deepcopy(r) for r in sorted(packed, key=lambda r: _rank(r), reverse=True)]


def render_context_block(
    records: List[Any],
    max_object_chars: int = 1200,
    max_total_chars: int = 6000,
    include_provenance: bool = False,
) -> str:
    """Buduje sformatowany markdown bloku kontekstu."""
    if not records:
        return ""

    lines = ["## ATLAS Memory Context"]
    for r in records:
        if isinstance(r, dict):
            subj = str(r.get("subject", ""))
            pred = str(r.get("predicate", ""))
            obj = str(r.get("object", ""))
            store = str(r.get("source", ""))
            meta = dict(r.get("metadata") or {})
            is_sup = bool(r.get("is_superseded") or meta.get("is_superseded") or meta.get("superseded"))
        else:
            subj = str(getattr(r, "subject", ""))
            pred = str(getattr(r, "predicate", ""))
            obj = str(getattr(r, "object", ""))
            store = str(getattr(r, "source", ""))
            meta = dict(getattr(r, "metadata", {}) or {})
            is_sup = bool(getattr(r, "is_superseded", False) or meta.get("is_superseded") or meta.get("superseded"))

        if pred.upper() == "SUPERSEDED_BY":
            continue
        if subj and subj == obj:
            continue
        if is_sup:
            continue

        if max_object_chars > 0 and len(obj) > max_object_chars:
            obj = obj[:max_object_chars] + " ...[truncated]"

        prov_str = f" [{store}]" if (include_provenance and store) else ""

        # Topologiczny render relacyjny / kauzalny
        path_steps = (r.get("steps") if isinstance(r, dict) else getattr(r, "steps", None)) or meta.get("steps")
        risk = (r.get("risk_level") if isinstance(r, dict) else getattr(r, "risk_level", None)) or meta.get("risk_level")
        impact = (r.get("impact_type") if isinstance(r, dict) else getattr(r, "impact_type", None)) or meta.get("impact_type")

        if path_steps and isinstance(path_steps, (list, tuple)):
            step_strs = []
            for s in path_steps:
                if isinstance(s, dict):
                    step_strs.append(f"{s.get('source', '')} -[{s.get('predicate', '')}]-> {s.get('target', '')}")
                else:
                    step_strs.append(f"{getattr(s, 'source', '')} -[{getattr(s, 'predicate', '')}]-> {getattr(s, 'target', '')}")
            path_chain = " -> ".join(step_strs)
            risk_tag = f", ryzyko: {risk}" if risk else ""
            rel_label = f" -({pred})->" if (pred and pred not in ("stated_memory", "state_is")) else ""
            lines.append(f"• [{subj}]{rel_label} [{obj}] (ścieżka: {path_chain}{risk_tag}){prov_str}")
        elif impact:
            rel_label = f" -({pred})->" if pred else ""
            lines.append(f"• [{subj}]{rel_label} [{obj}] (typ: {impact}){prov_str}")
        elif store == "graph" and pred and pred not in ("stated_memory", "state_is"):
            lines.append(f"• [{subj}] -({pred})-> [{obj}]{prov_str}")
        else:
            lines.append(f"• [{subj}] {obj}{prov_str}")

    result = "\n".join(lines)
    if max_total_chars > 0 and len(result) > max_total_chars:
        cut_idx = result.rfind("\n", 0, max_total_chars)
        if cut_idx > len("## ATLAS Memory Context"):
            result = result[:cut_idx]
        else:
            result = result[:max_total_chars]
    return result

