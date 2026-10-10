"""V87 — Zero-Trust Closeout (T1 registr redakcji, T2 hoisting df BM25,
T3 trwały licznik kroków L0-TTT, T4 mechaniczna bramka ochrony daemon.log).

RED-first: każdy test defektu poniżej PADA na 08e7216 (wyjątki oznaczone
w docstringu jako GUARD przechodzą przed i po naprawie — pilnują, żeby
naprawa nie cofnęła istniejącej ochrony).
"""
from __future__ import annotations

import ast
import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from atlas_memory.cognitive.retrieval import score as score_mod
from atlas_memory.cognitive.retrieval.score import score_candidates
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.server.atlas_daemon import AtlasDaemon

REDACTIONS_TABLE = "audit_redactions"
FORGED_ROW = "[REDACTED-FORGED]"
REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# T1 — registr redakcji (audit_redactions) jako jedyna legalizacja pominięcia
# ---------------------------------------------------------------------------
async def _kv_with_rows(tmp_path: Path, n: int = 3) -> VerifiedKVStore:
    kv = VerifiedKVStore(db_path=str(tmp_path / "atlas.db"))
    for i in range(n):
        await kv.set_state(f"fact:secret_{i}", f"sekret-{i}", confidence=1.0)
    return kv


def _audit_rows(kv: VerifiedKVStore) -> List[sqlite3.Row]:
    conn = kv._ensure_conn()
    return conn.execute("SELECT seq, key, value_hash, entry_hash, new_value FROM state_audit_log ORDER BY seq ASC").fetchall()


def _raw_update_new_value(kv: VerifiedKVStore, seq: int, new_value: str) -> None:
    conn = kv._ensure_conn()
    with conn:
        conn.execute("UPDATE state_audit_log SET new_value = ? WHERE seq = ?", (new_value, seq))


def _registry_rows(kv: VerifiedKVStore) -> List[sqlite3.Row]:
    conn = kv._ensure_conn()
    return conn.execute(f"SELECT seq, key, value_hash, reason FROM {REDACTIONS_TABLE} ORDER BY seq ASC").fetchall()


@pytest.mark.asyncio
async def test_t1_i_legal_redaction_is_authorised_and_chained(tmp_path: Path) -> None:
    """T1(i) RED: redakcja przez publiczne API → registry + valid=true + łańcuch nienaruszony."""
    kv = await _kv_with_rows(tmp_path)
    rows_before = _audit_rows(kv)
    target = rows_before[1]
    chain_before = [(r["seq"], r["key"], r["value_hash"], r["entry_hash"]) for r in rows_before]

    res = kv.redact_audit_entry(int(target["seq"]), reason="SEC-001")
    assert res.get("status") == "ok", res

    reg = _registry_rows(kv)
    assert len(reg) == 1, f"redakcja MUSI byc zarejestrowana, registry={[dict(r) for r in reg]}"
    assert int(reg[0]["seq"]) == int(target["seq"])
    assert reg[0]["key"] == target["key"]
    assert reg[0]["value_hash"] == target["value_hash"], "registry musi nosic ORYGINALNY value_hash"
    assert reg[0]["reason"] == "SEC-001"

    rows_after = _audit_rows(kv)
    assert str(rows_after[1]["new_value"]).startswith("[REDACTED-"), rows_after[1]["new_value"]
    chain_after = [(r["seq"], r["key"], r["value_hash"], r["entry_hash"]) for r in rows_after]
    assert chain_after == chain_before, "redakcja nie moze zmieniac zadnego elementu lancucha (poza new_value)"

    is_valid, broken = kv.verify_audit_log_sync(deep=True)
    assert (is_valid, broken) == (True, 0), f"legalna redakcja musi przejsc deep verify: {is_valid}, {broken}"


@pytest.mark.asyncio
async def test_t1_ii_forged_redaction_without_registry_is_tamper(tmp_path: Path) -> None:
    """T1(ii) RED (dzis przechodzi): jeden UPDATE „[REDACTED-..." bez wiersza w rejestrze = tamper."""
    kv = await _kv_with_rows(tmp_path)
    rows = _audit_rows(kv)
    victim = int(rows[1]["seq"])

    _raw_update_new_value(kv, victim, FORGED_ROW)

    is_valid, broken = kv.verify_audit_log_sync(deep=True)
    assert is_valid is False, "podrobiona redakcja BEZ wpisu w rejestrze nie moze przejsc deep verify"
    assert broken == victim, f"zlamany seq musi wskazywac podmieniony wiersz: {broken} != {victim}"

    daemon = AtlasDaemon.__new__(AtlasDaemon)
    engine = type("E", (), {})()
    engine.kv = kv
    daemon.engine = engine
    rpc = await daemon._handle_verify_audit_log({"deep": True})
    assert rpc["valid"] is False, rpc
    assert rpc["tampered"] is True, rpc


@pytest.mark.asyncio
async def test_t1_iii_plain_value_swap_is_still_detected(tmp_path: Path) -> None:
    """T1(iii) GUARD: zwykla podmiana wartosci (bez prefiksu) wykrywana przed i po naprawie."""
    kv = await _kv_with_rows(tmp_path)
    rows = _audit_rows(kv)
    victim = int(rows[0]["seq"])

    _raw_update_new_value(kv, victim, "niewinny-tekst")

    is_valid, broken = kv.verify_audit_log_sync(deep=True)
    assert is_valid is False
    assert broken == victim


@pytest.mark.asyncio
async def test_t1_iv_registry_is_keyed_by_seq_not_by_prefix(tmp_path: Path) -> None:
    """T1(iv) RED: jedna zarejestrowana redakcja NIE legalizuje redakcji na innym seq."""
    kv = await _kv_with_rows(tmp_path)
    rows = _audit_rows(kv)
    authorised, forged = int(rows[0]["seq"]), int(rows[2]["seq"])

    assert kv.redact_audit_entry(authorised, reason="SEC-001").get("status") == "ok"
    _raw_update_new_value(kv, forged, FORGED_ROW)

    is_valid, broken = kv.verify_audit_log_sync(deep=True)
    assert is_valid is False, "rejestr musi dzialac per-seq, nie per-prefiks"
    assert broken == forged


@pytest.mark.asyncio
async def test_t1_v_registry_row_must_match_value_hash(tmp_path: Path) -> None:
    """T1(v) RED: wpis w rejestrze z niezgodnym value_hash nie legalizuje redakcji."""
    kv = await _kv_with_rows(tmp_path)
    rows = _audit_rows(kv)
    victim = int(rows[1]["seq"])
    conn = kv._ensure_conn()
    with conn:
        conn.execute(
            f"INSERT OR REPLACE INTO {REDACTIONS_TABLE} (seq, key, value_hash, reason, created_at) VALUES (?,?,?,?,?)",
            (victim, rows[1]["key"], hashlib.sha256(b"inny-preobraz").hexdigest(), "forged", time.time()),
        )
    _raw_update_new_value(kv, victim, FORGED_ROW)

    is_valid, broken = kv.verify_audit_log_sync(deep=True)
    assert is_valid is False, "registry z niezgodnym value_hash nie moze legalizowac redakcji"
    assert broken == victim


# ---------------------------------------------------------------------------
# T2 — hoisting df w BM25 + nazwane stałe z proweniencją
# ---------------------------------------------------------------------------
def _reference_bm25_scores(query: str, candidates: List[Dict[str, Any]]) -> List[float]:
    """Wierna kopia algorytmu z 08e7216 (df liczony wewnatrz petli) — wzorzec parytetu."""
    import math
    import re

    from atlas_memory.cognitive.retrieval.score import _bilingual_stem
    from atlas_memory.core.linguistics import LanguageAssetManager

    polish_stopwords = LanguageAssetManager.get_instance().get_stopwords("pl")
    q_words = [w.lower() for w in re.findall(r"[^\W_]+", query) if len(w) >= 3 and _bilingual_stem(w) not in polish_stopwords]
    if not q_words:
        q_words = [w.lower() for w in re.findall(r"[^\W_]+", query) if len(w) >= 3]
    q_stems = [_bilingual_stem(w) for w in q_words]
    q_lower = query.lower().strip()

    doc_tokens: List[List[str]] = []
    for c in candidates:
        s, o = str(c.get("subject", "")).lower(), str(c.get("object", "")).lower()
        cand_text = f"{s} {o}".replace("_", " ").replace(":", " ").replace("-", " ")
        doc_tokens.append([_bilingual_stem(w) for w in re.findall(r"[^\W_]+", cand_text) if len(w) >= 3])

    N = len(doc_tokens)
    k1 = 1.2
    out: List[float] = []
    for i, d in enumerate(doc_tokens):
        s_score = 0.0
        for qt in q_stems:
            nq = sum(1 for doc in doc_tokens if qt in doc)
            if nq == 0:
                continue
            idf = math.log(1.0 + (N - nq + 0.5) / (nq + 0.5))
            tf = d.count(qt)
            s_score += idf * (tf * (k1 + 1)) / (tf + k1)
        s_cand, o_cand = str(candidates[i].get("subject", "")).lower(), str(candidates[i].get("object", "")).lower()
        if q_lower and (q_lower in s_cand or q_lower in o_cand):
            s_score += 1.0
        out.append(s_score)
    return out


def _fixture_candidates() -> List[Dict[str, Any]]:
    subjects = [
        "rule:preferred_editor", "rule:git:branch_strategy", "fact:user_role", "fact:project_stack",
        "preference:language", "constraint:no_force_push", "rule:test_first", "fact:daemon_socket",
    ]
    objects = [
        "nvim jest preferowanym edytorem uzytkownika", "branch strategy: zawsze worktree, nigdy main",
        "architect", "python pixi ruff pytest", "polski w rozmowie", "nigdy force push na main",
        "testy przed kodem RED GREEN", "~/.hermes/atlas.sock unix socket",
    ]
    cands = []
    for i, (s, o) in enumerate(zip(subjects, objects, strict=True)):
        cands.append({
            "subject": s, "predicate": "stated_memory", "object": o,
            "confidence": 1.0, "source_type": "user_explicit", "timestamp": 1_700_000_000.0,
            "vector_score": round(0.9 - 0.07 * i, 4),
        })
    return cands


def test_t2_bm25_parytet_z_referencja() -> None:
    """T2 GUARD: po hoistingu df wyniki BM25/ranking sa identyczne jak na 08e7216.

    Wartosci GOLDEN zmierzone na 08e7216 (przed refaktorem) — surowe wyjscie:
    ``python -c "from atlas_memory.cognitive.retrieval.score import score_candidates; ..."``
    na fiksturze ``_fixture_candidates()``. Zmiana algorytmu = czerwony test.
    """
    golden = {
        "preferowany edytor nvim": [
            ("rule:preferred_editor", 1.0),
            ("rule:git:branch_strategy", 0.8666),
            ("fact:user_role", 0.7481),
            ("fact:project_stack", 0.6699),
            ("preference:language", 0.5968),
            ("constraint:no_force_push", 0.5233),
            ("rule:test_first", 0.452),
            ("fact:daemon_socket", 0.3801),
        ],
        "python pytest ruff": [
            ("rule:preferred_editor", 0.95),
            ("rule:git:branch_strategy", 0.8666),
            ("fact:project_stack", 0.8611),
            ("fact:user_role", 0.7481),
            ("preference:language", 0.5968),
            ("constraint:no_force_push", 0.5233),
            ("rule:test_first", 0.452),
            ("fact:daemon_socket", 0.3801),
        ],
    }
    for query, expected in golden.items():
        got = score_candidates(query, [dict(c) for c in _fixture_candidates()], fusion_strategy="rrf")
        assert [(c["subject"], c["score"]) for c in got] == expected, f"parytet score zlamany dla {query!r}"


@pytest.mark.parametrize("query", ["preferowany edytor nvim", "branch strategy worktree", "python pytest ruff", "force push"])
def test_t2_bm25_scores_identyczne_z_petla_wewnetrzna(query: str) -> None:
    """T2 GUARD: surowe BM25 (przed normalizacja) musi byc 1:1 z wersja z df liczonym w petli."""
    cands = _fixture_candidates()
    doc_tokens = score_mod._tokenize_candidates_for_bm25(cands)
    q_stems = score_mod._query_stems(query)
    expected = _reference_bm25_scores(query, cands)
    got = score_mod.bm25_scores_for_corpus(q_stems, doc_tokens)
    # score_candidates dokłada bonus frazowy poza BM25 — odtwarzamy go tutaj,
    # zeby porownanie objelo tez stala LEX_EXACT_PHRASE_BONUS.
    q_lower = query.lower().strip()
    for i, c in enumerate(cands):
        s_cand, o_cand = str(c.get("subject", "")).lower(), str(c.get("object", "")).lower()
        if q_lower and (q_lower in s_cand or q_lower in o_cand):
            got[i] += score_mod.LEX_EXACT_PHRASE_BONUS
    assert got == expected


def test_t2_module_constants_are_named_with_provenance() -> None:
    """T2 RED: k1/b/bonus musza byc nazwanymi stalami modulowymi z komentarzem proweniencji."""
    assert score_mod.BM25_K1 == 1.2
    assert score_mod.BM25_B == 0.0
    assert score_mod.LEX_EXACT_PHRASE_BONUS == 1.0
    src = (REPO_ROOT / "src" / "atlas_memory" / "cognitive" / "retrieval" / "score.py").read_text(encoding="utf-8")
    assert "prowenien" in src.lower(), "stale BM25 wymagaja jawnego komentarza proweniencji (AGENTS.md #10)"


def _doc_tokens_scans_inside_query_loop(src: str) -> List[str]:
    """Zwraca lokalizacje skanow `doc_tokens` wewnatrz petli po terminach zapytania.

    Forma O(Q*N): petla po `q_stems` (lub `q_terms`), w ktorej ciele wystepuje
    jakikolwiek iterator po `doc_tokens` (petla zagniezdzona lub comprehension).
    """
    tree = ast.parse(src)
    found: List[str] = []

    def inner_scans(node: ast.AST, outer_line: int) -> None:
        for child in ast.walk(node):
            if child is node:
                continue
            line = getattr(child, "lineno", -1)
            if isinstance(child, (ast.For, ast.AsyncFor)) and "doc_tokens" in ast.unparse(child.iter):
                found.append(f"linia {outer_line} (petla po terminach) -> linia {line}: for ... in doc_tokens")
            elif isinstance(child, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)):
                text = ast.unparse(child)
                if "doc_tokens" in text and " in doc_tokens" in text:
                    found.append(f"linia {outer_line} (petla po terminach) -> linia {line}: {text[:70]}")

    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for outer in [n for n in ast.walk(fn) if isinstance(n, (ast.For, ast.AsyncFor))]:
            it = ast.unparse(outer.iter)
            if "q_stem" not in it and "q_term" not in it:
                continue
            inner_scans(ast.Module(body=list(outer.body), type_ignores=[]), outer.lineno)
    return found


def test_t2_gate_no_doc_scan_inside_query_loop() -> None:
    """T2 RED (bramka mechaniczna): skan doc_tokens wewnatrz petli po q_stems = cofniecie hoistingu df."""
    src = (REPO_ROOT / "src" / "atlas_memory" / "cognitive" / "retrieval" / "score.py").read_text(encoding="utf-8")
    violations = _doc_tokens_scans_inside_query_loop(src)
    assert not violations, f"df musi byc policzony RAZ przed petla scoringowa; skany w petli: {violations}"


def test_t2_gate_has_teeth_on_legacy_source() -> None:
    """T2 GUARD: bramka AST musi faktycznie lapac stara postac (dowod, ze nie jest martwa)."""
    legacy = (
        "def score_candidates(query, candidates):\n"
        "    doc_tokens = [[w for w in query.split()] for c in candidates]\n"
        "    q_stems = [w for w in query.split()]\n"
        "    for i, d in enumerate(doc_tokens):\n"
        "        for qt in q_stems:\n"
        "            nq = sum(1 for doc in doc_tokens if qt in doc)\n"
        "    return []\n"
    )
    assert _doc_tokens_scans_inside_query_loop(legacy), "bramka AST nie ma zebow — nie lapie starej postaci"


# ---------------------------------------------------------------------------
# T3 — trwały licznik kroków L0-TTT (regresja strażnika V57)
# ---------------------------------------------------------------------------
def _make_engine(db_path: Path):
    from atlas_memory.engine import HybridMemoryEngine

    return HybridMemoryEngine.create_default(db_path=str(db_path), qdrant_location=":memory:")


def _obs(engine: Any, name: str) -> None:
    from atlas_memory.core.models import ActionPlan

    engine.record_mental_transition(ActionPlan(name=name, parameters={"k": name}), session_id="s1")


def test_t3_step_counter_survives_engine_restart(tmp_path: Path) -> None:
    """T3 RED: licznik krokow TTT musi przetrwac restart silnika (dzis startuje od 0)."""
    db = tmp_path / "atlas.db"
    eng1 = _make_engine(db)
    for i in range(3):
        _obs(eng1, f"w{i}")
    assert eng1.ttt.step_count == 3, f"kroki w zyciu silnika: {eng1.ttt.step_count}"
    eng1.kv._ensure_conn().close()

    eng2 = _make_engine(db)
    assert eng2.ttt.step_count == 3, (
        f"po restarcie licznik L0-TTT wyzerowal sie ({eng2.ttt.step_count}) mimo 3 zapisow w bazie "
        "— straznik V57 raportuje 'brak uczenia' na kazdym swiezo wystartowanym daemonie"
    )
    eng2.kv._ensure_conn().close()


@pytest.mark.asyncio
async def test_t3_every_state_write_bumps_counter(tmp_path: Path) -> None:
    """T3 RED: KAZDY zapis stanu przez produkcyjny socket (set/task_feedback) podbija licznik."""
    daemon = AtlasDaemon(socket_path=tmp_path / "d.sock", pid_path=tmp_path / "d.pid", engine=_make_engine(tmp_path / "atlas.db"))
    engine = daemon.engine

    async def steps() -> int:
        return int((await daemon._handle_get_stats({}))["ttt_steps"])

    await daemon._handle_set({"key": "t3:alpha", "value": "wartosc alpha"})
    base = await steps()
    assert base > 0, f"set nie podbil licznika L0-TTT ({base})"

    await daemon._handle_set({"key": "t3:beta", "value": "wartosc beta"})
    assert await steps() > base, "drugi zapis nie podbil licznika"

    after_set = await steps()
    res = await daemon._handle_task_feedback({"key": "t3:alpha", "task_success": True})
    assert res.get("status") == "ok", res
    assert await steps() > after_set, "task_feedback zapisuje stan, wiec MUSI podbic licznik L0-TTT"

    engine.kv._ensure_conn().close()


# ---------------------------------------------------------------------------
# T4 — mechaniczna bramka ochrony produkcyjnego daemon.log
# ---------------------------------------------------------------------------
APPEND_MODES = {"a", "a+"}
LOG_NAME = "daemon.log"
_DESTRUCTIVE_ATTRS = {"truncate", "write_text", "write_bytes", "unlink", "write"}


def _mentions_daemon_log(node: ast.AST, bound: set[str]) -> bool:
    text = ast.unparse(node)
    if LOG_NAME in text:
        return True
    return any(isinstance(n, ast.Name) and n.id in bound for n in ast.walk(node))


def _bound_log_names(tree: ast.AST) -> set[str]:
    """Nazwy zmiennych, ktorych wartesc wyprowadza sie ze sciezki daemon.log (punkt stalego)."""
    bound: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                value = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                targets, value = [node.target.id], node.value
            else:
                continue
            if not targets or not _mentions_daemon_log(value, bound):
                continue
            for name in targets:
                if name not in bound:
                    bound.add(name)
                    changed = True
    return bound


def _scan_daemon_log_violations(root: Path) -> List[str]:
    """Zwraca listę naruszeń ochrony daemon.log w kodzie produkcyjnym pod ``root``."""
    violations: List[str] = []
    for py in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        bound = _bound_log_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == "open":
                if node.args and _mentions_daemon_log(node.args[0], bound):
                    mode = None
                    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                        mode = node.args[1].value
                    for kw in node.keywords:
                        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                            mode = kw.value.value
                    if mode not in APPEND_MODES:
                        violations.append(f"{py}:{node.lineno}: open(daemon.log, mode={mode!r}) — tylko dopisywanie")
            elif (isinstance(fn, ast.Name) and fn.id.endswith("FileHandler")) or (isinstance(fn, ast.Attribute) and fn.attr.endswith("FileHandler")):
                if node.args and _mentions_daemon_log(node.args[0], bound):
                    mode = None
                    for kw in node.keywords:
                        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                            mode = kw.value.value
                    if mode not in APPEND_MODES:
                        violations.append(f"{py}:{node.lineno}: {ast.unparse(fn)}(daemon.log, mode={mode!r}) — tylko dopisywanie")
            elif isinstance(fn, ast.Attribute) and fn.attr in _DESTRUCTIVE_ATTRS:
                if _mentions_daemon_log(fn.value, bound):
                    violations.append(f"{py}:{node.lineno}: .{fn.attr}(...) na sciezce daemon.log — zabronione")
    return violations


def test_t4_gate_has_teeth_on_injected_violations(tmp_path: Path) -> None:
    """T4 GUARD: bramka musi łapać każdy z zakazanych wzorców (dowód, że nie jest martwa)."""
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "truncating.py").write_text(
        "from pathlib import Path\n"
        "log = Path.home() / \".hermes\" / \"atlas\" / \"daemon.log\"\n"
        "def nuke():\n"
        "    open(log, \"w\")\n"
        "    log.write_text(\"\")\n"
        "    log.unlink()\n"
        "    f = open(log, \"a+\")\n"
        "    f.truncate(0)\n",
        encoding="utf-8",
    )
    (bad / "handler.py").write_text(
        "import logging\n"
        "from pathlib import Path\n"
        "logging.FileHandler(str(Path.home() / \".hermes\" / \"atlas\" / \"daemon.log\"))\n",
        encoding="utf-8",
    )
    (bad / "ok.py").write_text(
        "import logging\n"
        "from pathlib import Path\n"
        "lock = Path(\"/tmp/x.lock\")\n"
        "log_file = Path.home() / \".hermes\" / \"atlas\" / \"daemon.log\"\n"
        "handler = logging.FileHandler(str(log_file), mode=\"a\", encoding=\"utf-8\")\n"
        "with open(lock, \"a+\") as f:\n"
        "    f.truncate(0)\n",
        encoding="utf-8",
    )
    found = _scan_daemon_log_violations(bad)
    joined = "\n".join(found)
    assert joined.count("truncating.py") >= 4, f"nie wykryto wszystkich naruszen:\n{joined}"
    assert "handler.py" in joined, f"brak wykrycia FileHandler bez mode='a':\n{joined}"
    assert "ok.py" not in joined, f"launcher (append + truncate locka) nie moze byc naruszeniem:\n{joined}"


def test_t4_production_sources_never_truncate_daemon_log() -> None:
    """T4 GUARD: src/ i scripts/ nie moga otwierac/truncatowac produkcyjnego daemon.log."""
    violations = _scan_daemon_log_violations(REPO_ROOT / "src") + _scan_daemon_log_violations(REPO_ROOT / "scripts")
    assert not violations, "naruszenie ochrony daemon.log:\n" + "\n".join(violations)
