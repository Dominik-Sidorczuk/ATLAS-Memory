"""V89/T3 — BRAMKA LIVENESS: każdy moduł src/atlas_memory musi być osiągalny.

Konstytucja architektury V89: "700 unit testów to kpina", jeśli testy zielone
dla kodu, którego żadna produkcyjna ścieżka nie wywołuje. Dlatego bramka liczy
DOMKNIĘCIE IMPORTÓW (czysty `ast`, zero importowania pakietu, zero sieci)
z dwóch produkcyjnych wejść:

  1. atlas_memory.server.atlas_daemon            (daemon = jedyne wejście serwerowe)
  2. atlas_memory.hermes.*                       (provider/tools/prefix_guard/plugin_hooks
                                                  = wejście w procesie Hermesa)

i PADA, jeżeli jakikolwiek plik .py pod src/atlas_memory jest nieosiągalny.
Wyjątek: `__init__.py` pakietów (fasady, które nie muszą być importowane wprost).

Bramka jest celowo TANIA (<2 s) i deterministyczna: tylko readFile + ast.parse.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
ATLAS_SRC = REPO_ROOT / "src" / "atlas_memory"
SRC_ROOT = REPO_ROOT / "src"

# Produkcyjne wejścia: daemon RPC oraz pakiet integracji z Hermesem.
ENTRY_MODULES: List[str] = [
    "atlas_memory.server.atlas_daemon",
    "atlas_memory.hermes.atlas_provider",
    "atlas_memory.hermes.tools",
    "atlas_memory.hermes.prefix_guard",
    "atlas_memory.hermes.plugin_hooks",
]


def module_to_file(module: str) -> Path | None:
    """Ścieżka pliku modułu/pakietu w src/ (bez importowania czegokolwiek)."""
    if not module.startswith("atlas_memory"):
        return None
    rel = module[len("atlas_memory") :].lstrip(".")
    base = ATLAS_SRC if rel else ATLAS_SRC
    for candidate in (base / f"{rel.replace('.', '/')}.py", base / rel.replace(".", "/") / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def imported_modules(path: Path) -> Set[str]:
    """Wszystkie odwołania `import atlas_memory...` / `from atlas_memory... import ...` z pliku."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, ValueError, OSError):  # pragma: no cover - plik jest sprawdzany dalej
        return set()

    mods: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("atlas_memory"):
                    mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # import względny: rozwiązywany ścieżką pliku poniżej
                continue
            if node.module and node.module.startswith("atlas_memory"):
                mods.add(node.module)
                # `from atlas_memory import engine` -> kandydat atlas_memory.engine
                if node.module == "atlas_memory":
                    for alias in node.names:
                        mods.add(f"atlas_memory.{alias.name}")
    return mods


def import_closure(entries: Iterable[str]) -> Set[Path]:
    """BFS po domknięciu importów z listy modułów wejściowych."""
    seen: Set[Path] = set()
    queue: List[str] = list(entries)
    while queue:
        module = queue.pop()
        path = module_to_file(module)
        if path is None or path in seen:
            continue
        seen.add(path)
        for dep in imported_modules(path):
            dep_path = module_to_file(dep)
            if dep_path is not None and dep_path not in seen:
                queue.append(dep)
    return seen


def all_source_files() -> List[Path]:
    return sorted(p for p in ATLAS_SRC.rglob("*.py") if "__pycache__" not in p.parts)


def is_package_init(path: Path) -> bool:
    return path.name == "__init__.py"


def test_entries_are_resolvable() -> None:
    """Wejścia bramki nie mogą milcząco zniknąć — inaczej closure = pusta."""
    missing = [m for m in ENTRY_MODULES if module_to_file(m) is None]
    assert not missing, f"V89/T3: Produkcyjne wejścia nie istnieją w src/atlas_memory: {missing}"


def test_v89_module_liveness_gate() -> None:
    """Żaden plik .py w src/atlas_memory nie może być martwy (nieosiągalny z produkcji)."""
    reachable = import_closure(ENTRY_MODULES)
    assert reachable, "V89/T3: domknięcie importów jest puste — bramka nie działa"

    orphans = [p for p in all_source_files() if p not in reachable and not is_package_init(p)]

    assert not orphans, (
        "V89/T3 BRAMKA MARTWEGO KODU: "
        f"{len(orphans)} plik(ów) w src/atlas_memory jest nieosiągalnych z produkcji "
        "(atlas_memory.server.atlas_daemon | atlas_memory.hermes.*). Albo podłącz je do "
        "żywego wywołania, albo usuń (Konstytucja V89):\n"
        + "\n".join(f"  - {p.relative_to(SRC_ROOT)}" for p in orphans)
    )
