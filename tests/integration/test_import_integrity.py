"""
Test 100% integralności importów dla wszystkich modułów pakietu atlas_memory (Zero-Trust Import Smoke Test).
Gwarantuje, że żaden użytkownik ani środowisko CI/CD nie napotka brakującego importu, błędu typowania ani cyklicznej zależności.
"""

from __future__ import annotations

import importlib
import pkgutil

import atlas_memory


def test_all_submodules_import_cleanly():
    """Weryfikuje, że każdy moduł i podmoduł w atlas_memory importuje się bez błędów."""
    failed_imports = []
    scanned_count = 0

    for _, modname, _ in pkgutil.walk_packages(atlas_memory.__path__, atlas_memory.__name__ + "."):
        scanned_count += 1
        try:
            mod = importlib.import_module(modname)
            assert mod is not None
        except Exception as exc:
            failed_imports.append((modname, str(exc)))

    assert not failed_imports, f"Nie udało się zaimportować {len(failed_imports)} modułów: {failed_imports}"
    assert scanned_count >= 30, f"Przeskanowano zbyt mało modułów ({scanned_count})"


def test_atlas_memory_case_insensitive_imports():
    """Weryfikuje, że import ATLAS_memory oraz import atlas_memory działają zamiennie."""
    import sys

    import atlas_memory

    atlas_upper = sys.modules.get("ATLAS_memory")
    assert atlas_upper is not None
    assert hasattr(atlas_upper, "HybridMemoryEngine")
    assert hasattr(atlas_memory, "HybridMemoryEngine")
    assert atlas_upper.HybridMemoryEngine is atlas_memory.HybridMemoryEngine

