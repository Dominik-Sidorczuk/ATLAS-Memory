"""Global Pytest Configuration and Shared Test Fixtures for ATLAS."""
from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def clean_env():
    """Ensure clean test environment without side effects."""
    yield


_ATLAS_TEST_TMP_ROOT = Path(f"/tmp/atlas-t-{os.getpid()}")


@pytest.fixture()
def uds_dir() -> Iterator[Path]:
    """V100 F3/F4 (BUG-CAL38/39): krótki (<100 znaków) unikalny per-test katalog pod
    testowe sockety/pid UDS. TMPDIR (~/.hermes/cache/scratch) przekracza limit sun_path
    AF_UNIX (107 bajtów), więc bind na socketach pod tmp_path kończy się
    OSError: AF_UNIX path too long. Sprzątanie: per-testowe katalogi znikają po teście,
    root tylko gdy pusty."""
    _ATLAS_TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="uds-", dir=str(_ATLAS_TEST_TMP_ROOT)))
    assert len(str(path)) < 100, f"ścieżka testowych UDS musi być krótka (<100): {path}"
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            _ATLAS_TEST_TMP_ROOT.rmdir()  # czyścimy root tylko, gdy pusty
        except OSError:
            pass
