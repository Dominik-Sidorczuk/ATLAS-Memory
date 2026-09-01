"""Global Pytest Configuration and Shared Test Fixtures for ATLAS."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def clean_env():
    """Ensure clean test environment without side effects."""
    yield
