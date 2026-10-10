from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "benchmark" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)

