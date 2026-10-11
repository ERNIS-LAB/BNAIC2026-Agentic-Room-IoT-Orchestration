from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires a reachable live gateway and LLM backend",
    )


def pytest_collection_modifyitems(config, items):
    if os.getenv("RUN_LIVE_TESTS") == "1":
        return

    skip_live = pytest.mark.skip(
        reason="set RUN_LIVE_TESTS=1 to run live gateway tests"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_live)
