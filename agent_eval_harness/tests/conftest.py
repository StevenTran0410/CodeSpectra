"""CRITICAL: Haystack tracing env vars must be set before `haystack` is
imported anywhere in this process — these two lines must stay above every
other import in this file (mirrors cli.py's own ordering requirement).
"""
import os

os.environ.setdefault("HAYSTACK_CONTENT_TRACING_ENABLED", "true")
os.environ.setdefault("HAYSTACK_AUTO_TRACE_ENABLED", "false")

import tempfile  # noqa: E402
from collections.abc import AsyncGenerator  # noqa: E402

import pytest  # noqa: E402

from agent_eval_harness.store.database import close_db, init_db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
async def _init_test_db() -> AsyncGenerator[None, None]:
    """Mirrors backend/tests/conftest.py's own pattern: one fresh temp-dir
    SQLite DB per test session via the AEH_DATA_DIR env var."""
    tmpdir = tempfile.mkdtemp()
    os.environ["AEH_DATA_DIR"] = tmpdir
    await init_db()
    yield
    await close_db()
