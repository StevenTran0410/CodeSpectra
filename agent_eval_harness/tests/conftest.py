"""CRITICAL: Haystack tracing env vars must be set before `haystack` is
imported anywhere in this process — these two lines must stay above every
other import in this file (mirrors cli.py's own ordering requirement).
"""
import os

os.environ.setdefault("HAYSTACK_CONTENT_TRACING_ENABLED", "true")
os.environ.setdefault("HAYSTACK_AUTO_TRACE_ENABLED", "false")

import tempfile  # noqa: E402
from collections.abc import AsyncGenerator, Iterator  # noqa: E402

import pytest  # noqa: E402

from agent_eval_harness.config import AEHConfig  # noqa: E402
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


@pytest.fixture(autouse=True)
def _isolate_aeh_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Tests must never be influenced by an ambient .aeh/config.yaml sitting in
    a developer's cwd (e.g. left over from a live-run smoke test against a real
    or mock LLM provider) — CLI tests must always see an empty config unless a
    test explicitly constructs its own AEHConfig. Without this, a stray local
    config file silently changes default provider selection (FakeLLMClient vs
    CodeSpectraProxyClient) and breaks otherwise-offline tests.
    """
    monkeypatch.setattr(AEHConfig, "load", classmethod(lambda cls, path=None: cls()))
    yield
