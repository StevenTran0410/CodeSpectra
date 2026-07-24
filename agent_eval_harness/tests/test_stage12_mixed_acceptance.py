"""CS-317 gated acceptance test — wraps the REAL Stage-1->2 mixed-framework harness.

This is NOT a hermetic unit test: it drives the real discovery->expansion->build pipeline against a
live codespectra.db snapshot using the FSOFT provider (real LLM calls). It is SKIPPED by default so
the normal suite stays offline/deterministic; opt in with AEH_ACCEPTANCE=1 to run the real gate.

    AEH_ACCEPTANCE=1 python -m pytest tests/test_stage12_mixed_acceptance.py -s
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

_ENABLED = os.getenv("AEH_ACCEPTANCE") == "1"
_SNAPSHOT = os.getenv("AEH_ACCEPTANCE_SNAPSHOT", "cb66719d-7cfc-4254-bc68-4c2fa4299ff6")
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stage12_mixed_acceptance.py"

pytestmark = pytest.mark.skipif(
    not _ENABLED, reason="set AEH_ACCEPTANCE=1 to run the real Stage-1->2 mixed-framework acceptance gate"
)


@pytest.mark.anyio
async def test_stage12_mixed_framework_union_real_pipeline():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_cs317_acceptance", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = argparse.Namespace(
        snapshot=_SNAPSHOT, effort="medium", provider_id=None, node_budget=100, hop_cap=3,
    )
    exit_code = await mod.run(args)
    assert exit_code == 0, (
        "Real Stage-1->2 acceptance failed: the qa/analysis community must split into THREE separate "
        "single-framework candidates — Haystack (>=12), LangGraph (8 bound_method WITH StateGraph "
        "edges, deep_research), plain-python (QAAgent) — with NO blended framework anywhere. "
        "See harness stdout for the exact failing gate."
    )
