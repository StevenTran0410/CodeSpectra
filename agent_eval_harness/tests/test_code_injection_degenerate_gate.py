"""Exercises the injected run_eval.py template's degenerate-run gate — the guard that stops a run reporting success while its agents swallowed failures; every signal here stays target-neutral, using only span/latency/digest facts the harness itself records, never the shape or wording of the target's output."""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

import pytest

_TEMPLATES = Path(__file__).parent.parent / "agent_eval_harness" / "code_injection" / "templates"


@pytest.fixture
def run_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    for name in ("tracer.py", "run_eval.py"):
        shutil.copy(_TEMPLATES / name, tmp_path / name)
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = importlib.util.spec_from_file_location("aeh_injected_run_eval", tmp_path / "run_eval.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case(agent: str, trace: str, *, spans: int = 1, latency: int = 5000,
          error: str | None = None, digest: str = "") -> dict:
    return {
        "agent_id": agent, "trace_id": trace, "success": True, "spans": spans,
        "latency_ms": latency, "span_error": error, "output_len": 500,
        "output_digest": digest or f"digest-{trace}",
    }


def test_healthy_run_produces_no_findings(run_eval: ModuleType) -> None:
    results = [_case("a", f"t{i}", latency=4000 + i * 500) for i in range(4)]

    assert run_eval._degenerate_findings(results, {}) == []


def test_swallowed_failure_is_caught_despite_reported_success(run_eval: ModuleType) -> None:
    # A span-count-only gate would miss this: the agent caught its own exception, returned a schema-valid empty result, and was recorded as succeeded.
    results = [_case("a", f"t{i}", latency=6000) for i in range(3)]
    results.append(_case("a", "dead", latency=2))

    findings = run_eval._degenerate_findings(results, {})

    assert len(findings) == 1
    assert "dead" in findings[0]
    assert "median" in findings[0]


def test_span_error_is_caught_even_when_latency_is_normal(run_eval: ModuleType) -> None:
    results = [_case("a", "t1"), _case("a", "t2", error="ValidationError: field required")]

    findings = run_eval._degenerate_findings(results, {})

    assert any("ValidationError" in f for f in findings)


def test_zero_spans_is_caught(run_eval: ModuleType) -> None:
    results = [_case("a", "t1"), _case("a", "t2", spans=0)]

    findings = run_eval._degenerate_findings(results, {})

    assert any("no spans" in f for f in findings)


def test_constant_fallback_output_is_caught(run_eval: ModuleType) -> None:
    # Different inputs yielding a byte-identical answer means a constant is coming back, detected without the harness knowing what the target's fallback value looks like.
    results = [_case("a", f"t{i}", digest="same") for i in range(3)]

    findings = run_eval._degenerate_findings(results, {})

    assert any("byte-identical" in f for f in findings)


def test_latency_floor_scales_to_the_target_not_a_fixed_millisecond_count(run_eval: ModuleType) -> None:
    # A fast target's normal case must not be flagged just for being quicker than a slow target's — the floor is derived from each agent's own spread.
    fast = [_case("fast", f"t{i}", latency=30 + i) for i in range(4)]
    slow = [_case("slow", f"s{i}", latency=30_000 + i) for i in range(4)]

    assert run_eval._degenerate_findings(fast, {}) == []
    assert run_eval._degenerate_findings(slow, {}) == []


def test_latency_check_needs_enough_peers_to_have_a_median(run_eval: ModuleType) -> None:
    # With one case there is no distribution to compare against, so a threshold must not be invented; error and zero-span signals still apply.
    assert run_eval._degenerate_findings([_case("a", "t1", latency=1)], {}) == []


def test_gate_thresholds_are_overridable_from_wiring(run_eval: ModuleType) -> None:
    results = [_case("a", f"t{i}", latency=6000) for i in range(3)]
    results.append(_case("a", "slowish", latency=3000))

    assert run_eval._degenerate_findings(results, {}) == []
    findings = run_eval._degenerate_findings(results, {"gate": {"latency_floor_ratio": 0.9}})
    assert any("slowish" in f for f in findings)
