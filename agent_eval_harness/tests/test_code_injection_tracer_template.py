"""Exercises the injected tracer.py template against REAL haystack.tracing (haystack-ai is
already an AEH dependency) — the same protocol the target repo will actually invoke, copied
to an isolated directory so `out/` lands next to the copy, exactly as it will in `.aeh/`."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("haystack") is None, reason="haystack not installed"
)

_TEMPLATE_SRC = Path(__file__).parent.parent / "agent_eval_harness" / "code_injection" / "templates" / "tracer.py"


@pytest.fixture
def tracer_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setenv("HAYSTACK_CONTENT_TRACING_ENABLED", "true")
    dest = tmp_path / "tracer.py"
    shutil.copy(_TEMPLATE_SRC, dest)
    spec = importlib.util.spec_from_file_location("aeh_injected_tracer", dest)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module

    from haystack.tracing import disable_tracing
    disable_tracing()


def _log_lines(module: ModuleType) -> list[dict]:
    if not module._LOG_PATH.exists():
        return []
    return [json.loads(line) for line in module._LOG_PATH.read_text(encoding="utf-8").splitlines()]


def test_log_path_is_per_pid_sibling_out_dir(tracer_module: ModuleType) -> None:
    assert tracer_module._LOG_PATH.parent.name == "out"
    assert tracer_module._LOG_PATH.parent.parent == tracer_module._OUT_DIR.parent
    assert str(os.getpid()) in tracer_module._LOG_PATH.name


def test_register_tracer_attaches_to_haystack(tracer_module: ModuleType) -> None:
    from haystack.tracing import tracer as global_tracer

    tracer_module.register_tracer()

    assert global_tracer.actual_tracer is not None
    assert isinstance(global_tracer.actual_tracer, tracer_module.AehTracer)


def test_component_span_is_written_with_tags(tracer_module: ModuleType) -> None:
    tracer_module.register_tracer()
    from haystack.tracing import tracer as global_tracer

    with tracer_module.set_current_trace("trace-123"):
        with global_tracer.trace(
            "haystack.component.run",
            tags={
                "haystack.component.name": "project_identity",
                "haystack.component.type": "SectionAgentComponent",
                "haystack.component.input": {"snapshot_id": "s1"},
                "haystack.component.output": {"domain": "web app"},
            },
        ):
            pass

    lines = _log_lines(tracer_module)
    assert len(lines) == 1
    span = lines[0]
    assert span["record"] == "span"
    assert span["trace_id"] == "trace-123"
    assert span["component_id"] == "project_identity"
    assert span["span_type"] == "agent"
    assert json.loads(span["input_json"]) == {"snapshot_id": "s1"}
    assert json.loads(span["output_json"]) == {"domain": "web app"}
    assert span["parent_span_id"] is None


def test_pipeline_root_span_is_never_written_and_child_is_not_orphaned(tracer_module: ModuleType) -> None:
    tracer_module.register_tracer()
    from haystack.tracing import tracer as global_tracer

    with global_tracer.trace("haystack.pipeline.run"):
        with global_tracer.trace(
            "haystack.component.run", tags={"haystack.component.name": "auditor"}
        ):
            pass

    lines = _log_lines(tracer_module)
    assert len(lines) == 1  # the root span itself was never written
    assert lines[0]["parent_span_id"] is None  # not a dangling reference to the dropped root


def test_nested_component_spans_link_parent_to_child(tracer_module: ModuleType) -> None:
    tracer_module.register_tracer()
    from haystack.tracing import tracer as global_tracer

    with global_tracer.trace(
        "haystack.component.run", tags={"haystack.component.name": "outer"}
    ) as outer:
        with global_tracer.trace(
            "haystack.component.run", tags={"haystack.component.name": "inner"}
        ):
            pass

    lines = _log_lines(tracer_module)
    assert len(lines) == 2
    inner_record = next(l for l in lines if l["component_id"] == "inner")
    assert inner_record["parent_span_id"] == outer.span_id
