"""Code-injection tests: degenerate gate, file writer, tracer template, wiring, branch target."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from agent_eval_harness.code_injection.file_writer import (
    apply_marker_block,
    remove_marker_block,
    write_generated_file,
)
from agent_eval_harness.code_injection.injection_target import (
    BranchInjectionTarget,
    InjectionTargetError,
)
from agent_eval_harness.code_injection.wiring import build_wiring
from agent_eval_harness.mapping.system_map import Component, SystemMap

_needs_git = pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
    reason="git not available",
)
_needs_haystack = pytest.mark.skipif(
    importlib.util.find_spec("haystack") is None, reason="haystack not installed"
)

_TEMPLATES = Path(__file__).parent.parent / "agent_eval_harness" / "code_injection" / "templates"
_TEMPLATE_SRC = _TEMPLATES / "tracer.py"


# ---------------------------------------------------------------------------
# Degenerate-run gate (injected run_eval.py template) -- every signal here stays
# target-neutral, using only span/latency/digest facts the harness itself records.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# File writer (marker-block injection)
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as f:
        return f.read()


def test_write_generated_file_carries_do_not_edit_header(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "generated.py"
    write_generated_file(path, "x = 1\n")

    text = _read(path)
    assert text.startswith("# Code generated by CodeSpectra AEH. DO NOT EDIT.\n")
    assert "x = 1\n" in text


def test_apply_marker_block_no_anchor_inserts_at_top(tmp_path: Path) -> None:
    path = tmp_path / "existing.py"
    path.write_text("import os\nprint(os.getcwd())\n", encoding="utf-8", newline="")

    apply_marker_block(path, "hook", "import sys\n")

    text = _read(path)
    assert text.splitlines()[:4] == [
        "# aeh:begin hook", "import sys", "# aeh:end hook", "import os",
    ]


def test_apply_marker_block_with_anchor_inserts_after_matched_line(tmp_path: Path) -> None:
    path = tmp_path / "main.py"
    path.write_text("import a\nimport b\nimport c\n", encoding="utf-8", newline="")

    apply_marker_block(path, "router", "import d\n", anchor="import b", where="after")

    lines = _read(path).splitlines()
    assert lines == ["import a", "import b", "# aeh:begin router", "import d", "# aeh:end router", "import c"]


def test_apply_marker_block_before_anchor(tmp_path: Path) -> None:
    path = tmp_path / "main.py"
    path.write_text("import argparse\nimport sys\n", encoding="utf-8", newline="")

    apply_marker_block(path, "tracer", "import os\n", anchor="import argparse", where="before")

    lines = _read(path).splitlines()
    assert lines == ["# aeh:begin tracer", "import os", "# aeh:end tracer", "import argparse", "import sys"]


def test_apply_marker_block_rerun_replaces_in_place_not_duplicated(tmp_path: Path) -> None:
    path = tmp_path / "existing.py"
    path.write_text("import os\n", encoding="utf-8", newline="")

    apply_marker_block(path, "hook", "import sys\n")
    apply_marker_block(path, "hook", "import sys, json\n")

    text = _read(path)
    assert text.count("# aeh:begin hook") == 1
    assert "import sys, json" in text
    assert "import sys\n" not in text.replace("import sys, json\n", "")


def test_remove_marker_block_deletes_block_and_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "existing.py"
    path.write_text("import os\n", encoding="utf-8", newline="")
    apply_marker_block(path, "hook", "import sys\n")

    removed_first = remove_marker_block(path, "hook")
    removed_second = remove_marker_block(path, "hook")

    assert removed_first is True
    assert removed_second is False
    assert _read(path) == "import os\n"


def test_crlf_file_stays_fully_crlf_after_edit(tmp_path: Path) -> None:
    """Regression guard: a naive read/write round-trip must not normalize CRLF to LF (the exact risk flagged for backend/main.py, which is 100% CRLF)."""
    path = tmp_path / "main.py"
    original = "import argparse\r\nimport sys\r\n\r\ndef main():\r\n    pass\r\n"
    path.write_bytes(original.encode("utf-8"))

    apply_marker_block(path, "tracer", "import os", anchor="import argparse", where="before")

    raw = path.read_bytes()
    assert b"\n" not in raw.replace(b"\r\n", b"")  # every \n is part of a \r\n pair
    assert b"def main():\r\n    pass\r\n" in raw  # untouched tail preserved byte-for-byte


def test_multiline_block_into_crlf_file_has_no_stray_blank_lines(tmp_path: Path) -> None:
    """Regression test: a multi-line block built with plain "\\n" must not gain a stray blank line before the closing marker in a CRLF file."""
    path = tmp_path / "main.py"
    path.write_bytes(b"import argparse\r\nimport sys\r\n")
    block = 'import os\nos.environ.setdefault("X", "true")\n'

    apply_marker_block(path, "tracer", block, anchor="import argparse", where="before")

    raw = path.read_bytes()
    assert raw == (
        b"# aeh:begin tracer\r\n"
        b"import os\r\n"
        b'os.environ.setdefault("X", "true")\r\n'
        b"# aeh:end tracer\r\n"
        b"import argparse\r\n"
        b"import sys\r\n"
    )


# ---------------------------------------------------------------------------
# Tracer template -- exercised against REAL haystack.tracing, copied to an isolated
# directory so `out/` lands next to the copy, exactly as it will in `.aeh/`.
# ---------------------------------------------------------------------------


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
    path = module.log_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@_needs_haystack
def test_log_path_is_per_pid_sibling_out_dir(tracer_module: ModuleType) -> None:
    assert tracer_module.log_path().parent.name == "out"
    assert str(os.getpid()) in tracer_module.log_path().name


@_needs_haystack
def test_log_path_follows_aeh_out_dir_env(tracer_module: ModuleType, tmp_path, monkeypatch) -> None:
    """Output must follow the harness data dir, never sit in the target's source tree."""
    monkeypatch.setenv("AEH_OUT_DIR", str(tmp_path / "runs"))
    assert tracer_module.log_path().parent == tmp_path / "runs"


@_needs_haystack
def test_register_tracer_attaches_to_haystack(tracer_module: ModuleType) -> None:
    from haystack.tracing import tracer as global_tracer

    tracer_module.register_tracer()

    assert global_tracer.actual_tracer is not None
    assert isinstance(global_tracer.actual_tracer, tracer_module.AehTracer)


@_needs_haystack
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


@_needs_haystack
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


@_needs_haystack
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


# ---------------------------------------------------------------------------
# Wiring (build_wiring against SystemMap)
# ---------------------------------------------------------------------------


def _system_map() -> SystemMap:
    return SystemMap(
        target_system_id="test_system",
        components=[
            Component(id="project_identity", role="writer", entry_point="a:b"),
            Component(id="auditor", role="judge", entry_point="c:d"),
        ],
    )


def test_wiring_component_ids_come_from_system_map_not_guessed() -> None:
    """Component IDs derived from SystemMap, not guessed. Generic builder has no hardcodes."""
    wiring = build_wiring(_system_map(), plan_id="plan-abc")

    assert wiring["component_ids"] == ["auditor", "project_identity"]
    assert wiring["plan_id"] == "plan-abc"


# ---------------------------------------------------------------------------
# Branch injection target (git branch prepare/restore)
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    (tmp_path / "README.md").write_text("init", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "branch", "-M", "main"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    return tmp_path


def _current_branch(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()


def _branch_exists(repo: Path, name: str) -> bool:
    branches = subprocess.run(
        ["git", "branch", "--list", name],
        cwd=repo, capture_output=True, text=True,
    ).stdout
    return name in branches


@_needs_git
def test_prepare_creates_branch_without_switching(git_repo: Path) -> None:
    original = _current_branch(git_repo)
    info = BranchInjectionTarget.prepare(git_repo, "sess-abc")
    assert info.branch_name == "aeh/eval-sess-abc"
    assert info.current_branch == original
    assert _branch_exists(git_repo, "aeh/eval-sess-abc")
    # HEAD never moved — that's the entire point of this design.
    assert _current_branch(git_repo) == original


@_needs_git
def test_prepare_does_not_require_a_clean_tree(git_repo: Path) -> None:
    (git_repo / "dirty.txt").write_text("uncommitted", encoding="utf-8")
    # Creating a branch ref never touches the working tree, so a dirty tree is fine.
    info = BranchInjectionTarget.prepare(git_repo, "sess-123")
    assert _branch_exists(git_repo, "aeh/eval-sess-123")
    assert info.current_branch == "main"


@_needs_git
def test_prepare_idempotent_on_repeat_calls(git_repo: Path) -> None:
    info1 = BranchInjectionTarget.prepare(git_repo, "sess-abc")
    info2 = BranchInjectionTarget.prepare(git_repo, "sess-abc")
    assert info1.branch_name == info2.branch_name == "aeh/eval-sess-abc"
    assert info1.current_branch == info2.current_branch


@_needs_git
def test_restore_refuses_dirty_eval_branch(git_repo: Path) -> None:
    original = _current_branch(git_repo)
    BranchInjectionTarget.prepare(git_repo, "sess-xyz")
    subprocess.run(["git", "checkout", "aeh/eval-sess-xyz"], cwd=git_repo, check=True, capture_output=True)
    (git_repo / "dirty.txt").write_text("uncommitted on eval branch", encoding="utf-8")
    with pytest.raises(InjectionTargetError):
        BranchInjectionTarget.restore(git_repo, original)


@_needs_git
def test_restore_returns_to_original_branch(git_repo: Path) -> None:
    original = _current_branch(git_repo)
    BranchInjectionTarget.prepare(git_repo, "sess-ret")
    subprocess.run(["git", "checkout", "aeh/eval-sess-ret"], cwd=git_repo, check=True, capture_output=True)
    assert _current_branch(git_repo) == "aeh/eval-sess-ret"
    BranchInjectionTarget.restore(git_repo, original)
    assert _current_branch(git_repo) == original


@_needs_git
def test_prepare_raises_on_missing_base_ref(git_repo: Path) -> None:
    with pytest.raises(InjectionTargetError, match="nonexistent-ref"):
        BranchInjectionTarget.prepare(git_repo, "sess-noref", base_ref="nonexistent-ref")
    assert not _branch_exists(git_repo, "aeh/eval-sess-noref")
