from pathlib import Path

import pytest

from agent_eval_harness.code_injection.plan_renderer import _TEMPLATES_DIR, render_eval_plan_md
from agent_eval_harness.mapping.system_map import SystemMap, Component
from agent_eval_harness.planning.contract import (
    EvaluationContract,
    InvocationContract,
    KwargSpec,
)
from agent_eval_harness.planning.report import (
    AgentDataProfile,
    AgentPlanReport,
    EvaluationGate,
    EvaluationPlanReport,
)

_SM = SystemMap(
    target_system_id="test_system",
    components=[
        Component(
            id="retriever",
            role="retriever",
            entry_point="retriever.Retriever.retrieve",
            file="backend/retriever.py",
        ),
    ],
)
_WIRING = {
    "plan_id": "sess-test",
    "entrypoint": "retriever.Retriever.retrieve",
    "component_ids": ["retriever"],
}
_DS = [
    {
        "dataset_id": "ds_v1",
        "kind": "query_response",
        "case_count": 5,
        "gate_ids": ["retriever.precision.llm0"],
    }
]


def _render() -> str:
    return render_eval_plan_md(_SM, _WIRING, _DS, "sess-test", "aeh/eval-sess-test")


def test_render_contains_real_template_content() -> None:
    result = _render()
    for fname in ("tracer.py", "run_eval.py", "aeh_eval.py"):
        content = (_TEMPLATES_DIR / fname).read_text(encoding="utf-8")
        anchor = content.split("\n")[0][:80]
        assert anchor in result, f"{fname} content not found in rendered plan"


def test_render_contains_wiring_json() -> None:
    result = _render()
    assert '"plan_id": "sess-test"' in result


def test_render_contains_session_and_branch() -> None:
    result = _render()
    assert "sess-test" in result
    assert "aeh/eval-sess-test" in result


def test_render_contains_main_py_diff() -> None:
    result = _render()
    # Verify the main_py_diff.txt content is embedded (check a stable anchor line)
    diff_content = (_TEMPLATES_DIR / "main_py_diff.txt").read_text(encoding="utf-8")
    anchor = diff_content.split("\n")[0][:60]  # first comment line
    assert anchor in result


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _plan_report_fixture() -> EvaluationPlanReport:
    gate = EvaluationGate(
        id="retriever.precision.llm0",
        agent_id="retriever_agent",
        component="retriever",
        location="output",
        metric="assertion.precision",
        metric_class="assertion",
        toolkit="assertion",
        rationale="checks precision",
    )
    return EvaluationPlanReport(
        target_system_id="test_system",
        agents=[
            AgentPlanReport(
                agent_id="retriever_agent",
                role="retriever",
                label="Retriever",
                data_profile=AgentDataProfile(
                    agent_id="retriever_agent",
                    input_data="a query string",
                    output_data="ranked chunks",
                    internal_tools=["BM25Index.search"],
                    failure_modes=["returns empty on typo queries"],
                    consistency_notes=["declared upstream omits the reranker call"],
                ),
                contract=EvaluationContract(
                    agent_id="retriever_agent",
                    component_id="retriever",
                    invocation=InvocationContract(
                        callable="retriever.Retriever",
                        method="retrieve",
                        kwargs=[KwargSpec(name="query", annotation="str")],
                        source="ast",
                        citations=["backend/retriever.py:42"],
                    ),
                ),
                gates=[gate],
                needs_human=["confirm reranker is intentionally excluded"],
            )
        ],
    )


_MAIN_PY_FIXTURE = (
    "import argparse\n"
    "from api.external import router as external_router\n"
    'app.include_router(external_router, prefix="/api/external")\n'
)


def test_render_with_plan_report_includes_per_agent_context(tmp_path: Path) -> None:
    _write(tmp_path / "backend/main.py", _MAIN_PY_FIXTURE)
    _write(tmp_path / "backend/prompts.py", "RETRIEVER_SYSTEM = 'you are...'\n")
    _write(tmp_path / "backend/retriever.py", "from .prompts import RETRIEVER_SYSTEM\n")
    ds = [{**_DS[0], "example_case": {"id": "c1", "input": {"query": "hi"}}}]

    result = render_eval_plan_md(
        _SM, _WIRING, ds, "sess-test", "aeh/eval-sess-test",
        plan_report=_plan_report_fixture(), repo_root=tmp_path,
    )

    assert "### Agent: retriever_agent (role: retriever)" in result
    assert "a query string" in result
    assert "ranked chunks" in result
    assert "BM25Index.search" in result
    assert "returns empty on typo queries" in result
    assert "declared upstream omits the reranker call" in result
    assert "backend/prompts.py" in result and "RETRIEVER_SYSTEM" in result
    assert "retriever.Retriever.retrieve" in result
    assert "backend/retriever.py:42" in result
    assert "confirm reranker is intentionally excluded" in result
    assert '"id": "c1"' in result
    assert "## Step 0: Create and checkout the eval branch" in result
    assert "git checkout -b aeh/eval-sess-test main" in result


def test_render_step0_uses_the_given_base_ref(tmp_path: Path) -> None:
    _write(tmp_path / "backend/main.py", _MAIN_PY_FIXTURE)
    _write(tmp_path / "backend/retriever.py", "import os\n")

    result = render_eval_plan_md(
        _SM, _WIRING, _DS, "sess-test", "aeh/eval-sess-test",
        plan_report=_plan_report_fixture(), repo_root=tmp_path, base_ref="develop",
    )

    assert "git checkout -b aeh/eval-sess-test develop" in result


def test_render_with_plan_report_shows_fallback_when_prompt_not_found(tmp_path: Path) -> None:
    _write(tmp_path / "backend/main.py", _MAIN_PY_FIXTURE)
    _write(tmp_path / "backend/retriever.py", "import os\n")

    result = render_eval_plan_md(
        _SM, _WIRING, _DS, "sess-test", "aeh/eval-sess-test",
        plan_report=_plan_report_fixture(), repo_root=tmp_path,
    )

    assert "not auto-detected" in result
    assert "search `backend/retriever.py`" in result


def test_render_raises_when_main_py_missing_expected_anchors(tmp_path: Path) -> None:
    _write(tmp_path / "backend/main.py", "print('not the real main.py')\n")

    with pytest.raises(ValueError, match="anchor line"):
        render_eval_plan_md(
            _SM, _WIRING, _DS, "sess-test", "aeh/eval-sess-test",
            plan_report=_plan_report_fixture(), repo_root=tmp_path,
        )


def test_render_raises_when_main_py_file_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="backend/main.py not found"):
        render_eval_plan_md(
            _SM, _WIRING, _DS, "sess-test", "aeh/eval-sess-test",
            plan_report=_plan_report_fixture(), repo_root=tmp_path,
        )


def test_render_raises_when_plan_report_given_without_repo_root() -> None:
    with pytest.raises(ValueError, match="repo_root"):
        render_eval_plan_md(
            _SM, _WIRING, _DS, "sess-test", "aeh/eval-sess-test",
            plan_report=_plan_report_fixture(), repo_root=None,
        )
