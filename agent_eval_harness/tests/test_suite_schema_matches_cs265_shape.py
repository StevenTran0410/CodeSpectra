"""Test that the Suite/SuiteEntry schema can represent every field the evaluation
plan spec's example YAML uses, by round-tripping it through the pydantic models.
"""
from __future__ import annotations

import textwrap

import yaml

CS265_EXAMPLE_YAML = textwrap.dedent("""\
    entries:
      - id: planner.decomposition
        component: planner
        metric: geval.decomposition_coverage
        metric_class: llm_judge
        dataset:
          ref: t2_decomp_v2
        params:
          rubric: decomposition_v1
        rationale: "role=orchestrator => decomposition is this component's core contract"
        provenance: rule

      - id: guard_rule.classifier
        component: guard_rule
        metric: classifier.guard_accuracy
        metric_class: classifier
        dataset:
          ref: t2_guard_v1
        params:
          entry_point: "test_targets.multi_agent.components:GuardComponent"
        rationale: "role=input_guard.rule => deterministic rule-based classification"
        provenance: human_added

      - id: writer.faithfulness
        component: writer
        metric: ragas.faithfulness
        metric_class: llm_judge
        params:
          queries:
            - "What is the company vacation policy?"
        rationale: "role=writer => faithfulness is the core correctness property"
        provenance: rule
""")


def test_suite_schema_accepts_cs265_example() -> None:
    """All fields from the example YAML must parse without error."""
    from agent_eval_harness.metrics.suite import Suite

    data = yaml.safe_load(CS265_EXAMPLE_YAML)
    suite = Suite.model_validate(data)

    assert len(suite.entries) == 3


def test_suite_entry_has_all_cs265_fields() -> None:
    """Each SuiteEntry must carry id, component, metric, metric_class, dataset, params,
    rationale, and provenance."""
    from agent_eval_harness.metrics.suite import SuiteEntry

    entry = SuiteEntry(
        id="planner.decomposition",
        component="planner",
        metric="geval.decomposition_coverage",
        metric_class="llm_judge",
        dataset={"ref": "t2_decomp_v2"},
        params={"rubric": "decomposition_v1"},
        rationale="role=orchestrator => decomposition is this component's core contract",
        provenance="rule",
    )

    assert entry.id == "planner.decomposition"
    assert entry.component == "planner"
    assert entry.metric == "geval.decomposition_coverage"
    assert entry.metric_class == "llm_judge"
    assert entry.dataset is not None
    assert entry.dataset.ref == "t2_decomp_v2"
    assert entry.params["rubric"] == "decomposition_v1"
    assert entry.rationale != ""
    assert entry.provenance == "rule"


def test_suite_provenance_values() -> None:
    """provenance accepts exactly rule, human_added, llm_suggested."""
    from agent_eval_harness.metrics.suite import SuiteEntry

    for prov in ("rule", "human_added", "llm_suggested"):
        entry = SuiteEntry(
            id=f"test.{prov}",
            component="writer",
            metric="ragas.faithfulness",
            metric_class="llm_judge",
            provenance=prov,  # type: ignore[arg-type]
        )
        assert entry.provenance == prov


def test_suite_metric_class_values() -> None:
    """metric_class accepts exactly assertion, classifier, llm_judge."""
    from agent_eval_harness.metrics.suite import SuiteEntry

    for mc in ("assertion", "classifier", "llm_judge"):
        entry = SuiteEntry(
            id=f"test.{mc}",
            component="planner",
            metric="test_metric",
            metric_class=mc,  # type: ignore[arg-type]
        )
        assert entry.metric_class == mc


def test_load_suite_from_yaml_file(tmp_path) -> None:
    """load_suite() must accept both a dict-with-entries and a bare list."""
    from agent_eval_harness.metrics.suite import load_suite

    # Test dict form
    suite_file = tmp_path / "suite.yaml"
    suite_file.write_text(CS265_EXAMPLE_YAML, encoding="utf-8")
    suite = load_suite(suite_file)
    assert len(suite.entries) == 3

    # Test bare-list form
    list_yaml = textwrap.dedent("""\
        - id: test.entry
          component: writer
          metric: ragas.faithfulness
          metric_class: llm_judge
    """)
    list_file = tmp_path / "list_suite.yaml"
    list_file.write_text(list_yaml, encoding="utf-8")
    suite2 = load_suite(list_file)
    assert len(suite2.entries) == 1
