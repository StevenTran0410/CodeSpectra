"""CS-302 AC2 — the single most valuable item: every metric in METRIC_REGISTRY must resolve a
real scorer, so a metric can never be "registered without a run-path" again (the f98a43f /
list_field_prf1 failure shape: registry says yes, run-path says no). RED the moment a scorer-less
metric is added."""
from __future__ import annotations

from agent_eval_harness.metrics.assertions.registry import ASSERTIONS, _import_all
from agent_eval_harness.metrics.registry import METRIC_REGISTRY, MetricSpec

# The judge handlers sweep._dispatch_judge actually knows how to run.
_KNOWN_JUDGE_HANDLERS = {"geval", "ragas_faithfulness", "ragas_answer_relevancy", "tool_correctness"}


def _unresolvable_metrics(registry: dict[str, MetricSpec]) -> list[str]:
    """Metrics with no runnable scorer, by metric_class: assertions must be in the ASSERTIONS
    dict; llm_judges must carry a dispatch handler sweep recognizes."""
    if not ASSERTIONS:
        _import_all()
    unresolvable: list[str] = []
    for name, spec in registry.items():
        if spec.metric_class == "assertion":
            if name not in ASSERTIONS:
                unresolvable.append(name)
        elif spec.metric_class == "llm_judge":
            if spec.dispatch not in _KNOWN_JUDGE_HANDLERS:
                unresolvable.append(name)
        # classifier entries are scored via the dataset+entry_point path, no registry scorer.
    return unresolvable


def test_every_registry_metric_resolves_a_scorer() -> None:
    assert _unresolvable_metrics(METRIC_REGISTRY) == []


def test_gate_goes_red_when_a_scorerless_assertion_is_added() -> None:
    """Mutation proof (teeth): a registered-but-unscored assertion is flagged."""
    mutated = dict(METRIC_REGISTRY)
    mutated["__mutation_no_scorer__"] = MetricSpec(metric_class="assertion")
    assert "__mutation_no_scorer__" in _unresolvable_metrics(mutated)


def test_gate_goes_red_when_a_judge_has_unknown_dispatch() -> None:
    """Mutation proof: an llm_judge whose dispatch handler sweep cannot run is flagged."""
    mutated = dict(METRIC_REGISTRY)
    mutated["__mutation_bad_judge__"] = MetricSpec(
        metric_class="llm_judge", dispatch="not_a_real_handler"
    )
    assert "__mutation_bad_judge__" in _unresolvable_metrics(mutated)
