import importlib
from collections.abc import Callable, Coroutine
from typing import Any

from agent_eval_harness.datasets.types import DatasetCase

GeneratorFn = Callable[..., Coroutine[Any, Any, list[DatasetCase]]]

# Plugin registry: dataset kind -> its generator module. Modules are imported lazily so a
# generator's optional dependencies (e.g. deepeval, ragas) are never required unless it's used.
_GENERATOR_MODULES: dict[str, str] = {
    "guard_classification": "agent_eval_harness.datasets.generators.guard_classification",
    "qa_testset": "agent_eval_harness.datasets.generators.qa_testset",
    "decomposition_gold": "agent_eval_harness.datasets.generators.decomposition_gold",
    "sufficiency_labeled": "agent_eval_harness.datasets.generators.sufficiency_labeled",
    "snapshot_fixture": "agent_eval_harness.datasets.generators.snapshot_fixture",
    "snapshot_regression_baseline": "agent_eval_harness.datasets.generators.snapshot_regression_baseline",
    "field_match_gold": "agent_eval_harness.datasets.generators.field_match_gold",
    "recorded_report_replay": "agent_eval_harness.datasets.generators.recorded_report_replay",
    "synthetic_agent_io": "agent_eval_harness.datasets.generators.synthetic_agent_io",
    "metamorphic_relation": "agent_eval_harness.datasets.generators.metamorphic_relation",
}


def get_generator(kind: str) -> GeneratorFn:
    module_path = _GENERATOR_MODULES.get(kind)
    if module_path is None:
        raise ValueError(f"Unknown dataset kind: {kind}")
    return importlib.import_module(module_path).generate
