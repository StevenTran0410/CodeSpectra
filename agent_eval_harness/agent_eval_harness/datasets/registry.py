from collections.abc import Callable, Coroutine
from typing import Any

from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.llm.client import LLMClient

GeneratorFn = Callable[..., Coroutine[Any, Any, list[DatasetCase]]]

def get_generator(kind: str) -> GeneratorFn:
    if kind == "guard_classification":
        from agent_eval_harness.datasets.generators.guard_classification import generate
        return generate
    elif kind == "qa_testset":
        from agent_eval_harness.datasets.generators.qa_testset import generate
        return generate
    elif kind == "decomposition_gold":
        from agent_eval_harness.datasets.generators.decomposition_gold import generate
        return generate
    elif kind == "sufficiency_labeled":
        from agent_eval_harness.datasets.generators.sufficiency_labeled import generate
        return generate
    elif kind == "snapshot_fixture":
        from agent_eval_harness.datasets.generators.snapshot_fixture import generate
        return generate
    elif kind == "snapshot_regression_baseline":
        from agent_eval_harness.datasets.generators.snapshot_regression_baseline import generate
        return generate
    elif kind == "field_match_gold":
        from agent_eval_harness.datasets.generators.field_match_gold import generate
        return generate
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")
