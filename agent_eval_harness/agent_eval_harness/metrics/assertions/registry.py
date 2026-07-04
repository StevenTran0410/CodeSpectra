"""Assertion function registry (CS-263 §4.1).

Adding a new assertion type = create a new file + @register("name").
Zero changes to the sweep runner required.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_eval_harness.metrics.types import MetricResult

# spans: list[dict] (one trace's rows from get_spans_for_trace)
# component_id: which component this assertion targets
# params: from the System Map constraint's `value` or the suite entry's `params`
AssertionFn = Callable[[list[dict], str, dict], "MetricResult"]

ASSERTIONS: dict[str, AssertionFn] = {}


def register(name: str):
    """Decorator: register an assertion function under the given name."""

    def deco(fn: AssertionFn) -> AssertionFn:
        ASSERTIONS[name] = fn
        return fn

    return deco


def _import_all() -> None:
    """Import all assertion modules to populate ASSERTIONS.

    Called lazily on first use rather than at import time to avoid circular deps.
    """
    from agent_eval_harness.metrics.assertions import (  # noqa: F401
        allowed_downstream,
        arg_schema,
        max_items_per_call,
        max_retries,
        no_unnecessary_calls,
        retry_on_reject_required,
    )


def get_assertion(name: str) -> AssertionFn:
    """Retrieve a registered assertion, auto-importing all modules first."""
    if not ASSERTIONS:
        _import_all()
    if name not in ASSERTIONS:
        raise KeyError(f"Unknown assertion '{name}'. Registered: {sorted(ASSERTIONS)}")
    return ASSERTIONS[name]
