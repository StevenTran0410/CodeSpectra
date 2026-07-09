"""snapshot_regression_baseline: golden-master via one real pipeline run per fixture case.

NOT YET CALLABLE: capturing a live run requires calling into the target's own execution
surface, and no authenticated path to it exists yet (`CodeSpectraClient` is scoped to
`/api/external/*` and cannot reach the unauthenticated rerun route). `fulfillment.py` marks
any `snapshot_regression_baseline` group `needs_human` and never calls generate() for this
kind — this function exists only so the registry has something to import, and fails loudly
if ever called directly.
"""
from __future__ import annotations

from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.llm.client import LLMClient


async def generate(
    config: dict, llm_client: LLMClient | None, seed: int | None = None
) -> list[DatasetCase]:
    raise NotImplementedError(
        "snapshot_regression_baseline requires a live target-execution seam that "
        "doesn't exist yet (the ingest/injection machinery) — "
        "fulfillment.py must not call this; it marks the group needs_human first."
    )
