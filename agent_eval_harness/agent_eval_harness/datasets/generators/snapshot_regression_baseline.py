"""CS-282 §2 — snapshot_regression_baseline: golden-master via one real pipeline run
per fixture case.

NOT YET CALLABLE: capturing a live run requires calling into the target's own
execution surface, and no authenticated path to it exists yet — `/api/analysis/
rerun_section` is mounted without the `/api/external` token gate (CodeSpectra
backend/main.py:99), so `CodeSpectraClient` (scoped to `/api/external/*`,
discovery/client.py) cannot reach it, and AEH has no other route to the target's
execution. That seam is exactly CS-283/284's job (ingest/injection); a one-off HTTP
call here would duplicate it, uncoordinated. `fulfillment.py` marks any
`snapshot_regression_baseline` group `needs_human` with this reason and never calls
generate() for this kind — this function exists only so `datasets/registry.py`'s
`get_generator()` has something to import, and fails loudly if ever called directly.
"""
from __future__ import annotations

from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.llm.client import LLMClient


async def generate(
    config: dict, llm_client: LLMClient | None, seed: int | None = None
) -> list[DatasetCase]:
    raise NotImplementedError(
        "snapshot_regression_baseline requires a live target-execution seam that "
        "doesn't exist yet (CS-283/284's ingest/injection machinery) — "
        "fulfillment.py must not call this; it marks the group needs_human first."
    )
