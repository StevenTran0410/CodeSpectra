"""snapshot_fixture: zero-LLM dataset kind for repo-input agents. One case = one full
pipeline run; force_rerun is mandatory because CodeSpectra caches results by
(snapshot_id, model_id, mode)."""
from __future__ import annotations

from pydantic import BaseModel

from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.store.repository import new_id


class SnapshotFixtureConfig(BaseModel):
    dataset_name: str
    snapshot_ids: list[str]
    provider_id: str
    model_id: str


async def generate(
    config: dict, llm_client: LLMClient | None, seed: int | None = None
) -> list[DatasetCase]:
    parsed_config = SnapshotFixtureConfig.model_validate(config)
    dataset_name = parsed_config.dataset_name

    return [
        DatasetCase(
            id=new_id(),
            dataset=dataset_name,
            kind="snapshot_fixture",
            input={
                "shape": "kwargs",
                "kwargs": {
                    "snapshot_id": snapshot_id,
                    "provider_id": parsed_config.provider_id,
                    "model_id": parsed_config.model_id,
                    "force_rerun": True,
                },
            },
            expected=None,
            labels=None,
            provenance="synthetic",
        )
        for snapshot_id in parsed_config.snapshot_ids
    ]
