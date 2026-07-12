"""recorded_report_replay: replays recorded analysis reports sections as test cases."""
from __future__ import annotations

from pydantic import BaseModel

from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.store.repository import new_id


class RecordedReportReplayConfig(BaseModel):
    dataset_name: str
    reports: list[dict]


async def generate(
    config: dict, llm_client: LLMClient | None, seed: int | None = None
) -> list[DatasetCase]:
    parsed_config = RecordedReportReplayConfig.model_validate(config)
    cases = []
    for report_entry in parsed_config.reports:
        # report_entry carries {"report_id": ..., "snapshot_id": ..., "sections": {...}}
        report_id = report_entry.get("report_id")
        snapshot_id = report_entry.get("snapshot_id")
        sections = report_entry.get("sections")
        cases.append(
            DatasetCase(
                id=new_id(),
                dataset=parsed_config.dataset_name,
                kind="recorded_report_replay",
                input={
                    "shape": "all_sections",
                    "all_sections": sections,
                },
                expected=None,
                labels={
                    "report_id": report_id,
                    "snapshot_id": snapshot_id,
                },
                provenance="synthetic",
            )
        )
    return cases
