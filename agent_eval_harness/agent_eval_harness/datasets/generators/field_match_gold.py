"""field_match_gold: objective per-field gold for repo-input agents (repo_name, tech_stack,
runtime_type — never subjective fields), derived from the snapshot's own working tree on
disk rather than a rendered report; provenance starts `synthetic` until reviewed."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.store.repository import new_id

_TECH_MARKERS = {
    "package.json": "node",
    "pyproject.toml": "python",
    "requirements.txt": "python",
    "go.mod": "go",
    "Cargo.toml": "rust",
    "pom.xml": "java",
    "build.gradle": "java",
    "build.gradle.kts": "java",
    "Gemfile": "ruby",
    "composer.json": "php",
}


class FieldMatchGoldConfig(BaseModel):
    dataset_name: str
    snapshot_ids: list[str] | None = None
    local_paths: dict[str, str] | None = None
    snapshot_id: str | None = None
    local_path: str | None = None


async def generate(
    config: dict, llm_client: LLMClient | None, seed: int | None = None
) -> list[DatasetCase]:
    parsed_config = FieldMatchGoldConfig.model_validate(config)
    if parsed_config.snapshot_ids is not None and parsed_config.local_paths is not None:
        snapshot_ids = parsed_config.snapshot_ids
        local_paths = parsed_config.local_paths
    elif parsed_config.snapshot_id is not None and parsed_config.local_path is not None:
        snapshot_ids = [parsed_config.snapshot_id]
        local_paths = {parsed_config.snapshot_id: parsed_config.local_path}
    else:
        snapshot_ids = []
        local_paths = {}

    cases = []
    for snapshot_id in snapshot_ids:
        local_path = local_paths.get(snapshot_id)
        if not local_path:
            continue
        root = Path(local_path)

        field_paths: dict[str, Any] = {"repo_name": root.name}
        tech_stack = sorted({tech for marker, tech in _TECH_MARKERS.items() if (root / marker).exists()})
        if tech_stack:
            field_paths["tech_stack"] = tech_stack

        cases.append(
            DatasetCase(
                id=new_id(),
                dataset=parsed_config.dataset_name,
                kind="field_match_gold",
                input={"shape": "kwargs", "kwargs": {"snapshot_id": snapshot_id}},
                expected={"field_paths": field_paths},
                labels=None,
                provenance="synthetic",
            )
        )
    return cases
