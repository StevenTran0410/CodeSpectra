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
    snapshot_id: str
    local_path: str  # root of the cloned snapshot on disk


async def generate(
    config: dict, llm_client: LLMClient | None, seed: int | None = None
) -> list[DatasetCase]:
    parsed_config = FieldMatchGoldConfig.model_validate(config)
    root = Path(parsed_config.local_path)

    field_paths: dict[str, Any] = {"repo_name": root.name}
    tech_stack = sorted({tech for marker, tech in _TECH_MARKERS.items() if (root / marker).exists()})
    if tech_stack:
        field_paths["tech_stack"] = tech_stack

    return [
        DatasetCase(
            id=new_id(),
            dataset=parsed_config.dataset_name,
            kind="field_match_gold",
            input={"shape": "kwargs", "kwargs": {"snapshot_id": parsed_config.snapshot_id}},
            expected={"field_paths": field_paths},
            labels=None,
            provenance="synthetic",
        )
    ]
