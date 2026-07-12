"""AEH config — data dir + optional .aeh/config.yaml for live-run backend/provider settings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class AEHConfig:
    backend_url: str | None = None
    backend_token: str | None = None
    provider_id: str | None = None
    model_id: str | None = None
    reasoning_effort: str | None = None
    thinking_budget: int | None = None
    backend_source_path: str | None = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> AEHConfig:
        config_path = Path(path) if path else Path.cwd() / ".aeh" / "config.yaml"
        if not config_path.exists():
            return cls()
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return cls(
            backend_url=data.get("backend_url"),
            backend_token=data.get("backend_token"),
            provider_id=data.get("provider_id"),
            model_id=data.get("model_id"),
            reasoning_effort=data.get("reasoning_effort"),
            thinking_budget=data.get("thinking_budget"),
            backend_source_path=data.get("backend_source_path"),
        )
