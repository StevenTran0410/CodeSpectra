"""AEH config — data dir + optional .aeh/config.yaml for live-run backend/provider settings."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# eval cases generated per agent (small MVP default; override via AEH_CASES_PER_AGENT env for full runs).
DEFAULT_CASES_PER_AGENT: int = max(1, int(os.getenv("AEH_CASES_PER_AGENT", "5")))


@dataclass
class ContractConventions:
    """Target-specific strings threaded into contract_harvest so AEH stays generic. Empty/None by
    default -- a target that wants these conventions detected configures them via .aeh/config.yaml;
    absence degrades to "not detected", never a guess."""
    rerun_section_route: str | None = None
    pipeline_fallback_name_pattern: str | None = None
    plan_queries_symbol: str | None = None


@dataclass
class AEHConfig:
    backend_url: str | None = None
    backend_token: str | None = None
    provider_id: str | None = None
    model_id: str | None = None
    reasoning_effort: str | None = None
    thinking_budget: int | None = None
    backend_source_path: str | None = None
    contract_conventions: ContractConventions = field(default_factory=ContractConventions)

    @classmethod
    def load(cls, path: str | Path | None = None) -> AEHConfig:
        config_path = Path(path) if path else Path.cwd() / ".aeh" / "config.yaml"
        if not config_path.exists():
            return cls()
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        cc = data.get("contract_conventions") or {}
        defaults = ContractConventions()
        return cls(
            backend_url=data.get("backend_url"),
            backend_token=data.get("backend_token"),
            provider_id=data.get("provider_id"),
            model_id=data.get("model_id"),
            reasoning_effort=data.get("reasoning_effort"),
            thinking_budget=data.get("thinking_budget"),
            backend_source_path=data.get("backend_source_path"),
            contract_conventions=ContractConventions(
                rerun_section_route=cc.get("rerun_section_route", defaults.rerun_section_route),
                pipeline_fallback_name_pattern=cc.get(
                    "pipeline_fallback_name_pattern", defaults.pipeline_fallback_name_pattern
                ),
                plan_queries_symbol=cc.get("plan_queries_symbol", defaults.plan_queries_symbol),
            ),
        )
