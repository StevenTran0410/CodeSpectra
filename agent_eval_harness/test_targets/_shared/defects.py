"""6 planted-defect switches shared by T1/T2 (CS-261 §6).

Env-var driven, read once at build_pipeline() construction — never left to
nondeterministic LLM behavior, so every defect behaves identically under
FakeLLMClient and reproduces exactly in the offline test suite.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class DefectConfig:
    planner_overpack: bool = False
    guard_leak: bool = False
    wrong_tool: bool = False
    judge_rubber_stamp: bool = False
    writer_hallucinate: bool = False
    no_retry: bool = False

    @classmethod
    def from_env(cls) -> DefectConfig:
        return cls(
            planner_overpack=_flag("AEH_DEFECT_PLANNER_OVERPACK"),
            guard_leak=_flag("AEH_DEFECT_GUARD_LEAK"),
            wrong_tool=_flag("AEH_DEFECT_WRONG_TOOL"),
            judge_rubber_stamp=_flag("AEH_DEFECT_JUDGE_RUBBER_STAMP"),
            writer_hallucinate=_flag("AEH_DEFECT_WRITER_HALLUCINATE"),
            no_retry=_flag("AEH_DEFECT_NO_RETRY"),
        )

    def active_names(self) -> list[str]:
        return [name for name, value in vars(self).items() if value]


_HALLUCINATED_CLAIM = " Additionally, the policy guarantees a full refund with no time limit."


def maybe_hallucinate(content: str, defects: DefectConfig) -> str:
    if not defects.writer_hallucinate:
        return content
    return content + _HALLUCINATED_CLAIM
