from pathlib import Path

from agent_eval_harness.datasets.types import DatasetCase


def write_jsonl(cases: list[DatasetCase], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(case.model_dump_json() + "\n")

def read_jsonl(path: str | Path) -> list[DatasetCase]:
    cases = []
    p = Path(path)
    if not p.exists():
        return cases
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(DatasetCase.model_validate_json(line))
    return cases
