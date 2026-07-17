"""Common utilities shared across dataset generators."""
import json
import logging
from collections.abc import Callable
from typing import Any

from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.store.repository import new_id

logger = logging.getLogger("agent_eval_harness.datasets.generator_utils")


def apply_painpoint(prompt: str, painpoint: str | None) -> str:
    """Append a targeted-adversarial section steering generation toward a user-reported
    painpoint. No painpoint -> byte-identical prompt."""
    if not painpoint:
        return prompt
    return (
        f"{prompt}\n\n"
        f"IMPORTANT — also specifically generate cases probing this real-world "
        f"painpoint a user reported: {painpoint}\n"
        "At least one generated case must exercise this painpoint directly."
    )


def seed_cases_to_dataset_cases(
    dataset_name: str, kind: str, seed_cases: list[dict]
) -> list[DatasetCase]:
    """User-supplied concrete examples land verbatim as `handwritten` (skip review —
    a human already wrote them)."""
    return [
        DatasetCase(
            id=new_id(),
            dataset=dataset_name,
            kind=kind,
            input=sc.get("input", {}),
            expected=sc.get("expected"),
            labels=sc.get("labels"),
            provenance="handwritten",
        )
        for sc in seed_cases
    ]


def strip_markdown_code_block(content: str) -> str:
    """Strip markdown code block markers (```json, ```, etc) from content."""
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return content


def parse_json_with_fallback(
    content: str, fallback_count: int, fallback_fn: Callable[[int], Any]
) -> list[Any]:
    """Parse JSON content with a fallback generator for parse failures."""
    try:
        items = json.loads(content)
        if not isinstance(items, list):
            raise ValueError("Response is not a list")
        return items
    except Exception as e:
        logger.warning(f"parse_json_with_fallback: falling back after parse failure: {e}")
        return [fallback_fn(i) for i in range(fallback_count)]


def build_qa_testset_case(
    dataset_name: str, query: str, answer: str, contexts: Any
) -> DatasetCase:
    """Build a QA testset DatasetCase with standard structure."""
    if not isinstance(contexts, list):
        contexts = [contexts] if contexts else []

    return DatasetCase(
        id=new_id(),
        dataset=dataset_name,
        kind="qa_testset",
        input={"query": query},
        # "answer" kept for existing consumers; "expected_response" is the GroundTruth-reserved key downstream gates read as primary.
        expected={"answer": answer, "expected_response": answer},
        labels={"contexts": contexts},
        provenance="synthetic"
    )
