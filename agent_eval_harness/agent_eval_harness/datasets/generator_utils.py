"""Common utilities shared across dataset generators."""
import json
from typing import Any

from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.store.repository import new_id


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
    content: str, fallback_count: int, fallback_fn: callable
) -> list[Any]:
    """Parse JSON content with a fallback generator for parse failures."""
    try:
        items = json.loads(content)
        if not isinstance(items, list):
            raise ValueError("Response is not a list")
        return items
    except Exception:
        # Fallback: generate items using the provided function
        return [fallback_fn(i) for i in range(fallback_count)]


def build_qa_testset_case(
    dataset_name: str, query: str, answer: str, contexts: Any
) -> DatasetCase:
    """Build a QA testset DatasetCase with standard structure."""
    # Ensure contexts is a list
    if not isinstance(contexts, list):
        contexts = [contexts] if contexts else []

    return DatasetCase(
        id=new_id(),
        dataset=dataset_name,
        kind="qa_testset",
        input={"query": query},
        expected={"answer": answer},
        labels={"contexts": contexts},
        provenance="synthetic"
    )
