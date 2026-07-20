import random
from typing import Any

from pydantic import BaseModel

from agent_eval_harness.datasets.generator_utils import (
    apply_painpoint,
    parse_json_with_fallback,
    strip_markdown_code_block,
)
from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.llm.client import LLMClient, LLMMessage
from agent_eval_harness.mapping.system_map import load_system_map
from agent_eval_harness.store.repository import new_id


class DecompositionGoldConfig(BaseModel):
    dataset_name: str
    system_map_path: str
    component_id: str = "planner"
    count: int = 15  # Total count. Will be divided among clean, rambling, over_limit.
    painpoint: str | None = None

async def generate(
    config: dict, llm_client: LLMClient | None, seed: int | None = None
) -> list[DatasetCase]:
    parsed_config = DecompositionGoldConfig.model_validate(config)

    if seed is not None:
        random.seed(seed)

    system_map = load_system_map(parsed_config.system_map_path)
    component = system_map.component_by_id(parsed_config.component_id)
    if not component:
        raise ValueError(
            f"Component '{parsed_config.component_id}' not found in system map: "
            f"{parsed_config.system_map_path}"
        )

    limit = None
    for c in component.constraints:
        if c.name == "max_items_per_call":
            limit = c.value
            break

    if not llm_client:
        raise ValueError("LLM client is required for decomposition_gold generation")

    dataset_name = parsed_config.dataset_name
    cases: list[DatasetCase] = []

    # Without a mined max_items_per_call, there's no target system's real fan-out limit to
    # test cases against — skip the over_limit category rather than inventing one.
    num_categories = 3 if limit is not None else 2
    per_cat = max(1, parsed_config.count // num_categories)

    intents_count_range = (2, max(2, limit)) if limit is not None else (2, 4)
    prompt_clean = (
        f"Generate exactly {per_cat} unique examples of user queries containing multiple "
        f"distinct intents. Each query must contain between {intents_count_range[0]} and "
        f"{intents_count_range[1]} distinct intents.\n"
        f"Respond ONLY with a JSON list of objects, where each object has 'query' (string) "
        f"and 'intents' (list of strings).\n"
        f"Example output format:\n"
        f'[\n  {{\n    "query": "Book a flight to Paris and reserve a hotel",\n'
        f'    "intents": ["Book a flight to Paris", "Reserve a hotel"]\n  }}\n]\n'
        f"Do not include any Markdown wrap (like ```json) or explanation."
    )

    prompt_rambling = (
        f"Generate exactly {per_cat} unique examples of long, wordy, or rambling user "
        f"queries that contain EXACTLY ONE core intent.\n"
        f"Respond ONLY with a JSON list of objects, where each object has 'query' (string) "
        f"and 'intents' (list of strings containing exactly one item for the condensed intent).\n"
        f"Example output format:\n"
        f'[\n  {{\n    "query": "Hello, if it is not too much trouble, can you please book a '
        f'hotel in Paris?",\n    "intents": ["Book a hotel in Paris"]\n  }}\n]\n'
        f"Do not include any Markdown wrap (like ```json) or explanation."
    )

    prompts = [("clean", prompt_clean), ("rambling", prompt_rambling)]

    # 4. Over-limit Cases (> limit intents) — only meaningful when a real limit exists
    if limit is not None:
        over_limit_count = limit + 1
        prompt_over_limit = (
            f"Generate exactly {per_cat} unique examples of user queries containing exactly "
            f"{over_limit_count} distinct intents.\n"
            f"Respond ONLY with a JSON list of objects, where each object has 'query' (string) "
            f"and 'intents' (list of strings).\n"
            f"Example output format:\n"
            f'[\n  {{\n    "query": "Do A, B, and C",\n'
            f'    "intents": ["Do A", "Do B", "Do C"]\n  }}\n]\n'
            f"Do not include any Markdown wrap (like ```json) or explanation."
        )
        prompts.append(("over_limit", prompt_over_limit))

    for category, prompt in prompts:
        prompt = apply_painpoint(prompt, parsed_config.painpoint)
        resp = await llm_client.complete(
            [LLMMessage(role="user", content=prompt)],
            max_tokens=2048,
            temperature=0.7,
            json_mode=True
        )
        content = resp.content.strip()
        content = strip_markdown_code_block(content)

        def fallback_item(i: int) -> dict[str, Any]:
            if category == "rambling":
                fallback_intents = [f"intent {i}"]
            else:
                fallback_intents = [f"intent {i} A", f"intent {i} B"]
            return {
                "query": f"Mock {category} query {i}",
                "intents": fallback_intents
            }

        items = parse_json_with_fallback(content, per_cat, fallback_item)

        for item in items[:per_cat]:
            query = item.get("query", "")
            intents = item.get("intents", [])
            if not isinstance(intents, list):
                intents = [str(intents)]
            intents = [str(it) for it in intents]

            # "expected_response" is the GroundTruth-reserved key; downstream deterministic gates read it as primary.
            expected: dict[str, Any] = {"intents": intents, "expected_response": intents}

            if limit is not None and len(intents) > limit:
                call_split = [intents[i : i + limit] for i in range(0, len(intents), limit)]
                expected["call_split"] = call_split

            cases.append(DatasetCase(
                id=new_id(),
                dataset=dataset_name,
                kind="decomposition_gold",
                input={"query": query},
                expected=expected,
                labels={"category": category},
                provenance="synthetic"
            ))

    return cases
