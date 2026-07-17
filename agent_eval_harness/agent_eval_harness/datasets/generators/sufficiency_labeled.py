import json

from pydantic import BaseModel

from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.store import repository
from agent_eval_harness.store.repository import new_id


class SufficiencyLabeledConfig(BaseModel):
    dataset_name: str
    source_dataset_id: str

async def generate(
    config: dict, llm_client: LLMClient | None, seed: int | None = None
) -> list[DatasetCase]:
    parsed_config = SufficiencyLabeledConfig.model_validate(config)
    
    dataset_name = parsed_config.dataset_name
    source_id = parsed_config.source_dataset_id

    source_db_cases = await repository.get_dataset_cases(source_id)
    if not source_db_cases:
        raise ValueError(f"No source dataset cases found for dataset_id: {source_id}")
        
    cases: list[DatasetCase] = []
    
    for db_case in source_db_cases:
        input_data = json.loads(db_case["input_json"])

        labels_data = {}
        if db_case["labels_json"]:
            labels_data = json.loads(db_case["labels_json"])
            
        query = input_data.get("query", "")
        contexts = labels_data.get("contexts", [])
        
        # 1. Full context (Sufficient)
        cases.append(DatasetCase(
            id=new_id(),
            dataset=dataset_name,
            kind="sufficiency_labeled",
            input={"query": query, "context": contexts},
            expected={"sufficient": True},
            labels={"variant": "full_context", "source_case_id": db_case["id"]},
            provenance="synthetic"
        ))
        
        # 2. No context (Insufficient)
        cases.append(DatasetCase(
            id=new_id(),
            dataset=dataset_name,
            kind="sufficiency_labeled",
            input={"query": query, "context": []},
            expected={"sufficient": False},
            labels={"variant": "no_context", "source_case_id": db_case["id"]},
            provenance="synthetic"
        ))
        
        # 3. Partial context (Ambiguous / judgment call)
        if isinstance(contexts, list) and len(contexts) > 1:
            partial_contexts = contexts[:len(contexts) // 2]
            cases.append(DatasetCase(
                id=new_id(),
                dataset=dataset_name,
                kind="sufficiency_labeled",
                input={"query": query, "context": partial_contexts},
                expected=None,  # leaves expected empty for human judgment
                labels={"variant": "partial_context", "source_case_id": db_case["id"]},
                provenance="synthetic"
            ))
            
    return cases
