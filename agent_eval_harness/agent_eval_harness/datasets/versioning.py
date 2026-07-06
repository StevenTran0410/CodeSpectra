import re

from agent_eval_harness.store.repository import list_dataset_ids


async def next_version(base_name: str) -> str:
    """Find the next version string for base_name."""
    dataset_summaries = await list_dataset_ids()
    existing_ids = [d["dataset_id"] for d in dataset_summaries]

    max_version = 0
    # Match base_name_vN
    pattern = re.compile(rf"^{re.escape(base_name)}_v(\d+)$")
    for dataset_id in existing_ids:
        match = pattern.match(dataset_id)
        if match:
            version_num = int(match.group(1))
            if version_num > max_version:
                max_version = version_num

    return f"{base_name}_v{max_version + 1}"
