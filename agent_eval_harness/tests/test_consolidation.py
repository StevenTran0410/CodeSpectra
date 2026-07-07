import pytest
from unittest.mock import AsyncMock
from agent_eval_harness.discovery.consolidation import consolidate_candidates
from agent_eval_harness.llm.client import RateLimitExceeded, LLMResponse


class _StubClient:
    def __init__(self, edges_map=None, file_contents=None):
        self.edges_map = edges_map or {}
        self.file_contents = file_contents or {}

    async def get_symbol_edges(self, snapshot_id, file_path):
        return self.edges_map.get(file_path, {})

    async def read_file(self, snapshot_id, file_path):
        return {"content": self.file_contents.get(file_path, "")}


@pytest.mark.anyio
async def test_consolidation_clear_import_match() -> None:
    # Candidate 1: has wiring block (>=2 nodes) -> core seed is {"a.py", "b.py"}
    # Candidate 2: has only 1 node in wiring -> falls back to hub_paths {"c.py"}
    candidates = [
        {
            "community_id": 1,
            "name": "Block 1",
            "frameworks": ["haystack"],
            "cluster_files": ["a.py", "b.py", "shared.py"],
            "hub_paths": ["a.py"],
            "wiring_block": {
                "nodes": [
                    {"source_hint_file": "a.py"},
                    {"source_hint_file": "b.py"}
                ],
                "edges": []
            }
        },
        {
            "community_id": 2,
            "name": "Block 2",
            "frameworks": ["langgraph"],
            "cluster_files": ["c.py", "other.py"],
            "hub_paths": ["c.py"],
            "wiring_block": {
                "nodes": [
                    {"source_hint_file": "c.py"}
                ],
                "edges": []
            }
        }
    ]

    # "shared.py" has a symbol edge to "b.py" (Candidate 1's core seed)
    edges_map = {
        "shared.py": {
            "outgoing": [{"src_symbol": "shared.py::helper", "dst_symbol": "b.py::run"}],
            "incoming": []
        }
    }
    client = _StubClient(edges_map=edges_map)
    llm_client = AsyncMock()

    res = await consolidate_candidates(candidates, client, "snap", llm_client)
    assert len(res) == 2
    
    # Candidate 1 should match "shared.py" via import
    c1 = next(c for c in res if c["community_id"] == 1)
    assert "shared.py" in c1["matched_files"]
    assert c1["file_provenance"]["shared.py"] == "import_matched"
    assert c1["seed_anchor_kind"] == "wiring"

    # Candidate 2: "shared.py" does not match since it's import_matched to 1
    c2 = next(c for c in res if c["community_id"] == 2)
    assert "shared.py" not in c2["matched_files"]
    assert c2["seed_anchor_kind"] == "hub"  # only 1 node -> fallback to hub


@pytest.mark.anyio
async def test_consolidation_tied_and_unresolved() -> None:
    # No path or symbol edge signal connecting shared.py to either seed -> stays unresolved
    candidates = [
        {
            "community_id": 1,
            "name": "Block 1",
            "cluster_files": ["a.py", "shared.py"],
            "hub_paths": ["a.py"],
            "wiring_block": None
        },
        {
            "community_id": 2,
            "name": "Block 2",
            "cluster_files": ["b.py", "shared.py"],
            "hub_paths": ["b.py"],
            "wiring_block": None
        }
    ]

    client = _StubClient()  # no edges
    llm_client = AsyncMock()

    res = await consolidate_candidates(candidates, client, "snap", llm_client)
    # shared.py has no signals -> no LLM call is made, remains unmatched
    llm_client.complete.assert_not_called()
    for c in res:
        assert "shared.py" not in c.get("matched_files", [])


@pytest.mark.anyio
async def test_consolidation_llm_judge_resolves_tie() -> None:
    candidates = [
        {
            "community_id": 1,
            "name": "Block 1",
            "frameworks": ["haystack"],
            "cluster_files": ["dir/a.py", "dir/shared.py"],
            "hub_paths": ["dir/a.py"],
        },
        {
            "community_id": 2,
            "name": "Block 2",
            "frameworks": ["langgraph"],
            "cluster_files": ["dir/b.py", "dir/shared.py"],
            "hub_paths": ["dir/b.py"],
        }
    ]

    # dir/shared.py has a tie: path overlap is equal (best=2 for both since directories match)
    client = _StubClient(file_contents={"dir/shared.py": "some content"})
    llm_client = AsyncMock()
    llm_client.complete.return_value = LLMResponse(
        content='{"candidate_id": "1", "confidence": "high"}',
        model="fake"
    )

    res = await consolidate_candidates(candidates, client, "snap", llm_client)
    # llm_client should be called to resolve the tie
    llm_client.complete.assert_called_once()
    c1 = next(c for c in res if c["community_id"] == 1)
    assert "dir/shared.py" in c1["matched_files"]
    assert c1["file_provenance"]["dir/shared.py"] == "llm_linked"


@pytest.mark.anyio
async def test_consolidation_propagates_rate_limit() -> None:
    candidates = [
        {
            "community_id": 1,
            "name": "Block 1",
            "cluster_files": ["dir/a.py", "dir/shared.py"],
            "hub_paths": ["dir/a.py"],
        },
        {
            "community_id": 2,
            "name": "Block 2",
            "cluster_files": ["dir/b.py", "dir/shared.py"],
            "hub_paths": ["dir/b.py"],
        }
    ]

    client = _StubClient()
    llm_client = AsyncMock()
    llm_client.complete.side_effect = RateLimitExceeded("prov", "model")

    with pytest.raises(RateLimitExceeded):
        await consolidate_candidates(candidates, client, "snap", llm_client)
