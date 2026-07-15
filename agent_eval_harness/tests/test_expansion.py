import pytest
from agent_eval_harness.llm.client import LLMMessage, LLMResponse
from agent_eval_harness.discovery.expansion import expand_candidate
from tests._stubs import FakeCodeSpectraClient as _StubClient


class _StubLLMClient:
    """Duck-typed stand-in matching the real LLMClient.complete() protocol, not a stale shape."""

    def __init__(self, verdicts: dict[str, str]):
        self._verdicts = verdicts
        self.call_count = 0

    async def complete(self, messages: list[LLMMessage], *, json_mode: bool = False, **_kwargs) -> LLMResponse:
        self.call_count += 1
        prompt = "\n".join(m.content for m in messages)
        import re
        ids = re.findall(r"=== ID: (\S+) ===", prompt)
        
        verdicts_list = []
        for unique_id in ids:
            if "::" in unique_id:
                path, chunk_id = unique_id.split("::", 1)
            else:
                path = unique_id
            
            verdict = self._verdicts.get(unique_id) or self._verdicts.get(path) or "boundary"
            verdicts_list.append({
                "id": unique_id,
                "verdict": verdict,
                "reason": "stubbed"
            })
            
        import json
        content = json.dumps({"verdicts": verdicts_list})
        return LLMResponse(content=content, model="stub")


@pytest.mark.anyio
async def test_expansion_golden_flow() -> None:
    # Golden flow: starts at seed file_a (accept) -> neighbors file_b (expand) -> neighbors file_c (boundary)
    files = {
        "file_a.py": "class Agent: pass",
        "file_b.py": "import file_c",
        "file_c.py": "def utility(): pass"
    }
    neighbors = {
        "file_a.py": ["file_b.py"],
        "file_b.py": ["file_c.py"],
    }
    verdicts = {
        "file_a.py": "expand",
        "file_b.py": "expand",
        "file_c.py": "boundary"
    }

    client = _StubClient(files, neighbors)
    llm_client = _StubLLMClient(verdicts)
    candidate = {
        "cluster_files": ["file_a.py"],
        "hub_paths": []
    }

    res = await expand_candidate("snap-123", candidate, client, llm_client, node_budget=5, hop_cap=3)
    accepted_files = [item["file"] for item in res["accepted"]]
    assert accepted_files == ["file_a.py", "file_b.py"]
    assert res["boundary"] == ["file_c.py"]
    assert res["stop_reason"] == "frontier_exhausted"


@pytest.mark.anyio
async def test_expansion_node_budget() -> None:
    # Low node budget: seed file_a (accept) -> neighbor file_b (accept) with budget of 1
    files = {
        "file_a.py": "class Agent: pass",
        "file_b.py": "import file_a"
    }
    neighbors = {
        "file_a.py": ["file_b.py"]
    }
    verdicts = {
        "file_a.py": "expand",
        "file_b.py": "accept"
    }

    client = _StubClient(files, neighbors)
    llm_client = _StubLLMClient(verdicts)
    candidate = {
        "cluster_files": ["file_a.py"],
        "hub_paths": []
    }

    res = await expand_candidate("snap-123", candidate, client, llm_client, node_budget=1, hop_cap=3)
    accepted_files = [item["file"] for item in res["accepted"]]
    assert accepted_files == ["file_a.py"]
    assert res["stop_reason"] == "node_budget"


@pytest.mark.anyio
async def test_expansion_containment_negative_control() -> None:
    # Negative control: seed file_a is boundary, should not traverse its neighbors
    files = {
        "file_a.py": "class Utility: pass",
        "file_b.py": "class Agent: pass"
    }
    neighbors = {
        "file_a.py": ["file_b.py"]
    }
    verdicts = {
        "file_a.py": "boundary",
        "file_b.py": "accept"
    }

    client = _StubClient(files, neighbors)
    llm_client = _StubLLMClient(verdicts)
    candidate = {
        "cluster_files": ["file_a.py"],
        "hub_paths": []
    }

    res = await expand_candidate("snap-123", candidate, client, llm_client, node_budget=10, hop_cap=3)
    assert res["accepted"] == []
    assert res["boundary"] == ["file_a.py"]
    assert res["stop_reason"] == "frontier_exhausted"
    # Neighbors of boundary should never be called
    assert "file_b.py" not in client.read_calls


@pytest.mark.anyio
async def test_expansion_chunk_level_evidence_and_bypass() -> None:
    files = {
        "file_a.py": "class Agent: pass\ndef run(): pass",
        "file_b.py": "def helper(): pass",
    }
    neighbors = {
        "file_a.py": ["file_b.py"]
    }
    verdicts = {
        "file_b.py": "accept"
    }

    client = _StubClient(files, neighbors)
    llm_client = _StubLLMClient(verdicts)

    candidate = {
        "cluster_files": ["file_a.py"],
        "hub_paths": [],
        "evidence": [
            {
                "file": "file_a.py",
                "chunk_id": "Agent",
                "snippet": "class Agent: pass"
            }
        ],
        "wiring_block": {
            "nodes": [
                {"alias": "agent", "class_name": "Agent", "source_hint_file": "file_a.py"}
            ],
            "edges": []
        }
    }

    res = await expand_candidate("snap-123", candidate, client, llm_client, node_budget=5, hop_cap=3)
    accepted_files = [item["file"] for item in res["accepted"]]
    assert "file_a.py" in accepted_files
    assert "file_b.py" in accepted_files
    assert res["stop_reason"] == "frontier_exhausted"


@pytest.mark.anyio
async def test_expansion_respects_excluded_files() -> None:
    files = {
        "file_a.py": "class Agent: pass",
        "file_b.py": "def helper(): pass",
    }
    neighbors = {
        "file_a.py": ["file_b.py"]
    }
    verdicts = {
        "file_b.py": "accept"
    }

    client = _StubClient(files, neighbors)
    llm_client = _StubLLMClient(verdicts)

    candidate = {
        "cluster_files": ["file_a.py"],
        "hub_paths": [],
        "evidence": [
            {
                "file": "file_a.py",
                "chunk_id": "Agent",
                "snippet": "class Agent: pass"
            }
        ],
        "excluded_files": ["file_a.py"]
    }

    res = await expand_candidate("snap-123", candidate, client, llm_client, node_budget=5, hop_cap=3)
    accepted_files = [item["file"] for item in res["accepted"]]
    assert "file_a.py" not in accepted_files
    assert "file_b.py" not in accepted_files
    assert res["stop_reason"] == "frontier_exhausted"


@pytest.mark.anyio
async def test_expansion_captures_accepted_edges() -> None:
    files = {
        "file_a.py": "class Agent: pass",
        "file_b.py": "def helper(): pass",
        "file_c.py": "def database(): pass",
    }
    # Chain: file_a -> file_b -> file_c
    neighbors = {
        "file_a.py": ["file_b.py"],
        "file_b.py": ["file_c.py"],
    }
    verdicts = {
        "file_a.py": "expand",
        "file_b.py": "expand",
        "file_c.py": "accept",
    }

    client = _StubClient(files, neighbors)
    llm_client = _StubLLMClient(verdicts)

    candidate = {
        "cluster_files": ["file_a.py"],
        "hub_paths": [],
        "evidence": [
            {
                "file": "file_a.py",
                "chunk_id": "Agent",
                "snippet": "class Agent: pass"
            }
        ],
    }

    res = await expand_candidate("snap-123", candidate, client, llm_client, node_budget=5, hop_cap=3)
    accepted_files = [item["file"] for item in res["accepted"]]
    assert accepted_files == ["file_a.py", "file_b.py", "file_c.py"]

    edges = res["accepted_edges"]
    assert len(edges) == 2
    assert {"src": "file_a.py", "dst": "file_b.py"} in edges
    assert {"src": "file_b.py", "dst": "file_c.py"} in edges


@pytest.mark.anyio
async def test_expansion_batch_classification_fewer_calls() -> None:
    # 3 files in one frontier level (seed level)
    files = {
        "file_a.py": "class Agent: pass",
        "file_b.py": "class Tools: pass",
        "file_c.py": "class Router: pass",
    }
    neighbors = {}
    verdicts = {
        "file_a.py::None": "accept",
        "file_b.py::None": "accept",
        "file_c.py::None": "expand",
    }

    client = _StubClient(files, neighbors)
    llm_client = _StubLLMClient(verdicts)

    candidate = {
        "cluster_files": ["file_a.py", "file_b.py", "file_c.py"],
        "hub_paths": [],
        "evidence": [],
    }

    # All three files are in seeds, so they are processed in a single batch level
    res = await expand_candidate("snap-123", candidate, client, llm_client, node_budget=5, hop_cap=3)
    accepted_files = sorted([item["file"] for item in res["accepted"]])
    assert accepted_files == ["file_a.py", "file_b.py", "file_c.py"]
    # Verify that exactly 1 LLM call was made for this first batch level
    assert llm_client.call_count == 1


@pytest.mark.anyio
async def test_expansion_batch_classification_missing_degrades_to_boundary() -> None:
    files = {
        "file_a.py": "class Agent: pass",
        "file_b.py": "class Tools: pass",
        "file_c.py": "class Router: pass",
    }
    neighbors = {}
    
    verdicts = {
        "file_a.py::None": "accept",
        "file_c.py::None": "accept",
    }

    client = _StubClient(files, neighbors)
    class OmissionLLMClient:
        def __init__(self):
            self.call_count = 0

        async def complete(self, messages: list[LLMMessage], *, json_mode: bool = False, **_kwargs) -> LLMResponse:
            self.call_count += 1
            import json
            content = json.dumps({
                "verdicts": [
                    {"id": "file_a.py::None", "verdict": "accept", "reason": "stubbed"},
                    {"id": "file_c.py::None", "verdict": "accept", "reason": "stubbed"},
                ]
            })
            return LLMResponse(content=content, model="stub")

    llm_client = OmissionLLMClient()
    candidate = {
        "cluster_files": ["file_a.py", "file_b.py", "file_c.py"],
        "hub_paths": [],
        "evidence": [],
    }

    res = await expand_candidate("snap-123", candidate, client, llm_client, node_budget=5, hop_cap=3)
    accepted_files = [item["file"] for item in res["accepted"]]
    assert "file_a.py" in accepted_files
    assert "file_c.py" in accepted_files
    assert "file_b.py" not in accepted_files
    assert "file_b.py" in res["boundary"]
