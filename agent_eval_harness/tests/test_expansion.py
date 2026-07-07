import pytest
from agent_eval_harness.llm.client import LLMMessage, LLMResponse
from agent_eval_harness.discovery.expansion import expand_candidate

class _StubClient:
    def __init__(self, files: dict[str, str], neighbors: dict[str, list[str]]):
        self._files = files
        self._neighbors = neighbors
        self.read_calls = []
        self.neighbors_calls = []

    async def read_file(self, snapshot_id: str, rel_path: str, max_bytes: int = 200_000) -> dict:
        self.read_calls.append(rel_path)
        if rel_path in self._files:
            return {"content": self._files[rel_path], "snapshot_id": snapshot_id, "rel_path": rel_path, "truncated": False}
        return {"content": "", "snapshot_id": snapshot_id, "rel_path": rel_path, "truncated": False}

    async def get_symbol_edges(self, snapshot_id: str, file_path: str) -> dict:
        self.neighbors_calls.append(file_path)
        outgoing = []
        for n_path in self._neighbors.get(file_path, []):
            outgoing.append({
                "src_symbol": f"{file_path}::symbolA",
                "dst_symbol": f"{n_path}::symbolB",
                "edge_type": "call",
                "confidence_score": 1.0,
            })
            outgoing.append({
                "src_symbol": f"{file_path}::symbolB",
                "dst_symbol": f"{n_path}::symbolB",
                "edge_type": "call",
                "confidence_score": 1.0,
            })
            outgoing.append({
                "src_symbol": f"{file_path}::Agent",
                "dst_symbol": f"{n_path}::symbolB",
                "edge_type": "call",
                "confidence_score": 1.0,
            })
        return {
            "defined_symbols": ["symbolA", "symbolB", "Agent"],
            "outgoing": outgoing,
            "incoming": [],
        }


class _StubLLMClient:
    """Duck-typed stand-in matching the real LLMClient.complete() protocol, not a stale shape."""

    def __init__(self, verdicts: dict[str, str]):
        self._verdicts = verdicts

    async def complete(self, messages: list[LLMMessage], *, json_mode: bool = False, **_kwargs) -> LLMResponse:
        prompt = "\n".join(m.content for m in messages)
        for path in self._verdicts:
            if f"File path: {path}" in prompt:
                verdict = self._verdicts[path]
                return LLMResponse(
                    content=f'{{"verdict": "{verdict}", "reason": "stubbed"}}',
                    model="stub"
                )
        return LLMResponse(
            content='{"verdict": "boundary", "reason": "default"}',
            model="stub"
        )


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
    assert res["accepted"] == ["file_a.py", "file_b.py"]
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
    assert res["accepted"] == ["file_a.py"]
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
    assert "file_a.py" in res["accepted"]
    assert "file_b.py" in res["accepted"]
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
    assert "file_a.py" not in res["accepted"]
    assert "file_b.py" not in res["accepted"]
    assert res["stop_reason"] == "frontier_exhausted"
