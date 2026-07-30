"""Shared fake/stub clients reused across multiple test modules. Not a test module itself."""
from __future__ import annotations

from agent_eval_harness.llm.client import LLMMessage, LLMResponse


class FakeCodeSpectraClient:
    """Duck-typed stand-in for CodeSpectraClient's read_file/get_symbol_edges surface — used by
    expansion and consolidation tests. Pass `edges_map` for pre-built raw edge dicts keyed by file
    path, or `neighbors` for an adjacency list auto-expanded into synthetic call edges (needed to
    simulate multi-hop BFS traversal in expansion tests)."""

    def __init__(
        self,
        files: dict[str, str] | None = None,
        neighbors: dict[str, list[str]] | None = None,
        edges_map: dict[str, dict] | None = None,
        reports: list[dict] | None = None,
        full_reports: dict[str, dict] | None = None,
    ) -> None:
        self._files = files or {}
        self._neighbors = neighbors or {}
        self._edges_map = edges_map or {}
        self._reports = reports or []
        self._full_reports = full_reports or {}
        self.read_calls: list[str] = []
        self.neighbors_calls: list[str] = []

    async def read_file(self, snapshot_id: str, rel_path: str, max_bytes: int = 200_000) -> dict:
        self.read_calls.append(rel_path)
        return {
            "content": self._files.get(rel_path, ""),
            "snapshot_id": snapshot_id,
            "rel_path": rel_path,
            "truncated": False,
        }

    async def get_symbol_edges(self, snapshot_id: str, file_path: str) -> dict:
        self.neighbors_calls.append(file_path)
        if file_path in self._edges_map:
            return self._edges_map[file_path]

        outgoing = []
        for n_path in self._neighbors.get(file_path, []):
            for src_suffix in ("symbolA", "symbolB", "Agent"):
                outgoing.append({
                    "src_symbol": f"{file_path}::{src_suffix}",
                    "dst_symbol": f"{n_path}::symbolB",
                    "edge_type": "call",
                    "confidence_score": 1.0,
                })
        defined = ["symbolA", "symbolB", "Agent"]
        return {"defined_symbols": defined, "outgoing": outgoing, "incoming": []}

    async def get_snapshot(self, snapshot_id: str) -> dict:
        return {
            "id": snapshot_id,
            "local_repo_id": "test_repo",
            "local_path": self._files.get(f"snapshot_local_path_{snapshot_id}", "fake_local_path"),
        }

    async def list_reports(
        self,
        repo_id: str | None = None,
        workspace_id: str | None = None,
        limit: int = 30,
    ) -> list[dict]:
        res = self._reports
        if repo_id:
            res = [r for r in res if r.get("repo_id") == repo_id]
        return res[:limit]

    async def get_report(self, report_id: str) -> dict:
        return self._full_reports.get(report_id, {"id": report_id, "report": {"sections": {}}})


class KeyedFakeLLMClient:
    """Maps the first line of the user prompt (candidate_id-derived class_name) to a canned
    LLMResponse by substring key — used by map-builder golden and robustness tests."""

    def __init__(self, by_key: dict[str, LLMResponse], default: LLMResponse | None = None) -> None:
        self._map = by_key
        self._default = default
        self.calls: list[list[LLMMessage]] = []

    async def complete(
        self, messages, *, max_tokens: int = 512, temperature: float = 0.2, json_mode: bool = False
    ):
        self.calls.append(messages)
        first_line = messages[-1].content.split("\n")[0].strip()

        for key, resp in self._map.items():
            if key in first_line:
                return resp

        if self._default is not None:
            return self._default

        raise KeyError(f"KeyedFakeLLMClient: no canned response for prompt starting {first_line!r}")
