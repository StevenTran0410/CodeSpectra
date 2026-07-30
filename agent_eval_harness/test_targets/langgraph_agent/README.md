# T-LG — langgraph_agent

AEH's LangGraph synthetic test target (CS-312 §5). A `StateGraph` wiring the two
hard idioms Haystack fixtures never exercise:

- **2 function nodes** (`load_context`, `plan_step`) — plain module-level async
  functions passed to `add_node`, one with a real typed kwarg (`max_steps: int`).
- **1 class with 2 bound-method nodes** (`ResearchAgent._node_investigate`,
  `._node_synthesize`) — `add_node("x", self._node_*)`, owner class preserved.
- **Router + `add_conditional_edges`** (`_route`) with a dict path-map.
- **Sentinel edges** `add_edge(START, ...)` / `add_edge(..., END)`.

Scanner-only fixture: Stage 1-3 harvest is pure AST — the graph is never compiled
or run by any test. Mirrors the shape of a real bound-method LangGraph agent
without any dependency on a specific application's node names.

## Usage

```python
from agent_eval_harness.mapping.builder.scanners import LangGraphScanner

files = sorted((TARGETS_DIR / "langgraph_agent").glob("*.py"))
candidates = LangGraphScanner().scan(files)  # -> 4 candidates
```
