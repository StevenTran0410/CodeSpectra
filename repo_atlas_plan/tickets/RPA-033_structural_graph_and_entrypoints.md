# RPA-033 - Build the Structural Graph: Imports, Boundaries, Entrypoints, and Module Relations

* **Epic**: Intelligence
* **Priority**: P1
* **Relative Estimate**: L
* **Milestone**: Milestone D
* **Dependencies**: RPA-031, RPA-032

## 1) Goal

Build a graph/relationship layer strong enough for the app to infer architecture, boundaries, and important files instead of merely listing symbols.

## 2) Problem this ticket solves

A repo map only tells you “what exists.” To understand “what connects to what” and “what forms the structural backbone,” a graph is required.

## 3) Detailed scope

* Implement the file import/dependency graph.
* Detect entrypoint candidates: main/app bootstrap, route index, CLI entry, job runner, framework startup files.
* Detect relations such as controller -> service -> repository heuristically when the language/framework allows it.
* Compute basic metrics: indegree, outdegree, centrality proxy, orphan files, dense clusters.
* Generate graph artifacts used by the architecture overview and important files radar.

## 4) Implementation notes

* Do not try to build a perfect call graph in v1. An import graph plus heuristic relations is enough to create significant value.
* Node types should be normalized: file, symbol, config, entrypoint, external dependency.
* Centrality metrics can be highly useful for the “do not touch this file casually” layer.
* Add graph-backed navigation hooks for retrieval (neighbor expansion / boundary-aware traversal) so RPA-034 can test vectorless retrieval paths without rewriting the core index.
* Future-dev UX note: upgrade graph visualization to graph-DB style circular nodes/edges (interactive canvas) instead of text/link list; prioritize readability for medium-large repos.

## 5) Breakdown subtasks

* Implement import extraction.
* Implement relation heuristics based on naming/path/framework signals.
* Implement the entrypoint detector.
* Implement graph metrics computation.
* Persist graph artifacts and expose them to the analysis layer.

## 6) Acceptance criteria

* A usable graph artifact exists for the priority repositories.
* Entrypoint candidates are detected with demo-level acceptable accuracy.
* Important file scoring can use graph metrics as input signals.
* The architecture section is based on real data, not just LLM guesswork.

## 7) Out of scope

* Deep interprocedural static analysis.
* Full code intelligence comparable to an IDE/LSP platform.

## 8) Risks / watchpoints

* If heuristics are too aggressive, the graph will become wrong and produce false confidence.
* Framework-specific detection can expand scope very quickly.

## 9) Expected deliverables

* Graph builder
* Entrypoint detector
* Graph metrics artifacts

## 10) Definition of done

* Related code/service/UI work has been merged and runs in the local environment.
* The ticket has at least minimal tests or an appropriate verification checklist.
* Logging and error states are not left blank or unhandled.
* Related docs/settings have been updated.
* No major ambiguity remains open without being clearly documented.

## 11) Suggested QA checklist

* Re-run this ticket against at least 1 public repository or 1 internal sample repository.
* Check empty states, error states, and cancel/retry states where applicable.
* Restart the app and return to the flow to verify that the data remains correct.
* Verify that logs do not expose secrets or tokens.
