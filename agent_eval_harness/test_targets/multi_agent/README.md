# T2 — multi_agent

AEH's rich synthetic test target (CS-261 §6): a nested, branching topology
deliberately different in shape from T1 — `guard (rule+llm) -> planner
(decomposes intents, fan-out limit 2) -> worker (2 tools, one relevant one
decoy) -> judge (sufficiency verdict, one retry) -> writer`.

## Orchestration note

`PlannerComponent` is the only component besides `guard` scheduled directly on
the Haystack pipeline. It internally calls `worker`/`judge`/`writer` via plain
composition (each wrapped in its own manual span) because Haystack's static DAG
model can't cleanly express "judge rejects -> retry once" as a graph edge. This
matches the epic's own role semantics — CS-260 §3 defines `orchestrator` as
"decomposes intent, routes to sub-agents, **manages retries**," so retry
ownership living in the planner is the intended design, not a workaround.

## Planted defects (all 6, CS-261 §6)

| Switch | Effect |
|---|---|
| `AEH_DEFECT_PLANNER_OVERPACK` | planner ignores the fan-out-2 cap, sends all decomposed intents to worker in one call |
| `AEH_DEFECT_GUARD_LEAK` | guard's LLM stage verdict is forced to `pass` regardless of the (fake) classifier's actual output |
| `AEH_DEFECT_WRONG_TOOL` | worker calls `decoy_lookup` before `case_law_search` |
| `AEH_DEFECT_JUDGE_RUBBER_STAMP` | judge always returns `sufficient=True`, skipping its LLM check entirely |
| `AEH_DEFECT_WRITER_HALLUCINATE` | writer appends a fabricated claim (shared with T1) |
| `AEH_DEFECT_NO_RETRY` | planner ignores a judge rejection and proceeds straight to writer |

## Tier-2 fallback

**Not exercised for T2** — CS-261's Tier-2 acceptance test only requires T1
("pretending it's uninstrumentable"). T2's `entry_point` fields in
`system_map.yaml` are code-reference citations (CS-260 §4c), not tested
tier-2-callable functions.

## Golden map

`system_map.yaml` — 8 components (`guard_rule`, `guard_llm` sharing one
Haystack node disambiguated by the `aeh.check.kind` tag; `planner`; `worker`;
`case_law_search_tool`; `decoy_tool`; `judge`; `writer`), each constraint
carrying a `source:` file:line citation per epic convention.
