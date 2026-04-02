# 09 — Job Orchestration Design

---

## Responsibilities

The job orchestration layer (RPA-012) is responsible for:

1. Accepting an analysis job request (repo + provider + scan mode).
2. Managing job lifecycle state transitions (see `03_analysis_run_state_machine.md`).
3. Running pipeline stages sequentially, with checkpoints.
4. Emitting real-time progress events consumable by the frontend.
5. Supporting cooperative cancellation.
6. Persisting the completed `ReportArtifact`.

---

## Job creation flow

```
Frontend                     Electron IPC              Python Backend
   │                              │                          │
   │  analysis:start(req)  ──────►│                          │
   │                              │  POST /api/jobs/         │
   │                              │─────────────────────────►│
   │                              │  { job_id, status: "pending" }
   │◄─────────────────────────────│◄─────────────────────────│
   │                              │                          │
   │  subscribe to SSE ───────────┼─────────────────────────►│ /api/jobs/{id}/events
```

---

## Pipeline executor

The executor is a background `asyncio` task spawned per job. Each stage is a `PipelineStage` with:

```python
class PipelineStage(Protocol):
    name: str
    async def run(self, ctx: JobContext, cancel: asyncio.Event) -> StageResult: ...
```

`JobContext` carries:
- `repo: LocalRepo`
- `provider: ProviderConfig`
- `scan_mode: Literal["quick", "full"]`
- `artifact_builder: ReportArtifactBuilder`  ← accumulates section outputs
- `db: aiosqlite.Connection`

Cancellation check pattern inside a stage:

```python
for batch in batches:
    if cancel.is_set():
        raise CancelledError()
    await process(batch)
```

---

## Progress event shape

```json
{
  "event": "progress",
  "job_id": "abc123",
  "stage": "symbol_extract",
  "stage_index": 2,
  "total_stages": 7,
  "pct": 48,
  "message": "Parsing 612 / 1,270 files..."
}
```

Terminal events:

```json
{ "event": "done",      "job_id": "...", "report_id": "..." }
{ "event": "failed",    "job_id": "...", "error": "..." }
{ "event": "cancelled", "job_id": "..." }
```

---

## Concurrency

- **v1:** One active job at a time per app instance (simplest; avoids resource contention).
- Attempting to start a second job while one is running returns HTTP 409 with a clear message.
- Future: per-workspace concurrency limit.

---

## Stage registry (v1 full scan)

| Index | Stage name | Description |
|---|---|---|
| 0 | `preflight` | Validate repo path, branch checkout, provider reachability |
| 1 | `manifest` | File listing, language detection, ignore filtering |
| 2 | `symbol_extract` | Tree-sitter parsing, symbol index |
| 3 | `graph_build` | Import graph, entrypoint detection |
| 4 | `retrieval_prep` | Chunking, optional embedding |
| 5 | `generation` | LLM section generation (parallelized by section) |
| 6 | `report_write` | Assemble and persist `ReportArtifact` |

**Quick scan** skips stages 3 and 4; generation uses a reduced context window.

---

## Error handling

- Each stage wraps its work in `try/except`; unhandled exceptions transition the job to `failed`.
- Partial results from completed stages are **not** saved to the report artifact if a later stage fails — the entire artifact is either complete or absent.
- The error message is stored on the `AnalysisJob` record and surfaced to the frontend.
