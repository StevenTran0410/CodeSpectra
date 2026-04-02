# 03 — Analysis Run State Machine

---

## States

```
PENDING ──► RUNNING ──► DONE
                │
                ├──► FAILED
                │
                └──► CANCELLED  (user-initiated)
```

| State | Meaning |
|---|---|
| `pending` | Job created, waiting for executor slot |
| `running` | Actively indexing / generating |
| `done` | Report artifact available |
| `failed` | Terminal error; error message stored |
| `cancelled` | User cancelled while running or pending |

---

## Transitions

| From | To | Trigger |
|---|---|---|
| — | `pending` | User starts analysis run |
| `pending` | `running` | Executor picks up job |
| `pending` | `cancelled` | User cancels before execution starts |
| `running` | `done` | All pipeline stages complete successfully |
| `running` | `failed` | Unrecoverable error in any pipeline stage |
| `running` | `cancelled` | User cancels during execution (graceful stop) |
| `failed` | `pending` | User retries (creates new job, old job stays as `failed`) |

---

## Pipeline stages (inside `running`)

```
1. preflight       — disk space, branch checkout, provider reachability
2. manifest        — file listing, language detection, ignore filtering
3. symbol_extract  — Tree-sitter parsing, symbol index
4. graph_build     — import graph, entrypoint detection
5. retrieval_prep  — chunking, optional embedding
6. generation      — LLM calls per section (parallel where safe)
7. report_write    — assemble ReportArtifact, write to DB
```

Each stage emits progress events consumed by the UI progress panel (RPA-035).

---

## Cancellation contract

- Cancellation is **cooperative**: each stage checks a cancellation token between steps.
- On cancellation, any partially written data is rolled back or marked as incomplete.
- The job record transitions to `cancelled` with a `cancelled_at` timestamp.
- No orphaned temporary files are left on disk after cancellation cleanup.

---

## Resume / retry semantics

- A `failed` or `cancelled` job is **not resumed** — a new job is created.
- Completed stages are **not reused** across jobs (no incremental caching in v1).
- The old job record is kept for audit and comparison purposes.

---

## Progress reporting shape

```json
{
  "job_id": "...",
  "stage": "symbol_extract",
  "stage_index": 3,
  "total_stages": 7,
  "pct": 42,
  "message": "Parsing 1,204 Python files..."
}
```

Emitted via Server-Sent Events or WebSocket (decision deferred to RPA-012).
