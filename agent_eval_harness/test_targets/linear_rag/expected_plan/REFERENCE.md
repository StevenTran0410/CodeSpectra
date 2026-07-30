# REFERENCE — System Overview & Per-Agent Detail

## Target System

**ID**: linear_rag
**Branch**: aeh/eval-linear_rag

| Property | Value |
| --- | --- |
| Framework | unknown |
| Agents | 2 |
| Roles | retrieval_agent×1, writer×1 |
| Datasets | 2 |
| Total cases | 4 |

## Runbook

**Start**: 

**SHA256 (POSIX)**: sha256sum <file>
**SHA256 (Windows)**: certUtil -hashfile <file> SHA256

**Endpoints**: (none harvested — see runbook recovery below)

### 🔧 Runbook Not Auto-Detected

The start command / endpoints above could not be harvested from the target repo.
Before running the evaluation, discover them yourself:

1. Read the target's `README`, `Makefile`, `package.json` scripts, or `pyproject.toml`.
2. Identify how to start the target service and which endpoint runs an evaluation.
3. Record what you find in TASKS.md under RECON before proceeding.

Do **not** guess a command — if you cannot determine it, stop and report.


## Data

Cases are read **live** from the harness database — nothing is exported into this repo:

**Database**: `tmp/aeh.db`
**Table**: `dataset_cases` — columns `id`, `dataset_id`, `input_json`, `expected_json`,
`labels_json`, `provenance`

Only `generated+reviewed`, `handwritten` and `derived+reviewed` rows are ever run; anything
still `synthetic` is unreviewed and is skipped by the driver. Each case's owning agent comes
from `labels_json.agent_id`, and that is what selects its dispatch module.

Run output (spanlogs, `manifest.json`) is written to the harness data directory too, never
into this repo — see `aeh_out_dir` in `wiring.json`.

| Dataset ID | Kind | Cases |
| --- | --- | --- |
| `ds_retriever` | synthetic_agent_io | 2 |
| `ds_writer` | synthetic_agent_io | 2 |

## Provider Configuration

Read run_config.json after task T003 runs.

(not harvested — the coding agent must list providers at runtime; see discovery task)

### 🔧 Provider Listing Not Auto-Detected

The list of configured LLM providers/models was not harvested. Resolve it at runtime.

**What you actually need — four things, nothing else:**

1. Which providers exist — id / display name / kind
2. The model id to use
3. The `base_url` — where the running target answers
4. **How** the target supplies its credential at call time: the mechanism only (e.g. "the app
   injects it server-side", "it reads env var X") — never the value

You never need the key itself: the target authenticates its own outbound calls, and the
dispatch modules address the target, not the provider.

**How to get them:**

1. Prefer the target's own provider/config **service** (see each agent card's constructor
   dependencies for one tagged `llm_provider`) and its masked listing method, or a masked
   HTTP listing endpoint. These strip secrets for you.
2. If you must read the store directly:
   a. Read the **schema first** — column names only, no rows — to find which columns hold
      items 1-3.
   b. **Ask the human for permission**, showing the exact query and the exact column list.
      Wait for approval.
   c. `SELECT` those columns **by name**. Never `SELECT *`, and never a free-form blob column
      (`extra`, `config`, `settings`, `metadata`, `env`): credentials are routinely stored
      *inside* such a column's JSON, so it leaks a key even though the name looks harmless.
3. Show the masked list to the human and ask which provider/model to use (task T004).
4. If a key value appears in your output anyway: stop, report it, and tell the human to rotate
   that credential immediately. Never echo the value again.


---

## Per-Agent Cards

### Agent: retriever  (role: retrieval_agent)

**WHERE** (from code index):

| What | Location |
| --- | --- |
| Location | [NEEDS CLARIFICATION] |
| Component `retriever` | `:0` |

**Invocation mode**: `unsupported`

**WIRE**: no case_binding harvested — dispatch module uses empty kwargs.

### Agent: writer  (role: writer)

**WHERE** (from code index):

| What | Location |
| --- | --- |
| Location | [NEEDS CLARIFICATION] |
| Component `writer` | `:0` |

**Invocation mode**: `unsupported`

**WIRE**: no case_binding harvested — dispatch module uses empty kwargs.

---

## Appendix: Raw Facts

```json
{
  "session_id": "sess-linear_rag",
  "plan_id": "sess-linear_rag",
  "dataset_ids": ["ds_retriever", "ds_writer"],
  "agents": [
  {
    "agent_id": "retriever",
    "role": "retrieval_agent",
    "invocation_mode": "unsupported",
    "location": ""
  },
  {
    "agent_id": "writer",
    "role": "writer",
    "invocation_mode": "unsupported",
    "location": ""
  }
]
}
```
