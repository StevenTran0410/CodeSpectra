# TASKS — Evaluation Implementation Plan

**Total agents**: 5
**Batch size**: 4 (default, adjust if needed)

---

## M0 — Setup

> Tổng: 5 · Xong: 0 · Batch này: Setup · Còn: All

Preparation tasks before wiring agents.

- [ ] T001 Create and check out the eval branch `aeh/eval-langgraph_agent` from the current HEAD
      Every edit in this plan happens on that branch — never commit to the default branch.
      Verify: `git rev-parse --abbrev-ref HEAD` → `aeh/eval-langgraph_agent`

- [ ] T002 [P] Verify data is reachable — query tmp/aeh.db and count ≥1 case
      Verify: `sqlite3 tmp/aeh.db "SELECT COUNT(*) FROM dataset_cases;" | grep -qE '[1-9]'` → exit 0

- [ ] T003 [P] RECON — compare this plan against actual codebase, recording findings in `RECON.md`
      Verify: `RECON.md` exists and its first line reads `RECON: n/N khớp, m conflict`

- [ ] T004 Select provider and model, then write `.aeh/run_config.json`
      **Step 0 — check first:** if a run config already exists (`.aeh/run_config.json`, or an
      equivalent at the repo root), read it and skip straight to Verify. A human has already
      chosen, and the protocol below is only for when no such file exists.
      You need exactly four things from the provider store, and nothing else:
      1. which providers exist — id / display name / kind
      2. the model id to use
      3. the `base_url` — where the running target answers
      4. **how** the target supplies its credential at call time: the mechanism only
         (e.g. "the app injects it server-side", "it reads env var X"), never the value
      You never need the key itself — the target authenticates its own outbound calls.
      Prefer the target's own **masked** listing API/service method. If you must read the store:
      a. Read the **schema first** (`PRAGMA table_info(<table>)` / `DESCRIBE`) — column names
         only, no rows — to find which columns hold items 1-3.
      b. **Ask the human for permission**, showing the exact query and the exact column list.
         Wait for approval.
      c. `SELECT` those columns **by name**. Never `SELECT *`, and never a free-form blob column
         (`extra`, `config`, `settings`, `metadata`, `env`): credentials are routinely stored
         *inside* such a column's JSON, so it leaks a key even though the name looks harmless.
      Then show the masked list and ask which provider/model to use.
      Write `.aeh/run_config.json` with these keys (the dispatch modules read them):
      `provider_id`, `model_id`, `base_url` (where the running target answers), and
      `auth_header` only if the target requires one.
      Verify: `.aeh/run_config.json` parses and `provider_id` + `model_id` + `base_url` are non-empty.

Pre-flight — each of the four below is cheap, and each catches a whole class of failure that
otherwise surfaces much later disguised as something else. Do them before wiring any agent.

- [ ] T004a Prove the target can reach its LLM at all. Send a one-word prompt **through the
      target's own provider code path** (not a raw HTTP client) using the ids from
      `.aeh/run_config.json`.
      Verify: a non-empty completion comes back. On failure, **print the endpoint the client
      actually called** and compare it with the configured one before concluding anything —
      an auth error from an OpenAI-compatible gateway is indistinguishable from a bad
      credential unless you know where the request went. Do not assume the credential is wrong.

- [ ] T004b Prove the config points at the store that actually holds your data. Confirm the
      store your init path opens contains the record your `provider_id` names.
      Verify: the record is found. A target often ships more than one store file (a checked-in
      empty one beside the real one); picking wrong fails much later as a confusing "not found".

- [ ] T004c Cross-check every field the wiring binds against the real cases. For each agent,
      list the bound case fields that are absent from **every one of its cases** — not just
      the sample shown in REFERENCE.md.
      Verify: zero missing, or each gap recorded in `RECON.md` before any agent is wired. A
      field bound but never present arrives as null and is rejected deep inside the agent,
      where the error is easy to swallow.

- [ ] T004d Prove a dispatch call actually produces a span, once, before building eleven more.
      Wire the single simplest agent and run its verify.
      Verify: `.aeh/out/*.jsonl` gains at least one `span` record. If calling the entry method
      directly yields no span, the tracer only observes calls made through the target's own
      framework machinery — resolve that now and record the shape you used in `RECON.md`, so
      the remaining dispatch modules follow it instead of each rediscovering it.


[NEEDS CLARIFICATION: setup_discovery_tasks — see D1]


- [ ] D1 Clarify setup_discovery_tasks — Determine and document: setup_discovery_tasks
      Record the answer in § RECON before continuing. Never read or print a credential value while investigating.

🛑 GATE M0: All setup tasks complete. Proceed only if all checks green.

---

## M1 — Skeleton: install the harness files
> Điều kiện vào: [x] 🛑 GATE M0. These files are copied verbatim — do not edit them.

- [ ] T004e Determine the target's Python import root — the directory on `sys.path` when the app runs, from which its own package imports resolve — and create `.aeh/` directly inside it. `run_eval.py` puts its own parent on `sys.path`, so if `.aeh/` sits anywhere else the target's imports fail. Record the resolved path in `RECON.md`; every `python .aeh/run_eval.py …` command below is relative to it.
      Verify: `python -c "import <target_top_level_package>"` succeeds from `.aeh/`'s parent

- [ ] T005 If `.aeh/` already exists from an earlier plan, move it aside — `mv .aeh .aeh.bak-<timestamp>` — then start clean. Do not merge old and new installs, and never delete it: its `out/*.jsonl` and `manifest.json` are untracked, so git cannot recover them. Report what the old manifest claimed (`attempted`/`succeeded`) in RECON.md rather than summarising it from memory or from any doc.
      Verify: `.aeh/` is absent or empty before you copy anything into it

- [ ] T006 If you moved an install aside, copy every file listed under `agent_owned_files` in the OLD `.aeh.bak-*/wiring.json` back into the fresh `.aeh/`. Those files hold your implementations, not generated content — a reinstall that skips this step silently reverts them and the run regresses while the gates still pass. Add any further files you author to that list so the next reinstall keeps them too.
      Verify: every path in the old `agent_owned_files` exists in the new `.aeh/`, or the list was empty

- [ ] T007 Create `.aeh/tracer.py` — copy the matching block from CODE.md **verbatim**
      Verify: sha256 of `.aeh/tracer.py` == `c0e1eb07a982bce10d3e3119c47004d9ecfb430be5a530e762c112d3881bb426`
- [ ] T008 Create `.aeh/run_eval.py` — copy the matching block from CODE.md **verbatim**
      Verify: sha256 of `.aeh/run_eval.py` == `7d224279306cb9098e17f1fd2e7534b3695f0156c3f9724df1b7abd732d16bb2`
- [ ] T009 Create `.aeh/aeh_eval.py` — copy the matching block from CODE.md **verbatim**
      Verify: sha256 of `.aeh/aeh_eval.py` == `5983cebe1173b01e4e25d796371457efe4df445d64178960ff37e47e94e52b8b`
- [ ] T010 Create `.aeh/wiring.json` from CODE.md. Generated from this target's harvested signatures, so it is **not** sha-locked: if a binding contradicts the real signature, correct it here and record the change in RECON.md as `[Conflict]`.
      Verify: `.aeh/wiring.json` parses as JSON and every agent id in it has a dispatch entry
- [ ] T011 Create `.aeh/retrieval_stub.py` from CODE.md, then **implement its `=== IMPLEMENT THIS ===` region**: give the stub the real retrieval dependency's method names, signatures, return types and async-ness, reading them from the target's source. Not sha-locked — its interface belongs to the target, not to this plan.
      Verify: `grep -c "IMPLEMENT THIS" .aeh/retrieval_stub.py` → 0, and one real call through it returns content derived from the case's evidence (a shape mismatch can be swallowed by the caller and leave the agent running on nothing)
- [ ] T012 Locate the target's server entrypoint (the file that registers its routes) and apply the three marker blocks from CODE.md § "Server Entrypoint Edits"
      Record the file you edited in § RECON. Verify: `grep -c "aeh:begin" <that file>` → 3, and the service still starts
- [ ] T013 Create `.aeh/target_init.py` from CODE.md, then **implement its `=== IMPLEMENT THIS ===` region** — the target's once-per-process setup (database/connection init, config bootstrap). Not sha-locked: it is yours, and a newer plan can ship a newer `run_eval.py` without erasing it.
      Verify: `python .aeh/run_eval.py --verify` gets past setup without an "not initialised"-style error

🛑 GATE M1: start the target service (REFERENCE.md § Runbook) and confirm the eval route answers. Fail → stop, report the task id; do not start wiring agents.

---

## M2 — Batch 1/2: wire agents 1-4
> Tổng: 5 · Xong: 0 · Batch này: load_context, plan_step, investigate, synthesize · Còn: 1
> Điều kiện vào: [x] 🛑 GATE M1. New session → re-run the M1 Verify command first.

- [ ] T020 [P] Create `.aeh/dispatch/load_context.py` from its CODE.md block, then **implement its `=== IMPLEMENT THIS ===` region** (the generated kwargs above it are final — do not edit them)
      Verify: `python .aeh/run_eval.py --verify --agent load_context` → exit 0, ≥1 span, and `grep -c "IMPLEMENT THIS" .aeh/dispatch/load_context.py` → 0 leftover NotImplementedError
- [ ] T021 [P] Create `.aeh/dispatch/plan_step.py` from its CODE.md block, then **implement its `=== IMPLEMENT THIS ===` region** (the generated kwargs above it are final — do not edit them)
      Verify: `python .aeh/run_eval.py --verify --agent plan_step` → exit 0, ≥1 span, and `grep -c "IMPLEMENT THIS" .aeh/dispatch/plan_step.py` → 0 leftover NotImplementedError
- [ ] T022 [P] Create `.aeh/dispatch/investigate.py` from its CODE.md block, then **implement its `=== IMPLEMENT THIS ===` region** (the generated kwargs above it are final — do not edit them)
      Verify: `python .aeh/run_eval.py --verify --agent investigate` → exit 0, ≥1 span, and `grep -c "IMPLEMENT THIS" .aeh/dispatch/investigate.py` → 0 leftover NotImplementedError
- [ ] T023 [P] Create `.aeh/dispatch/synthesize.py` from its CODE.md block, then **implement its `=== IMPLEMENT THIS ===` region** (the generated kwargs above it are final — do not edit them)
      Verify: `python .aeh/run_eval.py --verify --agent synthesize` → exit 0, ≥1 span, and `grep -c "IMPLEMENT THIS" .aeh/dispatch/synthesize.py` → 0 leftover NotImplementedError

🛑 GATE M2: `python .aeh/run_eval.py --verify --batch 1` → 4/4 agents ≥1 span, 0 error. Fail → stop, report the task id.

---

## M3 — Batch 2/2: wire agents 5-5
> Tổng: 5 · Xong: 4 · Batch này: retrieve · Còn: 0
> Điều kiện vào: [x] 🛑 GATE M2. New session → re-run the M2 Verify command first.

- [ ] T024 [P] Create `.aeh/dispatch/retrieve.py` from its CODE.md block, then **implement its `=== IMPLEMENT THIS ===` region** (the generated kwargs above it are final — do not edit them)
      Verify: `python .aeh/run_eval.py --verify --agent retrieve` → exit 0, ≥1 span, and `grep -c "IMPLEMENT THIS" .aeh/dispatch/retrieve.py` → 0 leftover NotImplementedError

🛑 GATE M3: `python .aeh/run_eval.py --verify --batch 2` → 1/1 agents ≥1 span, 0 error. Fail → stop, report the task id.

---

## M4 — Run & hand back
> Điều kiện vào: [x] 🛑 GATE M3 (every agent batch green).

- [ ] T900 Run the full evaluation: `python .aeh/run_eval.py`
      Verify: `.aeh/out/manifest.json` exists and reports 0 errored cases
- [ ] T901 Hand back to AEH: the manifest path, this TASKS.md (with your ticks and any
      `[Drift]` / `[Conflict]` notes), and the RECON line.

🛑 GATE M4: manifest exists and every non-skipped agent produced ≥1 span.

---

## Appendix: RECON Results

**Your findings live in `RECON.md`, not here.** That file is yours: regenerating this plan
never overwrites it, so notes recorded there survive a new plan revision. Anything written
into this file is lost the next time the plan is rendered.

Create `RECON.md` beside this file if it does not exist, and keep the running tally plus every
`[Drift]` / `[Conflict]` entry in it.

---

## Reference: Configuration

**Database**: `tmp/aeh.db`
**Datasets**: ["ds_investigate", "ds_load_context", "ds_plan_step", "ds_retrieve", "ds_synthesize"]
**Provider**: See run_config.json
**Git branch**: `aeh/eval-langgraph_agent`
