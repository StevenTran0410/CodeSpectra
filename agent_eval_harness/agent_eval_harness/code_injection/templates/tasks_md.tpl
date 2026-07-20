# TASKS — Evaluation Implementation Plan

**Total agents**: {{scalar:plan.total_agents}}
**Batch size**: {{scalar:plan.batch_size}} (default, adjust if needed)

---

## M0 — Setup

> Tổng: {{scalar:plan.total_agents}} · Xong: 0 · Batch này: Setup · Còn: All

Preparation tasks before wiring agents.

- [ ] T001 Create and check out the eval branch `{{scalar:plan.branch_name}}` from the current HEAD
      Every edit in this plan happens on that branch — never commit to the default branch.
      Verify: `git rev-parse --abbrev-ref HEAD` → `{{scalar:plan.branch_name}}`

- [ ] T002 [P] Verify data is reachable — query {{scalar:plan.db_path}} and count ≥1 case
      Verify: `sqlite3 {{scalar:plan.db_path}} "SELECT COUNT(*) FROM dataset_cases;" | grep -qE '[1-9]'` → exit 0

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

{{marker:setup_discovery_tasks}}

{{table:discovery_tasks}}

🛑 GATE M0: All setup tasks complete. Proceed only if all checks green.

---

{{table:skeleton_tasks}}

---

{{table:batch_headers}}

---

{{table:finish_tasks}}

---

## Appendix: RECON Results

**Your findings live in `RECON.md`, not here.** That file is yours: regenerating this plan
never overwrites it, so notes recorded there survive a new plan revision. Anything written
into this file is lost the next time the plan is rendered.

Create `RECON.md` beside this file if it does not exist, and keep the running tally plus every
`[Drift]` / `[Conflict]` entry in it.

---

## Reference: Configuration

**Database**: `{{scalar:plan.db_path}}`
**Datasets**: {{scalar:plan.dataset_ids}}
**Provider**: See run_config.json
**Git branch**: `{{scalar:plan.branch_name}}`
