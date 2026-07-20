# AGENTS — Evaluation Plan Execution Guide

**Mission**: Implement the instrumentation defined in this plan to enable evaluation of {{scalar:plan.target_system_id}}.

**Status**: Read TASKS.md as the source of truth. Each `- [ ]` is your working checklist.

## Execution Loop

1. **Each session**: Open TASKS.md, find the first empty checkbox `- [ ]`.
2. **Execute the task**: Run the command, verify the condition.
3. **Tick the box** and move to the next.
4. **Hit a GATE** (red 🛑 line)? Run the Verify command. If it passes, continue. If it fails, stop and report the task ID.
5. **No workaround**: If a gate fails, the plan has detected an issue; report it back to AEH (see Hand-back).

## Session Resume

New session? Open TASKS.md, read the last completed gate (its Verify command), and re-run it to confirm state before continuing to the next task.

## Boundaries

**Always** (can execute immediately):
- Run verification commands
- Copy code blocks from CODE.md
- Edit marked regions (between `[P]` markers)
- Ask AEH-supplied clarifications (see REFERENCE.md)

**Ask First** (these are the only questions you may ask):
- Permission to read the target's provider/credential store — before running the query, show
  the human the exact columns you will read, the columns you will exclude, and why (T004)
- Which provider/model to use for the run (T004)

**Never** (do not do these):
- Read or print any value that looks like `api_key`, `sk-`, `AKIA`, `Bearer`, or JWT token
- Run `SELECT *` (or any unbounded read) against a table that can hold credentials — always
  name the columns explicitly
- Read free-form blob columns (`extra`, `config`, `settings`, `metadata`, `env`, …) from such
  a table: a key is commonly stored *inside* the JSON of an innocently-named column, so
  filtering by column name alone does **not** protect you
- If a secret does reach your output anyway: stop, say so immediately, and tell the human to
  rotate that credential. Do not repeat the value, not even to describe it.
- Modify code outside `[P]`-marked regions
- Skip gates or invent workarounds
- Assume anything not stated in this plan

## Spending

- Verify gates cost roughly one live model call per agent — run them freely, that is their job.
- The final full run calls the model **once per case**. It is billed and it is not cheap:
  ask the human before starting it, and say how many cases that is.

## Target-code defects

A bug in the target's own code, outside every `[P]` region, is **not yours to fix** — even
when it blocks the run. Record it in `RECON.md` with the evidence and hand back. Fixing target
code silently changes what is being evaluated.

## Drift & Conflict Protocol

The plan was generated from code. Code may have changed since then.

**[Drift]**: Plan says "line 42 has `import foo`" but you see it at line 45 → that's drift (code moved). Adjust and note it; continue.

**[Conflict]**: Plan says "agent X has method run()" but you don't see it anywhere → that's conflict.

**Escalation — read this before stopping.** When a generated artefact contradicts the target's
real interface, and the fix belongs inside a region this plan already told you to implement:
**adapt it there, record it in `RECON.md` as `[Conflict]`, and continue.** That is the marked
region working as intended, not a workaround.

STOP and hand back only when the fix would require editing a **sha-locked** file or a region
marked do-not-edit. Files that are yours — `target_init.py`, `retrieval_stub.py`, the
`IMPLEMENT THIS` regions, `wiring.json`, and anything listed in `agent_owned_files` — are
never a reason to stop.

## Hand-Back

If a gate fails or a conflict appears:
1. Save TASKS.md (include your notes: `[Drift]` or `[Conflict]` + details)
2. Note the failing task ID (e.g., T042)
3. Return the file to AEH with:
   - Which task failed (T###)
   - The error message or unmet condition
   - Any `[Drift]` or `[Conflict]` notes you added

---

**Next:** Open and read REFERENCE.md for target details, then start TASKS.md at T001.
