# CODE — Generated Artifacts

Each code block below is ready to copy verbatim to the specified file.

---

## Tracer

→ Create file at `.aeh/tracer.py`

**SHA256**: {{sha256:tracer}}

{{code:tracer}}

---

## Run Eval Script

→ Create file at `.aeh/run_eval.py`

**SHA256**: {{sha256:run_eval}}

{{code:run_eval}}

---

## Wiring Configuration

→ Create file at `.aeh/wiring.json`

**SHA256**: {{sha256:wiring}}

{{code:wiring}}

---

## Eval API Route

→ Create file at `.aeh/aeh_eval.py`

**SHA256**: {{sha256:aeh_eval}}

{{code:aeh_eval}}

---

## Per-Agent Dispatch Modules

{{table:dispatch_modules}}

---

## Server Entrypoint Edits

→ Target file: {{scalar:plan.main_entry_path}} — locate it in the target repo (see TASKS.md M1)

{{table:main_py_edits}}

---

## Target Setup

→ Create file at `.aeh/target_init.py`, **then implement its marked region**

`run_eval.py` calls into this once before the first case. It is deliberately a separate,
**not** sha-locked file so a newer plan can ship a newer `run_eval.py` without erasing your
implementation.

{{code:target_init}}

---

## Retrieval Stub

→ Create file at `.aeh/retrieval_stub.py`, **then implement its marked region**

Every agent that retrieves internally imports `make_retrieval_stub` from here and feeds it the
case's own evidence, so the agent reads the case instead of the real repository. Only the
plumbing is generated: the method shape belongs to the target's retrieval dependency, so you
give it that shape. This file is deliberately **not** sha-locked.

{{code:retrieval_stub}}
