# RECON — your findings

**This file is yours. Regenerating the plan never overwrites it.**

AGENTS.md, TASKS.md, REFERENCE.md and CODE.md are rebuilt from scratch every time the plan is
re-rendered — anything you write in them is lost. This file is not, so record here every
discovery you would want to survive a new plan revision.

Keep the tally on the line below current; T003's verify reads it.

```
RECON: ? khớp, ? conflict
```

---

## [Drift] — plan says one thing, code says another, and the plan still works

One entry per finding: what the plan claimed, what the code actually does, file:line, and
whether you adapted or ignored it.

## [Conflict] — plan contradicts the target and blocks progress

One entry per finding: the task id, what broke, the evidence, and what you did about it.
Per AGENTS.md § Escalation you may adapt inside your own marked region and continue — record
that here. Stop and hand back only when the fix would require editing a sha-locked file.

## Target defects

Bugs in the target's own code that you hit. Do not fix them; record them here with evidence
so a human can decide.

## Notes from a previous install

If T005 moved an earlier `.aeh/` aside, record what its manifest claimed
(`attempted` / `succeeded` / `status`) — read from the file, not from memory.
