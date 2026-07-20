# REFERENCE — System Overview & Per-Agent Detail

## Target System

**ID**: {{scalar:plan.target_system_id}}
**Branch**: {{scalar:plan.branch_name}}

{{table:target_overview}}

## Runbook

**Start**: {{scalar:runbook.start_command}}

**SHA256 (POSIX)**: {{scalar:runbook.sha256_posix}}
**SHA256 (Windows)**: {{scalar:runbook.sha256_windows}}

**Endpoints**: {{table:endpoints}}

{{block:runbook_missing}}

## Data

Cases are read **live** from the harness database — nothing is exported into this repo:

**Database**: `{{scalar:plan.db_path}}`
**Table**: `dataset_cases` — columns `id`, `dataset_id`, `input_json`, `expected_json`,
`labels_json`, `provenance`

Only `generated+reviewed`, `handwritten` and `derived+reviewed` rows are ever run; anything
still `synthetic` is unreviewed and is skipped by the driver. Each case's owning agent comes
from `labels_json.agent_id`, and that is what selects its dispatch module.

Run output (spanlogs, `manifest.json`) is written to the harness data directory too, never
into this repo — see `aeh_out_dir` in `wiring.json`.

{{table:datasets}}

## Provider Configuration

Read run_config.json after task T003 runs.

{{table:provider_listing}}

{{block:provider_missing}}

---

## Per-Agent Cards

{{table:agent_cards}}

---

## Appendix: Raw Facts

```json
{
  "session_id": "{{scalar:plan.session_id}}",
  "plan_id": "{{scalar:plan.plan_id}}",
  "dataset_ids": {{scalar:plan.dataset_ids}},
  "agents": {{table:agents_json}}
}
```
