# RPA-043 - Report viewer, evidence drawer, export, and compare runs

- **Epic**: Analysis UX
- **Priority**: P0
- **Relative estimate**: M
- **Milestone**: Milestone E
- **Dependencies**: RPA-002, RPA-040, RPA-041, RPA-042

## 1) Objective
Turn analysis artifacts into something readable, navigable, exportable, and comparable across scan runs.

## 2) Problem this ticket addresses
If output lives only in JSON/Markdown files, practical value drops sharply. Users need a UI to click through evidence, open the right file, and understand why a conclusion was produced.

## 3) Detailed scope
- Build a report viewer with sidebar sections, summary cards, badges, and an evidence drawer.
- Add components for confidence, unknowns, and follow-up questions.
- Support export of canonical JSON and Markdown reports.
- Support comparing the current run with a previous run using at least section-level diff.
- Support deep links from evidence to path/symbol in the repo viewer or an external editor path.

## 4) Implementation notes
- The UI does not need to be flashy, but it must be very clear and “investigable.”
- The evidence drawer is the trust anchor; invest there.
- Compare runs can start as diff summary/section hash in v1; no need for an overly heavy visual diff.

## 5) Subtask breakdown
- Build report viewer layout.
- Build evidence drawer and supporting data hooks.
- Wire export service.
- Build basic section diff for compare runs.
- Add open-file/open-path actions where the environment allows.

## 6) Acceptance criteria
- Users can view the full report inside the app.
- Each section can open an evidence drawer.
- JSON and Markdown export succeed.
- Compare runs show which sections changed.

## 7) Out of scope
- Collaborative comments/annotations.
- Cloud sharing of reports.

## 8) Risks / watch points
- If the viewer does not foreground evidence, the report will read like a generic AI essay.
- If export format diverges from the canonical schema, maintenance debt follows.

## 9) Expected deliverables
- Report viewer UI
- Evidence drawer
- Export pipeline
- Basic compare-runs UI

## 10) Definition of done
- Related code/services/UI are merged and runnable in a local environment.
- The ticket has minimal tests or an appropriate verification checklist.
- Logging and error states are not left empty.
- Related docs/settings are updated.
- No major ambiguity left open without a clear note.

## 11) Suggested QA checklist
- Re-run the ticket on at least one public repo or internal sample repo.
- Check empty state, error state, and cancel/retry state where applicable.
- Restart the app and re-enter the flow to confirm data still looks correct.
- Confirm logs do not leak secrets/tokens.
