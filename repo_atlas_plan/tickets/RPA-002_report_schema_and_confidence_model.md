# RPA-002 - Design the Output Schema for the Report, Evidence Model, and Confidence Model

* **Epic**: Foundation
* **Priority**: P0
* **Relative Estimate**: M
* **Milestone**: Milestone A
* **Dependencies**: RPA-001

## 1) Goal

Turn the report into a structured artifact that can be tested, diffed, and rendered instead of a loose block of free-form markdown.

## 2) Problem this ticket solves

If the LLM is allowed to return free-form text, the report becomes difficult to render, difficult to export, difficult to compare across runs, difficult to attach evidence to, and much more prone to hallucination without any anchoring structure.

## 3) Detailed scope

* Design the standard JSON schema for `ReportArtifact` and `SectionArtifact`.
* Lock the v1 section list: `project_identity`, `architecture_overview`, `repo_structure`, `conventions`, `feature_map`, `important_files`, `onboarding_reading_order`, `glossary`, `risks_unknowns`.
* Design the `EvidenceItem` model: file path, symbol, snippet reference, reason, confidence contribution.
* Design the minimum confidence model: `high` / `medium` / `low` + reason codes.
* Design the unknowns / blind spots model so every section has a place to explicitly state limitations.
* Lock the initial export format: canonical JSON + Markdown render.

## 4) Implementation notes

* The schema should separate the data model from the render model. The UI should render from canonical JSON; Markdown should be derived output only.
* Each section should include fields such as `summary`, `bullets`, `evidence`, `confidence`, `unknowns`, and `followup_questions`.
* The schema should include a version field to avoid breaking future releases when the structure changes.

## 5) Breakdown subtasks

* Draft schema v0 and review it with the person responsible for the UI viewer.
* Create 2 to 3 sample report JSON artifacts for a backend repo, a monorepo, and a library.
* Write rules defining which sections must include evidence and which sections may allow controlled empty evidence.
* Create the Markdown rendering spec derived from the JSON schema.

## 6) Acceptance criteria

* A clear JSON schema or TypeScript zod schema exists for the report.
* Sample artifacts exist and can be used as UI mock data.
* A schema versioning plan exists.
* All v1 sections follow a consistent format.

## 7) Out of scope

* Real prompt implementation for each section.
* Beautiful UI for the report viewer.

## 8) Risks / watchpoints

* A schema that is too loose will damage viewer quality and QA quality.
* A schema that is too rigid without space for unknowns will encourage output that pretends to be more certain than it really is.

## 9) Expected deliverables

* Report schema spec
* Sample JSON artifacts
* Markdown rendering spec

## 10) Definition of done

* Related code/service/UI work has been merged and runs in the local environment.
* The ticket has at least minimal tests or an appropriate verification checklist.
* Logging and error states are not left blank or unhandled.
* Related docs/settings have been updated.
* No major ambiguity remains open without being clearly documented.

## 11) Suggested QA checklist

* Re-run this ticket against at least 1 public repository or 1 internal sample repository.
* Check empty states, error states, and cancel/retry states where applicable.
* Restart the app and return to the flow to verify that the data remains correct.
* Verify that logs do not expose secrets or tokens.
