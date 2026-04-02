# RPA-044 - Risk, Complexity, and Hotspot Section Generator

- **Epic**: Analysis
- **Priority**: P1
- **Relative estimate**: M
- **Milestone**: Milestone E
- **Dependencies**: RPA-031, RPA-032, RPA-033, RPA-034

## 1) Objective
Generate the Risk / Complexity / Unknowns section of the report — identifying god objects, suspected circular imports, TODO/FIXME hotspots, coverage gaps, config risk areas, and files with high blast radius — so developers know where to tread carefully.

## 2) Problem this ticket addresses
Section J of the project specification ("Risk / Complexity / Unknowns") is one of the highest-value differentiators of Repo CodeSpectra, yet no existing ticket implements it. RPA-041 covers convention and anti-pattern analysis, RPA-042 covers feature mapping and important files, but neither addresses risk zones: files that are dangerously large, import clusters that suggest hidden coupling, test coverage gaps, or environment configs that are scattered and fragile.

## 3) Detailed scope
- Detect **god object / oversized file** candidates: files above a configurable line-count threshold, or files with unusually high symbol density.
- Detect **suspected circular import chains**: using the import graph from RPA-033, flag strongly-connected components and surface the shortest suspicious cycles.
- Detect **TODO / FIXME / HACK / XXX hotspots**: count and cluster by module, flag modules with disproportionate annotation density.
- Detect **config and environment risk**: environment variable references spread across too many files, hardcoded values in non-config locations, migration files with no corresponding rollback.
- Detect **test coverage shape**: ratio of test files to source files by module, modules with zero test files, and any obvious test naming mismatches.
- Detect **blast radius candidates**: reuse the graph centrality scores from RPA-033 to flag files that, if changed, would affect an unusually large part of the codebase.
- Generate a structured `risks_unknowns` section following the report schema from RPA-002, with evidence, confidence, and severity per risk item.

## 4) Implementation notes
- Risk findings must carry a **severity**: `high` / `medium` / `low`, not just a list. A single god object is high; a few scattered TODOs are low.
- Circular import detection does not need to be perfect — flagging strongly-connected components of size >= 3 is sufficient for v1.
- Test coverage detection should be structural (test file ratio by module), not instrumented coverage. Do not require running the test suite.
- The god object threshold should be configurable in settings, not hardcoded.
- Blast radius should reuse centrality data from RPA-033; do not recompute graph metrics independently.

## 5) Subtask breakdown
- Implement god object / oversized file detector.
- Implement suspected circular import chain detector using the existing graph.
- Implement TODO/FIXME/HACK hotspot scanner and module-level clustering.
- Implement config/env risk heuristics.
- Implement structural test coverage ratio by module.
- Implement blast radius list sourced from graph centrality.
- Generate the `risks_unknowns` section artifact with severity + evidence.
- Build viewer card for the risk section in the report UI.

## 6) Acceptance criteria
- The `risks_unknowns` section is present in the report with at least 2–3 meaningful findings on any real-world sample repo.
- Each finding has a severity, at least one evidence item (file path / symbol), and a short rationale.
- The section clearly distinguishes between certain findings (e.g., a 3000-line file) and suspected findings (e.g., possible circular import).
- Blast radius list is non-empty and includes files the reviewer would recognize as central.
- Viewer card renders findings grouped by severity.

## 7) Out of scope
- Instrumented runtime coverage (requires running test suite).
- Security vulnerability scanning.
- Automatic fix suggestions for any risk finding.

## 8) Risks / watch points
- God object detection by line count alone is a noisy signal; must combine with symbol density to reduce false positives.
- Circular import detection in dynamic languages (Python, JS) is approximate; confidence must reflect this.
- If the risk section is too long, it loses credibility. Cap the number of surfaced items per category and sort by severity.

## 9) Expected deliverables
- Risk detection engines (god object, circular imports, TODO hotspot, config risk, test gap, blast radius)
- `risks_unknowns` section generator
- Viewer card for risk section

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
- Verify that a repo with no TODOs produces an empty but valid findings list, not an error.
- Verify that severity labels are consistent and not all rated "high."
