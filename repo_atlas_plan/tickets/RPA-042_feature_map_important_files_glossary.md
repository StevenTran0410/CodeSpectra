# RPA-042 - Functionality-to-file map, important-files radar, and glossary extraction

- **Epic**: Analysis
- **Priority**: P0
- **Relative estimate**: L
- **Milestone**: Milestone E
- **Dependencies**: RPA-032, RPA-033, RPA-034

## 1) Objective
Answer the core onboarding question: where main functionality lives, which files are central to that functionality, and which files form the backbone of the whole repo.

## 2) Problem this ticket addresses
Many tools talk about architecture but never tell devs “to understand feature A, read these files” or “this file matters because…”.

## 3) Detailed scope
- Produce functionality clusters from routes/docs/domain terms/services/tests/config.
- Map each feature cluster to entrypoints, core files, supporting files, tests, and configs.
- Compute an important-file score from graph metrics, entrypoint flags, config criticality, and symbol density.
- Extract glossary/domain terms from names in code/docs/tests/config.
- Produce structured sections: feature_map, important_files_radar, glossary.

## 4) Implementation notes
- The feature map does not need to be 100% semantically correct on day one, but it must be practically useful.
- Important-files radar should include a per-file reason: entrypoint, wiring hub, config spine, persistence core, shared abstraction, blast radius.
- The glossary should prioritize repeated terms with business meaning; avoid noise from framework internals.

## 5) Subtask breakdown
- Implement feature-clustering heuristics.
- Implement important-file scoring formula.
- Implement glossary extractor and ranking.
- Generate structured sections and viewer cards.
- Manually evaluate on 2–3 repo archetypes.

## 6) Acceptance criteria
- Users can read the feature map and see which features live in which modules/files.
- Important-files radar gives a clear rationale for each file.
- The glossary surfaces useful domain terms.
- All sections include evidence/confidence.

## 7) Out of scope
- A complete end-to-end business process graph.
- Ownership/churn from git history.

## 8) Risks / watch points
- Feature clustering can mis-group if repo naming is poor.
- Important-file score can bias toward large files or heavily imported files without careful weighting.

## 9) Expected deliverables
- Feature clustering engine
- Important file scorer
- Glossary extractor
- Three analysis sections

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
