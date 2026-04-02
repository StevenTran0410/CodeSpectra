# RPA-041 - Convention/style analyzer and hidden anti-pattern discovery

- **Epic**: Analysis
- **Priority**: P1
- **Relative estimate**: L
- **Milestone**: Milestone E
- **Dependencies**: RPA-031, RPA-032, RPA-033, RPA-034

## 1) Objective
Surface the repo’s real coding conventions and implicit “forbidden” practices by combining statistics, AST patterns, and boundary heuristics.

## 2) Problem this ticket addresses
This is especially painful when devs join a new team: documentation never covers everything, and code review usually catches violations only after the fact.

## 3) Detailed scope
- Detect naming conventions and suffix/prefix patterns.
- Detect folder-role conventions such as service/controller/usecase/repository/handler.
- Detect import-boundary patterns and outliers.
- Detect style signals: functional vs class-heavy, if/else density, error-handling style, DI style, test layout style.
- Produce a Hidden Rules / Negative Conventions section with appropriate confidence.

## 4) Implementation notes
- Tree-sitter plus ast-grep-style structural rules are a strong foundation for this problem.[R6][R7]
- Do not assert an anti-pattern from only one or two samples; compare against the majority pattern.
- Separate “observed conventions” from “suspected rules” to avoid overclaiming.

## 5) Subtask breakdown
- Design a convention-signals catalog.
- Implement naming/path pattern miner.
- Implement AST-based structural scan for a set of sample rules.
- Implement outlier detector for boundary violations.
- Generate a structured conventions report.

## 6) Acceptance criteria
- The conventions section states at least several rules that are genuinely useful on a sample repo.
- There is a clear distinction between observed patterns and suspected anti-patterns.
- The section includes evidence and confidence.
- Do not invent rules if the repo is too heterogeneous.

## 7) Out of scope
- Static lint auto-fix.
- A full rule engine like a standalone linter product.

## 8) Risks / watch points
- Easy to overfit superficial naming conventions.
- If the confidence model is weak, the app will overstate conclusions.

## 9) Expected deliverables
- Convention mining engine
- Rule signal catalog
- Conventions/anti-pattern section generator

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
