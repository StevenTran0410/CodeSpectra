# RPA-050 - Evaluation harness, golden repos, and quality gates against hallucination

- **Epic**: Quality
- **Priority**: P1
- **Relative estimate**: M
- **Milestone**: Milestone F
- **Dependencies**: RPA-032, RPA-033, RPA-034, RPA-040, RPA-041, RPA-042

## 1) Objective
Measure report quality concretely instead of relying on subjective feel, and enforce minimal quality gates before shipping internally.

## 2) Problem this ticket addresses
Products like this easily look “smart” while being wrong. Without an evaluation harness, the team tunes blind and fools itself.

## 3) Detailed scope
- Select 3–5 golden repos representing main archetypes: backend service, web app, monorepo, library, worker-heavy.
- Write answer keys at section-expectation level: entrypoints, core modules, important files, domain terms, standout conventions.
- Build a harness that runs report generation automatically and scores a set of basic criteria.
- Define minimal quality gates for internal release.
- Log parse coverage, evidence coverage, schema validity, and section completeness.

## 4) Implementation notes
- Not everything can be reduced to one number, but there must be smoke metrics and a spot-check framework.
- Keep “schema valid” separate from “content useful”; both matter.
- Section factual accuracy should use a semi-manual review checklist in early phases.

## 5) Subtask breakdown
- Select and normalize golden repos.
- Write expected facts per repo.
- Implement evaluation runner.
- Implement schema/evidence coverage checks.
- Produce internal release checklist.

## 6) Acceptance criteria
- There is an initial golden-repo set with answer keys.
- Each meaningful build can run an evaluation smoke test.
- There is at least one quality gate before packaging a release.
- The team can see regression when retrieval/prompts/parsers change.

## 7) Out of scope
- A comprehensive academic benchmark.
- A heavy LLM-as-judge framework from day one.

## 8) Risks / watch points
- Too few golden repos invites overfitting.
- Without manual review checklist, metrics can mislead.

## 9) Expected deliverables
- Golden repo suite
- Evaluation runner
- Quality gate checklist

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
