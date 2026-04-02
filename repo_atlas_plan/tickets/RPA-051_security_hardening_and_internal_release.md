# RPA-051 - Security/privacy hardening, packaging, and internal release readiness

- **Epic**: Release
- **Priority**: P1
- **Relative estimate**: M
- **Milestone**: Milestone F
- **Dependencies**: RPA-010, RPA-011, RPA-022, RPA-023, RPA-024, RPA-043, RPA-050

## 1) Objective
Move the app from prototype to something staff can install and use internally without being reckless about security and privacy.

## 2) Problem this ticket addresses
The product is local-first desktop but touches git, the filesystem, credentials, and cloud APIs. Without hardening and solid packaging, trust incidents are easy.

## 3) Detailed scope
- Review Electron security baseline: contextIsolation, CSP, IPC validation, limiting navigation/external content.[R14]
- Review secret handling, log redaction, export redaction, and privacy badge/consent flow.
- Review denylist paths and secret-risk file handling in cloud mode.
- Prepare packaging for internal release on at least the first target OS.
- Write an internal install/run/troubleshooting guide.

## 4) Implementation notes
- This is not a full pentest, but it must be enough to avoid shipping naively.
- Prefer one OS for packaging first; do not commit to full multi-platform if resources are tight.
- The internal release guide must state current limits of parser/language/provider support.

## 5) Subtask breakdown
- Run a security checklist across the app.
- Review secret/log/export paths.
- Produce first package build for the target OS.
- Write release notes and known limitations.
- Run internal dogfooding with 2–3 users.

## 6) Acceptance criteria
- There is an internal build that installs successfully.
- No plaintext secrets in DB/logs/exports per policy.
- Cloud-mode warnings behave correctly.
- Install documentation and known limitations are clear.

## 7) Out of scope
- Readiness for a public external launch.
- Formal third-party security audit.

## 8) Risks / watch points
- Desktop packaging often surfaces mundane bugs that still burn a lot of time.
- If known limitations are not documented, internal feedback becomes noisy.

## 9) Expected deliverables
- Internal release build
- Security/privacy checklist results
- Install guide
- Known limitations document

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
