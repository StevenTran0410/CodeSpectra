# RPA-035 - Analysis Run Screen: Scan Mode Selection / Privacy Mode Toggle / Progress UX

- **Epic**: Analysis UX
- **Priority**: P0
- **Relative estimate**: M
- **Milestone**: Milestone E
- **Dependencies**: RPA-012, RPA-030, RPA-040, RPA-042

## 1) Objective
Build the dedicated Analysis Run screen that lets the user configure a scan before launching it, observe step-by-step progress, and cancel or retry — without the configuration being scattered across other screens.

## 2) Problem this ticket addresses
The project description describes Quick Scan vs Full Scan and Strict Local vs BYOK Cloud as explicit user choices before every run. Currently RPA-012 provides the job backend and RPA-040/042 provide section generators, but no ticket owns the run-configuration UI that ties everything together. Without this screen, the user has no way to understand what they are about to run or how long it will take.

## 3) Detailed scope
- Build the Analysis Run screen with the following controls:
  - **Scan mode selector**: Quick Scan / Full Scan with a short description of what each includes.
  - **Privacy mode selector**: Strict Local / BYOK Cloud (inherits from workspace default but can be overridden per run). Cloud mode must show the consent reminder inline.
  - **Target selector**: which repository snapshot / branch to run against.
  - **Estimated scope indicator**: number of files in manifest, approximate token budget.
- Show a step-by-step progress log during the run: cloning, manifesting, parsing, graphing, retrieving, generating, exporting.
- Allow cancel mid-run and show a clean stopped state.
- Allow retry from the last failed step where possible (depends on RPA-012 checkpoint support).
- Show a summary card when the run finishes: sections completed, sections with low confidence, elapsed time.

## 4) Implementation notes
- Quick Scan should include: manifest + symbol index + project identity + architecture + important files. Full Scan adds: embeddings + retrieval + feature map + convention mining + risk section.
- The privacy mode selector on this screen must not bypass the consent flow from RPA-022; it should re-confirm consent if the user switches to cloud mode for the first time.
- Progress log events should come from the `JobEvent` stream defined in RPA-012; this screen is purely a consumer, not a producer.
- The estimated scope indicator keeps expectation management honest and prevents the "why is it taking so long" friction.

## 5) Subtask breakdown
- Design the run configuration form and wire it to the job orchestrator.
- Implement Quick Scan vs Full Scan pipeline branching (deciding which pipeline steps to include).
- Implement the progress log panel consuming `JobEvent` stream.
- Implement cancel and retry actions tied to RPA-012 cancellation tokens.
- Build the post-run summary card with section completion status.

## 6) Acceptance criteria
- User can select Quick Scan or Full Scan and see a description of what each runs.
- User can select Strict Local or BYOK Cloud and is reminded of the privacy implication when selecting cloud.
- During the run, step-by-step progress is visible with named stages.
- Cancel stops the run cleanly and leaves the app in a stable state.
- Post-run summary shows which sections completed and flags any with low confidence.

## 7) Out of scope
- Per-section enable/disable toggles (too granular for v1).
- Scheduled/automated analysis runs.
- Remote observability or run sharing.

## 8) Risks / watch points
- Quick vs Full scope needs to be carefully defined so users do not expect Full Scan quality from a Quick Scan.
- If privacy mode toggle is poorly placed, users may accidentally switch to cloud mode without realizing it.

## 9) Expected deliverables
- Analysis run configuration UI
- Quick Scan / Full Scan pipeline branching logic
- Progress log panel
- Post-run summary card

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
- Verify that cancelling mid-run does not leave behind corrupted index artifacts.
- Verify that switching from local to cloud mode on this screen triggers the consent reminder.
