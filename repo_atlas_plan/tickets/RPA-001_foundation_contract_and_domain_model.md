# RPA-001 - Lock the Product Contract, Privacy Modes, and Domain Model

* **Epic**: Foundation
* **Priority**: P0
* **Relative Estimate**: M
* **Milestone**: Milestone A
* **Dependencies**: None

## 1) Goal

Lock the phase 1 scope, clearly define the two privacy modes, and establish the main user journeys and core domain model to prevent product drift during implementation.

## 2) Problem this ticket solves

If the contract is not locked early, the team will almost certainly start arguing midway through implementation about whether the app is allowed to send code to the cloud, what the report must contain, which artifact is the source of truth, and which objects are truly core to the system.

## 3) Detailed scope

* Clearly define the two modes: Strict Local and BYOK Cloud, including the wording shown in the UI.
* Lock the primary personas, JTBD, and the core product promise for phase 1.
* Define the product-level object model: Workspace, RepositoryConnection, RepositorySnapshot, AnalysisRun, ReportArtifact, SecretReference, ConsentRecord.
* Lock the high-level state machine for an analysis run: created -> cloning -> indexing -> generating -> completed/failed/cancelled.
* Lock the 5 most important user journeys: create workspace, configure provider, connect code host, run analysis, view/export report.
* Define the phase 1 boundary and explicit non-goals to avoid scope creep into code editing, agentic behavior, or task planning.
* Define the success criteria for phase 1 from both product and technical quality perspectives.

## 4) Implementation notes

* The output documentation from this ticket must be precise enough for all later tickets to follow directly. It must not be vague.
* A separate decision log should be created for questions that are likely to become contentious: desktop vs web, local embeddings vs provider-based embeddings, cloud mode warning wording, and which export format should be treated as first-class.
* It must be explicitly locked that “100% local” is only true in Strict Local Mode, to avoid misleading marketing language.

## 5) Breakdown subtasks

* Write a 1 to 2 page product contract summary for the dev team to read quickly.
* Draw the state diagram for an analysis run.
* Draw a preliminary entity map between Workspace, Repo, Snapshot, Run, and Report.
* Draft the UX wording for the privacy badge and consent banner.
* Create a list of open questions and mark which ones must be resolved before sprint 2.

## 6) Acceptance criteria

* A product contract document exists and has internal sign-off.
* A preliminary domain model exists and is shared between backend and UI.
* Clear wording exists for Strict Local and BYOK Cloud.
* There is no remaining ambiguity about what is in scope and out of scope for phase 1.
* Later tickets can directly reference the object names and state names from this document.

## 7) Out of scope

* Pixel-perfect UI details.
* Specific prompt design for each report section.
* Detailed accuracy evaluation model.

## 8) Risks / watchpoints

* If the privacy wording is finalized carelessly, many parts of the app will need to be revised later.
* If the domain model uses vague naming, the data layer and UI layer will drift apart.

## 9) Expected deliverables

* Foundation decision document
* Entity map
* Analysis run state diagram
* Privacy wording draft

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
