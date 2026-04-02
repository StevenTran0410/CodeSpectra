# RPA-040 - Generate the Project Identity Card, Architecture Overview, and Onboarding Digest

* **Epic**: Analysis
* **Priority**: P0
* **Relative Estimate**: M
* **Milestone**: Milestone E
* **Dependencies**: RPA-002, RPA-032, RPA-033, RPA-034

## 1) Goal

Generate the three sections with the highest immediate onboarding value: what this repository does, what its high-level architecture looks like, and where a reader should start.

## 2) Problem this ticket solves

Users need value within the first 5 to 10 minutes. If there is no usable summary early enough, the app will be seen as a toy even if the underlying backbone is strong.

## 3) Detailed scope

* Implement the section generator for the Project Identity Card.
* Implement the section generator for the Architecture Overview.
* Implement the section generator for the Onboarding Digest / Reading Order.
* Map evidence and confidence for each section.
* Render sample output in the viewer with support for opening files and using the evidence drawer.

## 4) Implementation notes

* This is where accuracy and readability should be prioritized over coverage.
* The reading order must be highly practical: which file to read first, why it matters, and what the reader will understand after reading it.
* The architecture overview must distinguish at least: entrypoint, main modules, integrations, config, and persistence hints.

## 5) Breakdown subtasks

* Write the retrieval recipe for each section.
* Write the structured prompt / templating layer.
* Create confidence heuristics for these sections.
* Create viewer components for identity / architecture / reading order.
* Review output across 2 to 3 different sample repositories.

## 6) Acceptance criteria

* After reading the report, the user can answer what the repository does and where to start.
* Each section includes evidence and confidence.
* The reading order does not just list files, but also explains why they should be read.
* Output remains stable against the agreed schema.

## 7) Out of scope

* Deep feature mapping.
* Deep convention mining.

## 8) Risks / watchpoints

* If the summary is too generic, users will lose trust immediately.
* If the reading order is meaningless, the product loses a major differentiator.

## 9) Expected deliverables

* 3 section generators
* Viewer components
* Prompt / retrieval recipes

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
