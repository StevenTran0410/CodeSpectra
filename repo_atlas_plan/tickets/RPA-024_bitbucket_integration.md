# RPA-024 - Bitbucket Integration: OAuth 2.0, API Token/SSH Clone Strategy, and Repository Discovery

* **Epic**: Code Hosts
* **Priority**: P1
* **Relative Estimate**: L
* **Milestone**: Milestone C
* **Dependencies**: RPA-011

## 1) Goal

Support Bitbucket well enough for enterprise use cases, while correctly reflecting the platform’s recent changes around app passwords and API tokens.

## 2) Problem this ticket solves

The real Bitbucket flow is more confusing than GitHub because repository discovery, API authentication, and Git clone authentication may use different mechanisms. If the app does not explain this clearly, users will fail at the first step.

## 3) Detailed scope

* Implement Bitbucket OAuth 2.0 for repository discovery and API access.
* Design the clone strategy to support either HTTPS token or SSH.
* Show correct messaging that API tokens are the new recommended path, while app passwords are being replaced and are no longer the default direction.
* Implement repository listing, search, workspace mapping, and repository details.
* Implement disconnect/reconnect flow and token rotation support.

## 4) Implementation notes

* The UI must clearly explain which token is used for API access and which credential is used for Git clone if they are not part of the same flow.
* Bitbucket Support states that API tokens can be used for Git commands, so this is the most practical path for HTTPS clone.
* If SSH is supported, the app should only store a reference/path or provide guidance, avoiding deep private key management in v1.

## 5) Breakdown subtasks

* Implement the Bitbucket auth service.
* Implement the repository discovery service.
* Implement the clone credential options UI: OAuth metadata / API token / SSH.
* Implement connection health check.
* Document known limitations directly in the UI/help text.

## 6) Acceptance criteria

* The user can connect Bitbucket to list/search repositories.
* The user can understand how to clone using either API token or SSH through the UI.
* The app does not encourage app passwords as the default legacy path.
* Reconnect/disconnect flow works correctly.

## 7) Out of scope

* Bitbucket Server/DC support.
* Advanced workspace admin flows.

## 8) Risks / watchpoints

* Bitbucket auth UX can easily create support burden if the wording is not clear enough.
* The platform may continue to evolve, so the abstraction layer must remain flexible.

## 9) Expected deliverables

* Bitbucket auth service
* Repository discovery UI
* Clone credential strategy UI/docs

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
