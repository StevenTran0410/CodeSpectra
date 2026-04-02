# RPA-023 - GitHub Integration: Authentication, Repository Discovery, and Repository Permission Flow

* **Epic**: Code Hosts
* **Priority**: P0
* **Relative Estimate**: M
* **Milestone**: Milestone C
* **Dependencies**: RPA-011

## 1) Goal

Allow the user to connect GitHub, view the repositories they have access to, select a repository, and prepare metadata for clone and sync operations.

## 2) Problem this ticket solves

Without GitHub integration, the phase 1 experience does not properly address the core use case of moving between internal/company projects.

## 3) Detailed scope

* Choose the MVP authentication flow for desktop: OAuth device flow or browser-based OAuth, depending on the most practical setup. GitHub supports device flow for non-web and headless applications.
* Store tokens securely through the secret service.
* Implement repository listing, search, and basic pagination.
* Implement the repository detail fetch needed for clone setup: default branch, visibility, owner, SSH/HTTPS URLs.
* Implement disconnect and reconnect flows.

## 4) Implementation notes

* GitHub Docs recommend GitHub Apps over OAuth apps for fine-grained permissions and short-lived tokens, but OAuth device flow is advantageous for MVP speed.
* The adapter/code host layer should be designed so that future backend auth changes do not require major UI changes.
* Repository discovery and clone authentication can remain separate logic paths if SSH is added later.

## 5) Breakdown subtasks

* Create the GitHub auth service.
* Create the repository discovery service and UI.
* Create the account summary card in settings.
* Create the revoke/disconnect path.
* Create unit tests for token persistence and repository mapping.

## 6) Acceptance criteria

* The user can connect GitHub successfully.
* The user can view and search repositories they have access to.
* The app stores the token securely and reconnects correctly after restart.
* Disconnect removes the token reference correctly.

## 7) Out of scope

* GitHub webhook sync.
* Advanced organization-level admin flows.

## 8) Risks / watchpoints

* OAuth flow and token refresh can create confusing UX if the wording is weak.
* Edge cases around repository permissions in org/private repos need early testing.

## 9) Expected deliverables

* GitHub auth service
* Repository discovery UI
* Connection management flow

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
