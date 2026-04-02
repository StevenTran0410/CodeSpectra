# RPA-011 - Design Local Storage, Secure Secrets, and the Settings Service

* **Epic**: Platform
* **Priority**: P0
* **Relative Estimate**: M
* **Milestone**: Milestone B
* **Dependencies**: RPA-010

## 1) Goal

Provide a proper local-first foundation for metadata, settings, and secrets so provider and code host tickets can integrate safely.

## 2) Problem this ticket solves

If tokens and API keys are carelessly stored in config files or plaintext SQLite, the app becomes a delayed security bomb. Also, if settings are scattered across the system, UX and migration will become painful.

## 3) Detailed scope

* Design the SQLite schema or persistence layer for workspace metadata, repository connections, analysis runs, and report metadata.
* Integrate secure secret storage through an OS keychain / safe storage abstraction.
* Design a unified Settings Service for both app-level settings and workspace-level settings.
* Define secret references so the app stores only a secret handle/reference, never the raw secret in the database.
* Support rotating, deleting, and validating secret entries.
* Design data retention settings: cache path, max cache size, cleanup policy.

## 4) Implementation notes

* Secret handling must have a clear API: `setSecret`, `getSecret`, `testSecret`, `deleteSecret`, `listSecretHandles`.
* The settings UI should allow the user to see which secrets exist without showing the raw key.
* Default settings, workspace settings, and repository settings must be separated to avoid chaotic overwrites.

## 5) Breakdown subtasks

* Design DB schema v0.
* Create the DB migration framework.
* Integrate a secure storage library appropriate for the desktop stack.
* Build the settings repository/service.
* Create a minimal settings UI for cache path and privacy defaults.

## 6) Acceptance criteria

* API keys and tokens are not stored in plaintext in SQLite or export logs.
* Metadata still persists correctly after app restart.
* Secrets can be added, deleted, and rotated through the UI or service layer.
* Settings have a clear hierarchy and load according to the correct precedence.

## 7) Out of scope

* Cloud sync of settings across multiple machines.
* Enterprise credential policy.

## 8) Risks / watchpoints

* Secret migration across OSes or environments may be difficult if the chosen library is unreliable.
* If the settings hierarchy is vague, app behavior will become hard to predict.

## 9) Expected deliverables

* Persistence layer
* DB schema + migrations
* Secret service
* Settings service/UI

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
