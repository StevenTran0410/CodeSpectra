# RPA-030 - Repository Clone/Sync Engine with a Snapshot Model and Large-Repository Strategy

* **Epic**: Ingestion
* **Priority**: P0
* **Relative Estimate**: L
* **Milestone**: Milestone D
* **Dependencies**: RPA-012, RPA-023

## 1) Goal

Clone and sync repositories locally in a stable way, with a clear snapshot model strong enough to support incremental indexing and rerun analysis.

## 2) Problem this ticket solves

If the sync engine lacks discipline, reports may be generated on a half-updated codebase, the wrong branch, or a dirty state. That is a fatal error because every downstream insight will then be wrong as well.

## 3) Detailed scope

* Design the `RepoSnapshot` model: repo id, branch/ref, commit hash, local path, synced_at, status.
* Implement clone via HTTPS/SSH depending on the adapter/code host settings.
* Implement fetch/pull/update strategy for repositories that already exist locally.
* Implement basic locking to prevent two jobs from touching the same repo path at the same time.
* Implement preflight checks: disk path, existing dirty folder, permissions, submodule detection.
* Prepare the design for large repositories: shallow clone / partial clone / sparse mode depending on policy.

## 4) Implementation notes

* Partial clone does not need to be enabled by default on day one, but the architecture path should allow it.
* The snapshot must be tied to a clear commit hash, because the report needs reproducibility.
* The repository working copy and index artifacts should be stored in separate directories.

## 5) Breakdown subtasks

* Implement the clone service.
* Implement the sync/update service.
* Implement snapshot metadata persistence.
* Implement repo path locking.
* Implement preflight validation and error messages.

## 6) Acceptance criteria

* Repository clone succeeds through the basic GitHub flow.
* Updating/syncing an existing repository succeeds.
* Each analysis run knows exactly which snapshot it is running against.
* Concurrent sync operations do not corrupt the local repository path.

## 7) Out of scope

* Git write operations such as commit/push.
* A complete monorepo package-level sparse selection UI.

## 8) Risks / watchpoints

* Different credential flows across code hosts may make the clone engine more complex.
* Large repositories or submodules may create early edge cases.

## 9) Expected deliverables

* Clone/sync service
* Snapshot metadata model
* Repo path locking

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
