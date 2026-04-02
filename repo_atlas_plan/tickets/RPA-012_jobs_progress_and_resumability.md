# RPA-012 - Build Job Orchestration, Progress Logging, Cancellation, and Resumability

* **Epic**: Platform
* **Priority**: P1
* **Relative Estimate**: M
* **Milestone**: Milestone B
* **Dependencies**: RPA-010, RPA-011

## 1) Goal

Create the backbone for all long-running tasks such as clone, index, parse, report generation, and export. The user must always be able to see what the app is doing, what step it is on, and be able to cancel or retry.

## 2) Problem this ticket solves

Without a job layer, large repositories will make the app look frozen. Without progress logs, debugging accuracy and performance becomes almost blind.

## 3) Detailed scope

* Design the Job model: `id`, `type`, `scope`, `status`, `progress`, `started_at`, `finished_at`, `cancellable`, `retryable`, `logs`.
* Create an orchestrator that runs background jobs per workspace/repository.
* Create standardized progress reporting by steps: cloning / indexing / parsing / retrieval / generation / export.
* Create a cancellation token or cooperative cancellation mechanism.
* Create a reasonable retry/resume strategy for jobs that are not fully idempotent.
* Persist job history so the user can review previous runs.

## 4) Implementation notes

* A highly complex distributed-system-style job queue is not required. But a proper state machine and standardized log events are required.
* `JobEvent` stream and `JobSnapshot` state should be separated so the UI can subscribe and render more easily.
* Steps such as clone, index, and generate should use standardized progress step names shared across the whole app.

## 5) Breakdown subtasks

* Design the Job types enum and event schema.
* Create the in-app logger for run history.
* Create cancellation hooks for long-running steps.
* Create a UI panel for progress / logs / retry.
* Create tests for the failed job -> retry path.

## 6) Acceptance criteria

* When the user runs analysis, they can see step-by-step progress.
* A running analysis can be cancelled and the app does not end up in a broken state.
* Failed jobs produce sufficient logs for debugging.
* Run history is visible in the UI.

## 7) Out of scope

* Distributed job queue.
* Remote observability stack.

## 8) Risks / watchpoints

* If cancellation is not cooperative, the app may leave behind half-built indexes.
* If logging is too sparse, debugging accuracy and performance will become painful.

## 9) Expected deliverables

* Job orchestrator
* Progress event model
* Run history UI
* Cancellation / retry support

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
