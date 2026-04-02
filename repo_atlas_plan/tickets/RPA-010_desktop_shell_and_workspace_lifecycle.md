# RPA-010 - Build the Desktop Shell, Navigation, and Workspace Lifecycle

* **Epic**: Platform
* **Priority**: P0
* **Relative Estimate**: L
* **Milestone**: Milestone B
* **Dependencies**: RPA-001

## 1) Goal

Build a stable desktop app shell so the main flows have a home: home, provider setup, code host connections, repository setup, analysis run, report viewer, and settings.

## 2) Problem this ticket solves

Without a proper application shell, the team is likely to build isolated screens in an ad hoc way, create tangled state management, and make later integration of jobs, progress tracking, and reports much harder.

## 3) Detailed scope

* Choose and build the MVP desktop shell framework: Electron + React + TypeScript.
* Set up application-level navigation for the 7 core screens.
* Set up the global app layout: sidebar / top navigation / content area / status bar.
* Create the basic workspace lifecycle: create, open, rename, delete workspace.
* Set up local routing, error boundaries, loading skeletons, and empty states.
* Establish the minimum IPC boundary between the renderer and the main process.
* Set up app config bootstrap, app logs path, and user data path.

## 4) Implementation notes

* Electron security must be treated as a requirement, not something postponed until the end. Enable `contextIsolation`, apply CSP for local content, and do not load remote untrusted content.[R14]
* The UI can be very simple, but the information architecture must be correct so it does not need to be rebuilt later.
* There should be reusable components for status badges such as: local, cloud, synced, indexing, failed.

## 5) Breakdown subtasks

* Initialize the monorepo / app structure.
* Set up build/dev scripts, auto-reload, and the packaging skeleton.
* Create app shell components and route placeholders.
* Create the workspace store and the basic persistence layer.
* Create a status bar showing job state and the selected workspace/repository.

## 6) Acceptance criteria

* The desktop app can launch and navigate between the core screens.
* Workspace create/open/delete works reliably.
* There is no arbitrary remote content loaded in the renderer.
* Error boundaries and empty states are usable.
* Main process and renderer communicate through a minimal, controlled IPC layer.

## 7) Out of scope

* Detailed visual polish.
* Real provider or code host integrations.

## 8) Risks / watchpoints

* If the IPC design is careless, security hardening later will become difficult.
* If routes and state are structured incorrectly, the report viewer and job screens will be harder to extend.

## 9) Expected deliverables

* Desktop shell codebase
* Workspace lifecycle UI
* Electron security baseline configuration

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
