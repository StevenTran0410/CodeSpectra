# RPA-036 - Repository Setup Screen: Branch Selection / Sync Settings / Per-Repo Ignore Override

- **Epic**: Analysis UX
- **Priority**: P1
- **Relative estimate**: S
- **Milestone**: Milestone D
- **Dependencies**: RPA-023, RPA-024, RPA-025, RPA-030, RPA-031

## 1) Objective
Give the user a dedicated screen to configure each repository before indexing: choose the target branch or commit, set sync behavior, and override ignore rules per repository — so the clone/sync engine and manifest engine receive the right inputs.

## 2) Problem this ticket addresses
RPA-030 implements the clone/sync backend and RPA-031 implements the ignore engine, but neither ticket owns the UI that lets the user control them. Currently there is no place in the app for a user to choose a non-default branch, pin a specific tag, or add custom ignore patterns for one repository without affecting all others. This creates a problem for mono-repos, feature branches, and repos with unusual vendor layouts.

## 3) Detailed scope
- Build the Repository Setup screen accessible after selecting a repository from a code host or local folder.
- **Branch / ref picker**: list available branches and tags fetched from the code host API (or local git), allow the user to select one or type a specific commit SHA. Default to the remote default branch.
- **Sync settings**: choose between always pull latest vs pin to selected ref. Show the last synced commit hash.
- **Per-repo ignore overrides**: let the user add or remove custom ignore glob patterns on top of the workspace-level defaults. Show the effective combined list.
- **Submodule settings**: toggle whether submodules should be detected and noted (no deep indexing of submodules in v1).
- Show an estimated file count after applying ignore rules (using manifest delta from RPA-031) so the user can tune before committing to a full run.

## 4) Implementation notes
- Branch listing should be fetched lazily when the user opens the picker, with a loading state, not eagerly at workspace load time.
- The per-repo ignore pattern UI should distinguish between workspace-level defaults (read-only here, link to settings) and repo-level overrides (editable here).
- Settings from this screen are persisted to the `RepositoryConnection` record in the storage layer (RPA-011).
- If the user changes the target branch after a snapshot exists, the app should warn that the next sync will produce a new snapshot and invalidate cached indexes.

## 5) Subtask breakdown
- Build the branch/tag/ref picker component with API-backed listing.
- Build the sync settings toggle (always latest vs pinned ref).
- Build the per-repo ignore pattern editor and preview.
- Persist repository settings to the storage layer.
- Add a branch-change warning when a previous snapshot already exists.

## 6) Acceptance criteria
- User can select a non-default branch before running analysis.
- User can add repo-specific ignore patterns and see the effective combined ignore list.
- Settings persist correctly after app restart.
- Changing branch after an existing snapshot triggers a warning that the index will be rebuilt.
- Estimated file count after ignore rules is visible before committing to a run.

## 7) Out of scope
- Deep submodule indexing.
- Per-folder selective indexing beyond ignore patterns.
- Git history browsing.

## 8) Risks / watch points
- Branch listing API calls may be slow for large organizations; loading state and error handling are required.
- Ignore rule misconfiguration is easy to accidentally exclude important source files; the estimated file count preview helps but does not eliminate the risk.

## 9) Expected deliverables
- Repository setup screen
- Branch/ref picker component
- Per-repo ignore override editor
- Repository settings persistence

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
- Test changing branch after a snapshot exists and verify the warning appears.
- Test that per-repo ignore patterns do not leak into other repositories in the same workspace.
