# RPA-025 - Local Folder Import: Open a Local Repository Without a Code Host

- **Epic**: Code Hosts
- **Priority**: P0
- **Relative estimate**: S
- **Milestone**: Milestone C
- **Dependencies**: RPA-011, RPA-030

## 1) Objective
Allow the user to open any local folder as a repository directly, without connecting to GitHub or Bitbucket, so that Strict Local Mode is genuinely self-contained from day one.

## 2) Problem this ticket addresses
The project description emphasizes "local folder import" as a first-class path, yet all current code-host tickets require OAuth or API tokens. A developer working with a local clone, an air-gapped codebase, or a private monorepo should be able to start analysis immediately without any auth flow.

## 3) Detailed scope
- Add a "Open Local Folder" entry point in the code-host connection screen and on the workspace home screen.
- Implement a native folder-picker dialog through Electron's `dialog.showOpenDialog`.
- Validate the selected path: check it is a real directory, check whether a `.git` folder is present, and warn clearly if it is not a git repository (still allow it with a notice).
- Extract basic repo metadata from local git config when available: remote URL, current branch, HEAD commit hash.
- Create a `LocalFolderSnapshot` that follows the same `RepoSnapshot` model used by the clone/sync engine (RPA-030), so all downstream indexing tickets remain unaware of the source type.
- Show a "local folder" badge in the privacy indicator (no code leaves device, no auth required).

## 4) Implementation notes
- The folder picker should default to the user's home directory or the last opened path, not the app directory.
- If `.git` is absent, the app can still index the folder for file manifest and symbol extraction; it just cannot produce a commit-hash-pinned snapshot. The UI should communicate this clearly.
- The `RepoSnapshot` model should carry a `source_type` field: `github`, `bitbucket`, `local_folder`. Downstream tickets must not assume `source_type === github`.
- No credential entry or OAuth flow should appear for this path.

## 5) Subtask breakdown
- Add folder-picker dialog and path validation service.
- Implement local git metadata reader (branch, HEAD, remote URL hint).
- Create `LocalFolderSnapshot` and wire it into the `RepoSnapshot` model.
- Add "Open Local Folder" UI entry in workspace home and code-host screen.
- Show appropriate badge and messaging when `.git` is absent.

## 6) Acceptance criteria
- User can select a local folder and proceed to analysis without any OAuth or token input.
- If the folder is a valid git repo, the snapshot captures branch and commit hash.
- If the folder has no `.git`, the app continues with a clear notice rather than erroring.
- All downstream indexing tickets (RPA-031, RPA-032, etc.) work against a local folder snapshot with no code changes.
- The privacy badge shows "Local only — no data leaves this device."

## 7) Out of scope
- Watching the folder for live file-system changes.
- Syncing a local folder back to a remote.
- Submodule resolution for local folders.

## 8) Risks / watch points
- Users may select very large monorepos by accident; preflight size warning should be considered.
- Path resolution on Windows vs macOS may need testing (spaces in path, network drives).

## 9) Expected deliverables
- Folder-picker dialog integration
- Local git metadata reader
- `LocalFolderSnapshot` model extension
- "Open Local Folder" UI entry points

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
- Test with a folder that has no `.git` and confirm the app does not crash.
- Test with a folder path containing spaces on Windows.
