# 05 — Open Questions

Log of unresolved decisions. Each item must be resolved before the relevant ticket enters implementation.

---

| # | Question | Blocking ticket | Status | Resolution |
|---|---|---|---|---|
| OQ-01 | Should workspace be required before adding a local repo, or can repos exist globally? | RPA-025, RPA-030 | Resolved | Repos are global for now; workspace association deferred to RPA-030 |
| OQ-02 | Should `selected_branch` trigger an automatic `git checkout`, or should the analysis engine read file content via `git show <branch>:path`? | RPA-030, RPA-031 | Resolved | Use `git show <branch>:path` — never touch the working tree |
| OQ-03 | Where should cloud API keys be stored — SQLite `extra` field or OS keychain? | RPA-022, RPA-011 | Resolved (v1) | SQLite `extra` for v1; migrate to OS keychain in RPA-051 security pass |
| OQ-04 | Should the analysis job emit progress via SSE (Server-Sent Events) or WebSocket? | RPA-012, RPA-035 | Open | SSE preferred (simpler, HTTP/1.1 compatible); confirm in RPA-012 |
| OQ-05 | What is the maximum repo size (file count / disk size) before the app refuses or warns? | RPA-030, RPA-031 | Open | Current heuristic: warn if root has >200 entries or heavy dirs (node_modules, .venv); hard limit TBD |
| OQ-06 | Should embeddings be optional in v1 (analysis degrades gracefully without them)? | RPA-034, RPA-040 | Open | Leaning yes — lexical retrieval as fallback so analysis is not blocked on embedding model availability |
| OQ-07 | Report comparison across runs — same DB table or separate diff artifact? | RPA-043 | Open | TBD in RPA-043 design |
| OQ-08 | Submodule handling for local folders — index, skip, or warn? | RPA-031 | Open | Skip with warning in v1 |
| OQ-09 | Windows path encoding — spaces and Unicode in paths tested? | RPA-025, RPA-030 | Partially resolved | Folder picker tested with spaces; Unicode drive letters not yet tested |
| OQ-10 | Should DeepSeek use the generic OpenAI-compatible adapter or a dedicated one? | RPA-022 | Resolved | Dedicated adapter (same OpenAI wire format, different base URL and model list) |
