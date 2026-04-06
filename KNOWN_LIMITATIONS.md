# CodeSpectra — Known Limitations

Internal release reference. Updated for v0.1.0.

---

## Language Support

### Symbol Extraction (Repo Map)

| Language | Extractor | Quality |
|---|---|---|
| Python | stdlib `ast` | High — exact line numbers, full signatures, nested classes |
| TypeScript | tree-sitter | High — classes, interfaces, types, enums, arrow functions |
| JavaScript | tree-sitter | High — same walker as TS |
| Go | tree-sitter | High — functions, methods with receiver type, structs, interfaces |
| Java | tree-sitter | High — classes, interfaces, enums, methods, constructors |
| Rust | tree-sitter | High — structs, enums, traits, impl methods |
| C | tree-sitter | Medium — functions, structs, enums, typedef; complex macros may fail |
| C++ | tree-sitter | Medium — same as C + class specifiers; templates not fully resolved |
| Other | Regex fallback | Low — class/function patterns only, no nested detection |

**Not supported:** Ruby, PHP, Swift, Kotlin, Scala, Dart, Elixir, Haskell, and most others fall back to regex (limited quality).

### Structural Graph

- Import graph is **file-level only** — no function-call graph.
- Cross-file name resolution (stack graphs) is not implemented. "Find all usages" does not exist.
- Dynamic imports (`importlib`, `require()` with variable) are not tracked.
- Submodule/monorepo boundaries are not handled specially.

### Analysis Agents

- All agents are LLM-powered. Output quality depends heavily on the model used.
- Small/fast models (< 7B parameters, 8K context) may produce shallow or incomplete sections.
- Recommended minimum: GPT-4o-mini, Claude Haiku, Gemini Flash, or Llama 3 8B (Ollama).
- For best results on large repos: GPT-4o, Claude Sonnet, or Gemini Pro.

---

## Privacy & Cloud Mode

- **Strict Local mode**: No code leaves the machine. Only local Ollama/LM Studio models supported.
- **BYOK Cloud mode**: Code excerpts (chunks, up to ~375 tokens each) are sent to the configured cloud provider. The user must explicitly consent before first cloud run.
- API keys are stored in the local SQLite database (`userData/codespectra.db`). They are masked in API responses but stored in plaintext in the DB — do not share or back up the DB file to untrusted locations.
- Log files are written to `userData/logs/`. API keys and tokens are redacted from log output, but exception stack traces may occasionally include partial URLs.

---

## Repository Handling

- Maximum practical repo size: ~50K files. Larger repos will index but analysis may be slow.
- Binary files, generated files, and files matching ignore patterns are excluded from indexing.
- Repositories without Git are supported (no branch picker available).
- Shallow clones (`--depth=1`) are the default. Full clone is optional but much slower on large repos.
- Git submodules are not automatically followed.

---

## Performance

- **Build Graph**: O(n²) edge computation — repos with > 5K source files may take 30–90 seconds.
- **Prepare Snapshot (Index)**: Scales linearly with file count. ~1K files ≈ 2–5 seconds.
- **Analysis Run**: Depends on LLM response time. Each agent makes 1–2 LLM calls. Expect 1–5 minutes for a full scan with a cloud model, 5–20 minutes with a local model.
- C++ native acceleration is available for graph centrality, SCC detection, and keyword scanning. If the native module is not loaded, Python fallbacks are used (2–5× slower for large repos).

---

## Packaging / Installation

- **Windows**: NSIS installer for x64. Requires Windows 10 or later.
- **macOS**: DMG. Universal (x64 + arm64). Requires macOS 12+.
- **Linux**: AppImage (x64).
- The Python backend `.venv` is **not bundled** in the installer. A bundled Python runtime (PyInstaller) is required for production builds. Dev builds require Python 3.11+ installed separately.
- No automatic update mechanism in v0.1.0.

---

## UI / UX

- Report viewer does not support side-by-side comparison of two runs yet (planned in a future ticket).
- Graph visualization is a basic force-directed layout — large graphs (> 200 nodes) become unreadable. A node-circle/graph-DB-style layout is planned (RPA-052).
- The Retrieval Debug panel is a developer tool, not a user-facing feature.

---

## Out of Scope for v0.1.0

- Public-facing release / app store distribution
- Formal security audit
- Bitbucket integration (planned RPA-024)
- Evaluation harness / quality gates (planned RPA-050)
- Automatic re-indexing on file change
- Multi-workspace collaboration
- Team/organization features
