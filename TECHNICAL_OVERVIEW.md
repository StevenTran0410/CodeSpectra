# CodeSpectra — Technical Overview

> Prepared for internal demo. Covers system design, key technical decisions, and engineering highlights.

---

## 1. What the system does

CodeSpectra is a **desktop codebase intelligence workbench**. You point it at a repository (local folder or any Git URL), it builds a structured semantic index of the code, then runs a multi-agent LLM pipeline to produce a 11-section onboarding report.

Target use case: a developer joining a large, unfamiliar codebase. In practice also useful for periodic audit — "what has changed, where did debt accumulate, what is the riskiest part right now."

The report covers: project identity, architecture layers, folder structure, coding conventions, anti-patterns, feature map, important files, recommended reading order, domain glossary, risk hotspots, and a meta-audit of all previous sections.

---

## 2. System architecture

```
┌────────────────────────────────────────────┐
│  Electron shell (Node.js)                  │
│  Window management · IPC bridge · OS keychain │
└──────────────┬─────────────────────────────┘
               │ HTTP localhost
┌──────────────▼─────────────────────────────┐
│  Python / C++ backend  (FastAPI + asyncio) │
│                                            │
│  ├── Indexing pipeline                     │
│  │     Tree-sitter AST parsing             │
│  │     Native C++ graph extension          │
│  │     SQLite chunk store (WAL mode)       │
│  │                                         │
│  ├── Retrieval service                     │
│  │     Hybrid lexical + structural scoring  │
│  │     Boundary-aware chunk expansion      │
│  │                                         │
│  └── Analysis pipeline                     │
│        11 LLM agents (A–K)                 │
│        Haystack AsyncPipeline (DAG)        │
│        PipelineMemoryContext               │
└──────────────┬─────────────────────────────┘
               │ IPC
┌──────────────▼─────────────────────────────┐
│  React renderer  (Vite + TypeScript)       │
│  Live per-section streaming cards          │
└────────────────────────────────────────────┘
```

The Electron shell is intentionally thin — it spawns the Python server, bridges OS-level access (file dialogs, keychain for API keys), and routes IPC to HTTP. All analysis logic lives in Python.

---

## 3. Indexing layer

### 3.1 Symbol extraction with Tree-sitter + AST

The indexer extracts symbols (functions, classes, methods, interfaces, enums, type aliases) from source files using two paths:

| Language | Method |
|---|---|
| Python | Native `ast` module — full fidelity, respects decorators and nested classes |
| TypeScript / JavaScript | Tree-sitter with language-specific walkers for arrow functions, class methods, interface declarations, type aliases, enums |
| C / **C++** | Tree-sitter (`tree_sitter_cpp`) — handles `function_definition`, `struct_specifier`, `class_specifier`, template declarations, namespaced symbols |
| Go, Java, Rust | Tree-sitter with per-language walkers |
| Other | Regex fallback |

**C++ specifically**: Tree-sitter is used to parse `.cpp`, `.cc`, `.cxx`, `.hpp` files. It correctly handles brace-heavy syntax where regex would fail — especially nested namespaces, template bodies, and class member definitions. Symbols from C++ files feed into the same structural graph as all other languages.

### 3.2 Native C++ extension — four accelerated functions

A single C++ extension module (`backend/native/graph_native.cpp`, compiled with pybind11 via `build_native_graph.py`) exports **four functions** used at different stages of the pipeline:

| Function | Used in | What it does |
|---|---|---|
| `compute_scores` | `StructuralGraphService.build()` | Takes the full import-edge list, counts in-degree and out-degree per file, scores each file as `indegree×3 + outdegree`, returns sorted list. The top 30 files become `top_central_files` — a permanent signal fed into every retrieval query for that repo. |
| `expand_neighbors` | `StructuralGraphService.neighbors()` | BFS expansion up to N hops from a seed file, bounded by a node limit. Returns the local subgraph (nodes + edges). Used by the frontend graph viewer and for subgraph-context retrieval. Uses `std::unordered_map` adjacency list for O(1) per-hop lookups. |
| `compute_scc` | `static_risk.py` — circular import detection | Iterative Tarjan's strongly-connected-components on the directed import graph. Returns all SCCs with ≥ 2 nodes, sorted by size descending. Detects circular import chains that Python's runtime hides but that become maintenance risks. The iterative implementation avoids C++ stack overflow on deep graphs. |
| `scan_keywords_bulk` | `static_risk.py` — TODO/FIXME hotspot scan | Word-boundary-aware substring scan across all (rel_path, content) pairs for a list of keywords. Returns only matches with count > 0. Faster than Python regex on large file sets — handles hundreds of files in a single pass with `std::string::find` + boundary checks. |

All four have a Python fallback path. If the native module is not built, `static_risk.py` falls back to `re.findall`, and `compute_scc` falls back to a pure-Python Tarjan implementation. `compute_scores` and `expand_neighbors` in `StructuralGraphService` raise `RuntimeError` without the native module — they are considered mandatory for the graph feature.

The build is a one-time step: `python scripts/build_native_graph.py` compiles with `/O2 /std:c++17` (MSVC on Windows) or `-O3 -std=c++17` (GCC/Clang on Linux/macOS).

### 3.3 Chunking strategy

Files are split into overlapping chunks with sizes tuned per category and language:

| Category | Target chunk size |
|---|---|
| Docs | 1800 chars |
| Config | 1200 chars |
| Test | 1400 chars |
| Python / TS / JS source | 1500 chars |
| Other source | 1300 chars |

Overlap is proportional to chunk size (typically 20–25%) to avoid losing function signatures at boundaries.

Chunks are stored in full in SQLite. Truncation happens only at prompt-render time, preserving the full raw text for future reuse.

---

## 4. Retrieval system

### 4.1 Why no vector database

CodeSpectra does **not** use a vector database or embedding model. The retrieval is entirely lexical + structural:

- No GPU or embedding model download required — works out of the box with no extra setup
- Pure-Python except for the C++ keyword scanner — no binary dependencies like FAISS, Annoy, or Chroma
- Scoring is fully interpretable: every rank signal is a named float you can log and tune
- Works well on small/medium repos where dense embeddings overfit to generic code vocabulary

The tradeoff: semantic similarity queries ("find code that handles authentication") work less well than keyword queries ("find files containing JWT or Bearer"). In practice, the agent query sets are engineered to be concrete enough that this is not a bottleneck.

### 4.2 Retrieval flow: from query to prompt context

A full retrieval call for one agent section follows these steps:

```
1. Agent calls RetrievalService.retrieve(RetrieveRequest(snapshot_id, section, queries, max_results=30))
2. Service loads all chunks for that snapshot from SQLite (category-filtered)
3. For each chunk, compute a score from multiple signals
4. Sort by score, take top N (up to max_results)
5. For each top chunk, run boundary expansion check (see 4.3)
6. Pass the bundle to render_bundle() which:
     - Applies per-section token budget cap
     - Truncates each chunk to excerpt_chars (2000 chars)
     - Gives boundary-expanded chunks a larger allocation
     - Annotates each excerpt with score + signal reason codes
7. Returns formatted string injected into the agent's user prompt
```

### 4.3 Scoring signals

Each chunk gets a composite score from five independent signals:

| Signal | Source | Description |
|---|---|---|
| **Lexical hit** | Chunk content vs query tokens | Tokenises both query and content into alphanumeric words, computes overlap ratio. Higher weight for exact multi-word matches. |
| **Path affinity** | File path vs query | Path tokens (directory names, filename stem) matched against query terms. A query like "architecture" boosts files under `src/arch/`, `core/`, `infrastructure/`. |
| **Category match** | File category vs section | Each section has a preferred category set (e.g. ARCHITECTURE prefers `source, config, infra`; GLOSSARY prefers `source, docs`). A chunk from a matching category gets a flat bonus. |
| **Graph centrality** | `compute_scores` output | The file's score from the C++ extension (`indegree×3 + outdegree`), normalised 0–1 over the whole repo. Files that many other files depend on (e.g. `src/db/client.ts`) rank higher for structural queries. |
| **Boundary expansion bonus** | Post-expansion tag | Chunks tagged `boundary-expanded` receive a small score boost so they are not displaced by shorter chunks with marginally higher raw scores. |

Final rank = weighted sum. Weights are not ML-learned — they are fixed constants tuned empirically against a set of reference repos.

### 4.4 Boundary-aware chunk expansion

When a selected top-N chunk likely ends mid-function, the retrieval service fetches the adjacent chunk:

- **Brace languages** (C++, JS, TS, Go, Java, Rust, C#): count `{` vs `}` in the chunk. If unbalanced (more opens than closes), the function body is not yet closed — expand.
- **Python**: scan last line's indentation depth against the `def`/`class` block that started near the top of the chunk. If the block's body never returns to column 0, expand.

The expanded chunk is tagged `boundary-expanded` and the prompt renderer gives it a larger excerpt character allocation than a normal chunk. This prevents the single most common RAG hallucination: LLM receiving a function whose body is cut off mid-logic, then inventing the rest.

### 4.5 Section budgets and render pipeline

Each section has a reserved token budget for its retrieval evidence:

| Section | Budget |
|---|---|
| Architecture | 14 000 tokens |
| Feature Map | 14 000 tokens |
| Important Files | 12 000 tokens |
| Conventions | 10 000 tokens |
| Glossary | 7 000 tokens |

`render_bundle()` enforces these budgets by iterating the scored chunk list and accumulating token estimates (`len(text) // 4`) until the budget is exhausted. Chunks are not truncated mid-sentence — each gets either its full excerpt allocation or is dropped entirely.

Budgets were increased 40–75% from initial values after log analysis showed truncated retrieval context on a 70K-line TypeScript codebase.

---

## 5. LLM analysis pipeline

### 5.1 Eleven dedicated agents (A–K)

Each section of the report is owned by exactly one LLM agent with:
- A fixed JSON output schema (enforced via prompt + `instructor`-style repair)
- Its own retrieval strategy (query set, section budget, result limit)
- A typed `run()` method — can be called standalone, not just from the pipeline

| Agent | Class | Responsibility |
|---|---|---|
| A | `ProjectIdentityAgent` | Repo name, purpose, runtime type, tech stack |
| B | `ArchitectureAgent` | Layers, frameworks, entrypoints, external integrations |
| C | `StructureAgent` | Folder-to-role mapping, structural overview |
| D | `ConventionsAgent` | Naming, async, DI, test, error-handling conventions |
| E | `ViolationsAgent` | Anti-patterns and convention violations (uses D's output as negative-space context) |
| F | `FeatureMapAgent` | Feature-level mapping with files and data flow |
| G | `ImportantFilesAgent` | Entrypoints, backbone files, risky files, read-first picks |
| H | `OnboardingAgent` | Recommended reading order for a new engineer (depends on G) |
| I | `GlossaryAgent` | Domain terms grounded in evidence files |
| J | `RiskAgent` | Risk hotspots, complexity signals, combined with static risk pre-analysis |
| K | `AuditAgent` | Meta-review of A–J: confidence per section, weakest sections, coverage quality |

Agent K receives **no retrieval** — it operates purely as a meta-analyst over the other agents' structured outputs.

### 5.2 Haystack AsyncPipeline — how it is used

**What Haystack provides here**: a typed DAG executor. You declare nodes (components) and directed edges (connections). Haystack schedules components whose inputs are satisfied, runs them concurrently subject to a concurrency limit, and routes outputs to the next layer. We do not use Haystack for LLM calls, prompts, or retrieval — those are all custom services.

**Component wrapper** (`_SectionAgentComponent`):

Each agent is wrapped in a `_SectionAgentComponent` that:
1. Implements Haystack's `@component` protocol with explicit typed ports (`run()` + `run_async()` must have identical signatures)
2. Accepts `ctx: dict` (shared pipeline context: provider_id, model_id, snapshot_id, mem_ctx, etc.) plus up to 10 optional upstream output ports (`identity_output`, `architecture_output`, ..., `risk_output`)
3. Calls the agent's `run_async()` coroutine via an injected `runner` lambda
4. On success, fires the `on_section_done` callback (streams section + timing to the UI)
5. On failure, calls the `fallback` lambda and emits `status="error"` — the pipeline keeps running

**Port wiring** (`pipeline.connect`):

```python
# Dependency edges — Haystack will not start B until A completes
pipeline.connect("project_identity.output", "architecture.identity_output")
pipeline.connect("project_identity.output", "structure.identity_output")
pipeline.connect("project_identity.output", "feature_map.identity_output")
pipeline.connect("architecture.output",     "feature_map.architecture_output")
pipeline.connect("conventions.output",      "violations.conventions_output")
pipeline.connect("important_files.output",  "onboarding.important_files_output")

# Auditor receives all 10 section outputs
pipeline.connect("project_identity.output", "auditor.identity_output")
pipeline.connect("architecture.output",     "auditor.architecture_output")
# ... × 10
```

Haystack performs strict type checking on connections at `connect()` time. Input ports typed `dict = None` accept both `dict` (from upstream) and `None` (default when not connected), which is how optional dependencies work.

**Execution**:

```python
async for partial in pipeline.run_async_generator(
    data={name: {"ctx": ctx} for name in component_names},
    include_outputs_from=component_names,
    concurrency_limit=11,   # configurable via ANALYSIS_PIPELINE_CONCURRENCY env var
):
    for component_name, output_map in partial.items():
        section_letter = _COMPONENT_TO_SECTION.get(component_name)
        sections[section_letter] = output_map["output"]
```

`run_async_generator` yields `partial` dicts each time any component finishes — allowing the outer caller to stream results incrementally to the UI without waiting for all 11 agents.

**Execution graph** (derived from the connect declarations):

```
Wave 0 (parallel):  ProjectIdentity · Conventions · ImportantFiles · Glossary · Risk
Wave 1 (parallel):  Architecture ← A    Structure ← A    Violations ← D    Onboarding ← G
Wave 2:             FeatureMap ← A + B
Wave 3:             Auditor ← all A–J
```

On a fast cloud model the total wall-clock time is dominated by Wave 0 (5 agents in parallel) + Wave 1 (4 agents in parallel) — typically 2–4 LLM calls deep rather than 11 sequential.

### 5.3 PipelineMemoryContext

Before the pipeline starts, a `prefetch_pipeline_context()` call fetches four shared data bundles in parallel:

| Bundle | Used by |
|---|---|
| `arch_bundle` (architecture retrieval) | ArchitectureAgent (B), StructureAgent (C) |
| `folder_tree` (flat directory listing) | ProjectIdentityAgent (A), StructureAgent (C) |
| `doc_files` (README, docs, wikis) | ProjectIdentityAgent (A) |
| `manifest_files` (package.json, pyproject.toml, etc.) | ProjectIdentityAgent (A) |

Each agent checks: *"do I already have this data in memory?"* and skips its own retrieval call if yes. If the prefetch fails (e.g. network issue, timeout), agents fall back to independent retrieval. The memory context is **opt-in, not required**.

Log output when context is active:
```
[pipeline] PipelineMemoryContext ready: arch_bundle=24 chunks, folder_tree=1840 chars, ...
[ArchitectureAgent] using arch_bundle from mem_ctx: 24 chunks
[StructureAgent] using arch_bundle from mem_ctx: 24 chunks
```

### 5.4 Cross-agent context propagation

Certain agents enrich their LLM prompt with upstream outputs:

| Connection | What is passed |
|---|---|
| A → B | Project identity summary prepended to Architecture prompt |
| A → C | Project identity summary prepended to Structure prompt |
| A → F | Project identity summary prepended to Feature Map prompt |
| B → F | Architecture summary prepended to Feature Map prompt |
| D → E | Conventions output used as negative-space context (what we expect, so E finds what deviates) |
| G → H | Important files list used to anchor the Onboarding reading order |

This creates a layered context model: later agents build on the structured understanding established by earlier ones, not just raw retrieval.

### 5.5 Token management

| Parameter | Before | After | Change |
|---|---|---|---|
| Total max output tokens (all agents) | 176 000 | 31 500 | −82% |
| `render_bundle` chunk limit | 28 | 40 | +43% |
| `render_bundle` excerpt chars | 1 500 | 2 000 | +33% |
| Agent `max_results` | 18–20 | 30 | +57% |

The reduction in output tokens came from right-sizing each agent based on observed output sizes from real reports — not guessing. Agents that produce terse structured JSON (e.g. ProjectIdentity: ~500 tokens actual) previously had 16 000 reserved. The freed budget is reallocated to retrieval (more chunks, larger excerpts).

---

## 6. LLM provider model

CodeSpectra is **provider-agnostic**. The user selects a provider + model at analysis start. All eleven agents execute on whichever model is selected.

| Mode | Providers | Inference location |
|---|---|---|
| Strict Local | Ollama, LM Studio | Entirely on-device. Nothing leaves the machine. |
| BYOK Cloud | OpenAI, Anthropic, Gemini, DeepSeek | Code context sent to selected provider. Explicit consent required. |

API keys are stored in the local SQLite DB and never logged.

---

## 7. Streaming UI

Sections appear incrementally as each agent completes — the UI does not wait for the full report.

Flow:
1. Agent completes → pipeline fires `on_section_done(section, status, duration_ms, output)`
2. Callback writes to a per-job event queue in memory
3. Frontend polls `GET /api/analysis/events/{job_id}` (SSE-style long-poll)
4. React renders the section card immediately, with agent timing displayed

The frontend has dedicated section cards (`SectionCardA.tsx` through `SectionCardK.tsx`) with typed renderers for each section's JSON schema.

---

## 8. Static analysis integration

Before the LLM pipeline runs, two static analyzers produce pre-computed signals:

- **`static_convention.py`**: scans the repo for code pattern signals — async usage consistency, error handling style, import organization, naming conventions at scale.
- **`static_risk.py`**: computes file-level complexity signals — cyclomatic complexity proxies, dependency fan-in/fan-out, unusual size patterns.

These outputs are injected directly into `ConventionsAgent` (D), `ViolationsAgent` (E), and `RiskAgent` (J) as grounding context — LLM sees both static evidence and retrieved code together.

---

## 9. Repository import model

Users can import repositories in two ways:

| Method | Behavior |
|---|---|
| Local folder | Directory is scanned in-place. No copy. |
| Git URL | `git clone` is executed to `%USERPROFILE%\CodeSpectra\repos\`. Works with any Git host. |

The source type (GitHub / Bitbucket / other) is auto-detected from the URL for display purposes only. The clone mechanism is identical regardless of host.

---

## 10. Data layer

- **Database**: SQLite with WAL mode (`aiosqlite` for async access)
- **Stored**: workspace configs, repo manifests, file snapshots, retrieval chunk index, analysis reports
- **Location**: `%APPDATA%\CodeSpectra\codespectra.db` (Windows default)
- **Report versioning**: payload at `version: 2`, with compatibility reader for older `sections_v2` format

---

## 11. Engineering highlights summary

| Area | What is notable |
|---|---|
| No vector DB | Custom lexical + structural retrieval. Fully interpretable, no embedding dependency. |
| C++ Tree-sitter | Symbol extraction for C/C++ codebases via `tree_sitter_cpp` — handles templates, namespaces, class members accurately. |
| Native C++ extension | Graph centrality (PageRank over file dependency graph) compiled as `.pyd`/`.so` for performance. Used in retrieval scoring. |
| Boundary expansion | Chunks that cut mid-function are automatically expanded one step and given larger prompt allocation. |
| Haystack DAG | Multi-wave parallel execution with typed ports, fan-in/fan-out wiring, no manual async coordination. |
| Memory context | Shared retrieval bundles pre-fetched once, reused across agents — eliminates redundant DB queries with graceful fallback. |
| Cross-agent chaining | Structured outputs of upstream agents flow as context into downstream agents (A→B, A→C, B→F, D→E). |
| Token right-sizing | 82% reduction in total reserved output tokens after measuring real report sizes, reallocated to richer retrieval context. |
| Strict local mode | Complete on-device analysis with Ollama/LM Studio — zero data exfiltration. |
| Streaming cards | Each section renders as it completes, with agent timing. No waiting for full report. |
