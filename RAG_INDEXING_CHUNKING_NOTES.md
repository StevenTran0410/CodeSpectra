# RAG Indexing & Chunking Notes

Practical reference for how CodeSpectra builds retrieval context for analysis agents.

---

## 1) Goal

For each analysis section, fetch the most complete and relevant code context that is:

- evidence-traceable (file / chunk / reason codes)
- function-boundary aware (no truncated functions mid-context)
- sized to fill the LLM's context window efficiently, not just "enough to not crash"

---

## 2) Indexing flow

`backend/domain/retrieval/service.py`

1. Read `manifest_files` rows for a snapshot.
2. Skip noisy categories (`generated`, `asset`, `secret-risk`, `other`).
3. Read file content (lenient UTF-8).
4. Normalize whitespace.
5. Split into overlapping chunks by language/category-aware target size.
6. Store **full chunk content** into `retrieval_chunks.content`.
7. Store lightweight lexical preview rows into `retrieval_indexes`.

Chunks are stored with full content — no pre-truncation. Truncation only happens at prompt render time.

---

## 3) Chunking strategy

Target chunk sizes by file category/language:

| Category / Language | Target size (chars) |
|---|---|
| docs | 1800 |
| config | 1200 |
| test | 1400 |
| Python / TS / JS | 1500 |
| other source | 1300 |

Overlap between consecutive chunks: `max(120, target_size // 8)` chars.

---

## 4) Function-boundary expansion (new)

After scoring and selecting the top-N chunks, the system detects whether a chunk **ends mid-function**:

- **Brace-delimited languages** (JS/TS/Go/Java/Rust/C/C++): unbalanced `{` > `}` → mid-function.
- **Python**: indented last line + chunk contains a `def`/`class` → mid-body.

If mid-function is detected, the **next adjacent chunk** (same file, `chunk_index + 1`) is fetched and merged. Only **one hop** is done — expansion never chains. This means:

- If the next chunk completes the function: full function is sent to LLM.
- If the next chunk starts another function before ending the first: still included, but stops there.

Merged chunks are tagged with `boundary-expanded` in `reason_codes` and receive double the excerpt cap at prompt render time (`3000 chars` vs `1500 chars`).

---

## 5) Retrieval strategy

Two modes per section:

- **hybrid**: lexical term frequency + section-category hint + graph centrality + path semantics
- **vectorless**: same signals, stronger graph/path priors

Scoring signals per chunk:

| Signal | Description |
|---|---|
| `lexical-hit` | Term frequency in content + path match (+2 per path term) |
| `section-category-match` | +1.4 if file category matches section's expected category set |
| `graph-centrality-hint` | Up to +2.6 for top-ranked files in structural graph |
| `symbol-path-hint` | +0.6× for path tokens that match query terms |
| `vectorless-graph-prior` | Extra +1.8 for graph-central files in vectorless mode |

### Section budgets

Significantly increased to use available LLM context properly:

| Section | Token budget |
|---|---|
| Architecture | 10,000 |
| Feature Map | 10,000 |
| Important Files | 8,000 |
| Conventions | 7,000 |
| Glossary | 5,000 |

Prior budget was 1,600–2,600 tokens — roughly 4× under-utilising the LLM's context window. Modern models (GPT-4, Claude, Gemini) have 128K–200K context; even local 8B models support 8K+.

---

## 6) Prompt rendering (`render_bundle`)

`backend/domain/analysis/prompts.py`

| Parameter | Value |
|---|---|
| `limit` | 28 chunks |
| `excerpt_chars` | 1500 chars per chunk (~375 tokens) |
| `boundary-expanded` chunks | 3000 chars cap (double, to preserve full function body) |

Format per evidence item:
```
[N] file=path/to/file.py chunk=0 score=0.812 reasons=lexical-hit,graph-centrality-hint
<actual code excerpt here>

---
```

---

## 7) Symbol extraction (repo map)

`backend/domain/repo_map/service.py`

Symbols feed into graph construction, centrality scoring, and retrieval scoring.

| Language | Extractor | Notes |
|---|---|---|
| Python | `stdlib ast` module | Ground truth parser; tree-sitter as fallback on SyntaxError |
| TypeScript | tree-sitter 0.25 | Handles classes, interfaces, types, enums, arrow fns |
| JavaScript | tree-sitter 0.25 | Same walker as TS, TS-only nodes disabled |
| Go | tree-sitter 0.25 | Detects receiver type for method → `parent_name` |
| Java | tree-sitter 0.25 | Nested class stacks, constructor tracking |
| Rust | tree-sitter 0.25 | `impl` block context → methods get correct `parent_name` |
| C | tree-sitter 0.25 | Recursive declarator unwrapping for function names |
| C++ | tree-sitter 0.25 | Same as C + `class_specifier` |
| Other | Regex fallback | Class/function patterns only |

All languages use the new individual tree-sitter packages (`tree-sitter-python`, `tree-sitter-go`, etc.) — **not** the deprecated `tree-sitter-languages` bundle which was incompatible with tree-sitter >= 0.22.

---

## 8) RAG handoff into analysis agents

```
RunDirectorAgent
  └─ RetrievalBrokerAgent        ← LLM plans section-specific queries
       └─ RetrievalService        ← scores + selects chunks + boundary expansion
            └─ RetrievalBundle    ← passed to each section agent
```

Relevant files:
- `backend/domain/analysis/orchestrator.py`
- `backend/domain/analysis/retrieval_broker.py`
- `backend/domain/analysis/agent_pipeline.py`
- `backend/domain/analysis/prompts.py` (`render_bundle`)

Static analysis runs in parallel with retrieval planning:
- `static_risk.py` → god objects, circular imports, TODO hotspots, blast radius, test gap, config risk
- `static_convention.py` → naming conventions, folder roles, import boundary violations

Static findings are injected directly into agent prompts as ground-truth pre-computed context (not retrieved).

---

## 9) Evidence format (section output contract)

Each section agent must return:

```json
{
  "content": "...",
  "confidence": 0.0–1.0,
  "evidence_files": ["path/to/file.py"],
  "blind_spots": ["what the agent could not determine"],
  "details": { }
}
```

`EvidenceAuditorComposerAgent` validates all drafts, cross-checks evidence files, and assembles the final report.

---

## 10) Known gaps

- No vector embedding reranker (purely lexical + graph scoring). Adding embeddings would significantly improve recall for semantically related but lexically distant files.
- Cross-file name resolution (stack graphs) not implemented — symbol search finds definitions but not all call sites / references.
- No incremental re-indexing — changes to repo require full rebuild of chunks and graph.

---

## 11) Tuning checklist

When retrieval quality feels weak, tune in this order:

1. **Query set** — improve broker prompt / default queries per section.
2. **Chunk size / overlap** — increase for languages with large functions.
3. **Section budget** — increase if important files are being cut off.
4. **Scoring weights** — category/graph/path bonuses in `service.py`.
5. **render_bundle limit** — increase `limit=28` if budget headroom exists.
6. **boundary expansion** — check `reason_codes` in Retrieval Debug panel for `boundary-expanded` hits.

Do not tune blindly — compare before/after on a fixed repo.
