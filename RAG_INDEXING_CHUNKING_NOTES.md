# RAG Indexing and Chunking Notes

Reference for how CodeSpectra builds retrieval context for analysis agents.

## Indexing Flow

Primary service: `backend/domain/retrieval/service.py`

High-level flow:

1. Read snapshot manifest rows.
2. Skip noisy categories (generated/assets/secrets/noise).
3. Load file content with lenient text decoding.
4. Normalize and split into overlapping chunks.
5. Persist full chunk content in `retrieval_chunks`.
6. Persist lexical index rows for fast scoring.

Chunks are stored in full. Truncation happens only when rendering prompts.

## Chunking Strategy

Chunk target size is category/language aware and uses overlap to preserve continuity.

- Docs chunks are larger.
- Config chunks are smaller.
- Source/test chunks are balanced around function readability.
- Overlap is proportional to chunk size.

## Boundary-Aware Expansion

After top chunk selection, retrieval attempts one-step adjacent expansion when a chunk appears to end mid-function.

- Brace languages: detects likely unmatched braces.
- Python: detects likely indented block continuation.

Behavior:

- Expand at most one adjacent chunk.
- Tag expanded results as `boundary-expanded`.
- Keep expanded context larger at prompt-render stage.

This reduces cut-off function context without exploding token usage.

## Retrieval Modes

Two effective modes are used:

- `hybrid`: lexical + graph/path/category signals
- `vectorless`: non-embedding path with stronger structural priors

Signals include:

- lexical/path token hits
- section-category matches
- structural graph centrality hints
- path-based relevance heuristics

## Prompt Context Rendering

Prompt renderer: `render_bundle` in `backend/domain/analysis/prompts.py`

- Limits number of chunks per section.
- Applies per-chunk character caps.
- Gives larger caps to `boundary-expanded` chunks.
- Includes score + reason codes with each excerpt.

## Symbol Extraction Quality

Symbol extraction service: `backend/domain/repo_map/service.py`

Current quality profile:

- High quality for Python and major typed/compiled languages using AST or Tree-sitter.
- Regex fallback remains limited for unsupported languages.

This directly affects structural graph quality and retrieval quality.

## Agent Context Model (Current)

There is no centralized retrieval broker.

- Each section agent queries retrieval service directly.
- Agents can inject additional non-retrieval context (for example static convention/risk summaries).
- Orchestration is Haystack-based, but retrieval and provider routing remain custom services.

## Practical Tuning Order

When output quality is weak, tune in this order:

1. Agent query sets
2. Chunk size / overlap
3. Section budget caps
4. Retrieval scoring weights
5. Bundle render limits
6. Boundary expansion heuristics

Do not tune blindly. Compare against fixed repos and fixed prompts.

