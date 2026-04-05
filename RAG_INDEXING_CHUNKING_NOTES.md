# RAG Indexing & Chunking Notes

Small practical reference for how CodeSpectra currently builds retrieval context for analysis agents.

---

## 1) Goal

For each analysis section, fetch the smallest useful code context that is:

- relevant to the section intent
- evidence-traceable (file/chunk/reason)
- cheap enough in token budget to run repeatedly

---

## 2) Current indexing flow

Implemented in `backend/domain/retrieval/service.py`.

1. Read manifest rows for a snapshot (`manifest_files`).
2. Skip noisy categories (`generated`, `asset`, `secret-risk`, `other`).
3. Read file content (lenient UTF-8).
4. Normalize whitespace.
5. Split into chunks by category/language-aware target size.
6. Store chunks into `retrieval_chunks`.
7. Store lightweight lexical preview rows into `retrieval_indexes`.

Output: retrieval index per `snapshot_id`.

---

## 3) Chunking strategy (current)

Heuristic chunk size by file type:

- docs: larger chunks
- config: smaller chunks
- tests: medium chunks
- code (py/ts/js): medium-large chunks
- others: default medium

Overlap is applied between consecutive chunks to reduce boundary loss.

Why this works now:

- keeps architecture text coherent enough for LLM
- avoids exploding chunk count on big repos
- still supports precise file/chunk references for evidence

---

## 4) Retrieval strategy (current)

Two modes:

- `hybrid`: lexical score + section category hint + graph centrality hint + path token hint
- `vectorless`: lexical + stronger graph/path priors, no embedding dependency

Section budgets (token caps) are fixed in retrieval service and enforced when selecting chunks.

Scoring signals include:

- query term frequency in content/path
- section-category match
- graph centrality ranking
- path semantics (`index`, `router`, `service`, etc.)

---

## 5) RAG handoff into analysis agents

Current orchestration path:

- `RunDirectorAgent` -> `RetrievalBrokerAgent` -> section agents -> auditor/composer

Code references:

- `backend/domain/analysis/orchestrator.py`
- `backend/domain/analysis/retrieval_broker.py`
- `backend/domain/analysis/agent_pipeline.py`
- `backend/domain/analysis/prompts.py`

Broker merges/deduplicates evidences and passes `RetrievalBundle` to each section agent.

Each section output must include:

- `content`
- `confidence`
- `evidence_files`
- `blind_spots`

---

## 6) Known gaps (important)

- No true embedding reranker yet (still lexical/graph heavy).
- Claim-level evidence validation is still basic.
- Section K (confidence/evidence narrative) is not fully matured.
- Prompt quality can still be improved for stricter evidence grounding.

---

## 7) Practical tuning checklist

When retrieval quality feels weak, tune in this order:

1. Query set quality per section (broker prompts/default queries).
2. Max results per section (`director` plan).
3. Chunk size/overlap heuristics.
4. Scoring weights (category/graph/path bonuses).
5. Evidence budget caps.

Do not optimize blindly. Track before/after on a fixed golden repo set.

