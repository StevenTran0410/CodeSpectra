# RPA-034 - Chunking, Embeddings, Lexical Search, and the Retrieval Layer for Section-Specific Context

* **Epic**: Intelligence
* **Priority**: P1
* **Relative Estimate**: L
* **Milestone**: Milestone D
* **Dependencies**: RPA-031, RPA-032, RPA-033, RPA-020

## 1) Goal

Create a retrieval layer smart enough to provide each report section with the right type of context, the right amount of context, and the right evidence.

## 2) Problem this ticket solves

Without strong retrieval, the report will either become too generic, too token-expensive, or prone to fabrication because it lacks the right context at the right point.

## 3) Detailed scope

* Design the chunking strategy for source code, config, docs, and tests.
* Integrate local embeddings by default; provider-based embeddings should be fallback/optional.
* Create a minimum lexical search index.
* Create hybrid retrieval: lexical + embeddings + graph hints + symbol hints.
* Create section-specific retrievers: architecture, conventions, feature_map, important_files, glossary.
* Create token budget logic for each section.

## 4) Implementation notes

* Continue treats embeddings as a separate role in the system and recommends local embeddings for local flows; this is a strong fit for Repo CodeSpectra.[R3]
* Repomix shows that token awareness is highly valuable when packaging repository context.[R5]
* Retrieval should return context bundles with reason codes so it is possible to debug later why a section pulled a specific file.

## 5) Breakdown subtasks

* Implement chunking rules per file class.
* Implement the embedding pipeline and storage.
* Implement the lexical search index.
* Implement hybrid ranking.
* Implement section-specific retrieval APIs.

## 6) Acceptance criteria

* Each section can call its own dedicated retriever.
* Context bundles include evidence metadata and token estimates.
* Local embeddings work in the default flow or have a clear fallback path.
* Retrieval logs/debug output are sufficient to investigate quality issues.

## 7) Out of scope

* A complex reranker model in v1.
* Cross-repository retrieval.

## 8) Risks / watchpoints

* The embeddings pipeline may become heavy if manifesting and chunking are not optimized first.
* Hybrid ranking without enough observability will be very difficult to tune.

## 9) Expected deliverables

* Chunking service
* Embedding index
* Lexical search index
* Hybrid retrievers

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
