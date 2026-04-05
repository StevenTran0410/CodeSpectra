# CodeSpectra Analysis Agents

This document defines the analysis agent topology for CodeSpectra onboarding reports.
It covers responsibilities, handoff flow, runtime decisions, and ticket mapping.

---

## 1) Design Goal

Produce an evidence-backed onboarding report that answers:

- What the system does and how it is organized
- Which files/symbols matter most
- Where to start reading
- Which conventions exist and where they are broken
- Which risky/complex areas need caution

---

## 2) Merged Agent Set (Optimized)

To keep runtime simpler and reduce orchestration overhead, related roles are merged.

### A. Run Director Agent

Merged from: moderator + run controller.

Responsibilities:
- Start/stop analysis run
- Decide section execution order
- Enforce token/time budget
- Handle retry/cancel behavior

Outputs:
- `run_plan`
- `run_state` (step statuses)
- run-level summary state

### B. Retrieval Broker Agent

Merged from: context retriever + retrieval strategy router.

Responsibilities:
- Build/use retrieval index (chunks + lexical)
- Pick retrieval mode per request (`hybrid` or `vectorless`)
- Return context bundles with evidence metadata

Outputs:
- `ContextBundle[]`:
  - file path
  - chunk excerpt
  - reason codes
  - token estimate

### C. Structure Intelligence Agent

Merged from: identity + architecture + repo narrative + importance radar.

Covers report sections:
- A) Project Identity Card
- B) Architecture Overview
- C) Repo Structure Narrative
- G) Important Files Radar
- H) Onboarding Reading Order

Outputs:
- section payloads for A/B/C/G/H
- evidence references for each claim

### D. Convention Intelligence Agent

Merged from: convention analyzer + forbidden/negative convention detector.

Covers report sections:
- D) Coding Convention & Team Style
- E) Forbidden Things / Negative Conventions

Outputs:
- inferred team conventions
- inferred boundary rules ("do not do X")
- outlier examples and violation candidates

### E. Domain & Risk Intelligence Agent

Merged from: feature map + glossary + risk/complexity detector.

Covers report sections:
- F) Functionality / Feature Map
- I) Glossary / Domain Terms
- J) Risk / Complexity / Unknowns

Outputs:
- feature map
- domain glossary
- risk matrix + hotspots + unknowns

### F. Evidence Auditor & Composer Agent

Merged from: evidence validator + confidence scorer + report composer.

Covers report section:
- K) Confidence & Evidence

Responsibilities:
- Verify claims have supporting evidence
- Assign confidence (`high` / `medium` / `low`)
- Record blind spots
- Compose final report artifact

Outputs:
- section K
- final report JSON/Markdown

---

## 3) Inter-Agent Flow

1. **Run Director** initializes run and section plan.
2. For each section group, **Run Director -> Retrieval Broker** requests context.
3. **Retrieval Broker** returns `ContextBundle`.
4. Intelligence agent (Structure / Convention / Domain-Risk) generates section drafts.
5. **Evidence Auditor & Composer** validates drafts and assigns confidence.
6. **Composer** emits final report.

Hard rule:
- Intelligence agents do not fetch repository data directly.
- They only consume context bundles from Retrieval Broker.

---

## 4) Runtime Architecture Decision (Python/C++)

### Orchestration choice

Decision: **custom Python orchestration (plain service pipeline), not LangGraph**.

Reasons:
- predictable control flow
- lower framework overhead
- easier debugging in local desktop runtime
- direct fit with existing FastAPI + job orchestration model

LangGraph can be re-evaluated later if routing/branching complexity grows significantly.

### C++ usage policy

- Agent coordination stays in Python.
- C++ is only for compute-heavy hotspots.

Current and planned C++ hotspots:
- graph neighbor expansion / graph scoring
- retrieval lexical scoring over large chunk sets
- optional rank-fusion kernels for hybrid retrieval

Do not use C++ for prompt orchestration, section writing, or report composition.

---

## 5) Mapping to Tickets

- RPA-033: structural graph + graph navigation hints
- RPA-034: chunking/retrieval/A-B comparison (hybrid vs vectorless)
- RPA-035: analysis run UX and model/privacy controls
- RPA-040/041/042/044: section generation from this agent stack

---

## 6) Interface Contracts (Recommended)

### ContextBundle

- `bundle_id`
- `section`
- `query`
- `mode`
- `budget_tokens`
- `used_tokens`
- `items[]` with:
  - `rel_path`
  - `chunk_id`
  - `excerpt`
  - `reason_codes[]`
  - `token_estimate`

### SectionDraft

- `section_id`
- `content`
- `claims[]`
- `evidence_refs[]`

### AuditResult

- `section_id`
- `confidence`
- `supported_claims`
- `unsupported_claims`
- `blind_spots[]`

### FinalReport

- sections A-K
- confidence summary
- evidence index

