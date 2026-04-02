# Repo CodeSpectra 

## Local-first Codebase Intelligence & Onboarding Assistant

### Main Project Description Document

Version: v0.1
Status: Proposed for implementation
Document language: English
Primary goal of the first phase: **analyze codebases extremely fast, clearly, and structurally** to help developers who are new to a project or returning to an old project after a long time away.

---

## 1) Executive Summary

**Repo CodeSpectra** is a local-first desktop app designed to:

1. connect to code hosts such as GitHub / Bitbucket,
2. pull or sync the codebase onto the user's machine,
3. index the codebase entirely on-device,
4. use either a local LLM or a cloud LLM selected by the user to generate a **structured codebase understanding map**.

What makes Repo CodeSpectra different is that it is **not an AI IDE for editing code**. It is more like an “onboarding brain” specialized in understanding, remembering, and explaining repositories.

### The problem it solves

In real development work, there are two extremely common pain points:

* joining a large codebase and not knowing where to start reading,
* having worked on a repository before, then returning later and having almost completely lost your “muscle memory” of its structure, conventions, flows, and critical files.

Repo CodeSpectra exists to answer questions such as:

* what does this repository actually do?
* where does the core business logic live?
* where is the entrypoint, and which files should be read first?
* what conventions is the team following?
* what anti-patterns or “unwritten rules” exist in this repository?
* which modules or files map to each major functionality?
* if I want to understand feature A, what reading order should I follow?

---

## 2) The most important product decision

## 2.1 Choose a **desktop app first**, not web-first

Clear recommendation: **build a desktop app first**.

### Why

* The app needs access to the local filesystem, git CLI, local cache, code indexes, background jobs, secret storage, and code host authentication flows. Desktop is naturally more suitable than web for these requirements.
* OpenAI recommends not deploying API keys in client-side environments such as browsers or mobile apps, and Google Gemini also advises against exposing API keys in client-side production environments.[R16][R17]
* If a web version is still desired later, it should only be:

  * a viewer for pre-generated reports, or
  * a client running through a local companion service / local gateway,
  * **not** the place where users directly enter cloud API keys for production usage.

### Recommended shell stack

**Recommended MVP**: Electron + React + TypeScript
Why:

* fast MVP development,
* strong JS/TS ecosystem for git tooling, parsing pipelines, local DB, and packaging,
* easier hiring and faster iteration.

**v2 / lower-footprint option**: Tauri + React
If a smaller footprint or tighter security boundary is desired later, gradual migration can be evaluated. Electron has a clear hardening checklist but must be secured carefully.[R14] Tauri also emphasizes trust boundaries and security-by-default in its own philosophy.[R15]

---

## 3) The contradiction that must be resolved from day one: “100% local” vs “using cloud API keys”

The original idea has two modes:

1. local LLM,
2. cloud LLM using the user’s API key.

These two modes are **not equivalent in terms of privacy**.

### The truth that must be stated clearly

* If using **local LLM mode**, it is possible to get very close to “zero code leaves the device.”
* If using **cloud API mode**, the selected context sent to the model will leave the user’s machine. This is a direct consequence of calling model APIs with requests containing prompts, messages, or content.[R18][R19][R20]

### The cleanest solution

Repo CodeSpectra must explicitly separate this into **2 privacy modes**:

#### A. Strict Local Mode

* Clone repo locally
* Index locally
* Embeddings locally
* Retrieval locally
* Generation locally
* No snippets/code/context leave the device

#### B. BYOK Cloud Mode

* Clone repo locally
* Index locally
* Retrieval locally
* Only send **selected evidence** (snippet / symbol / selected summary) to the cloud provider
* Must include:

  * a consent screen,
  * a clear warning badge stating “selected context may leave device,”
  * settings to redact paths / truncate snippets / block certain folders

This point is critically important because it affects UX, legal wording, documentation, and detailed implementation.

---

## 4) Product vision

Repo CodeSpectra should evolve into a **Codebase Intelligence Workbench** with 3 development layers:

### Phase 1: Codebase Understanding

Understand the repository, structure, conventions, feature map, critical files, and onboarding flow.

### Phase 2: Repo Q&A and Decision Support

Repository-aware Q&A, feature explanations, and technical answers grounded in evidence.

### Phase 3: Task Review / Planner

Based on repository context, suggest implementation plans, impact analysis, review checklists, and implementation drafts.

**Phase 1 is the mandatory backbone and must be built solidly.**
If Phase 1 is weak, Phases 2 and 3 will be “building a castle on sand.”

---

## 5) Target users

### Persona A. New developer joining the team

* does not know the domain,
* does not know the team’s implicit folder structure,
* does not know which files are the “front door” of the system.

### Persona B. Developer returning to an old repository

* once knew the repo but forgot the main flow,
* needs a fast refresh in 15 to 30 minutes.

### Persona C. Tech lead / senior developer

* wants a solid summary for onboarding new team members,
* wants visibility into hotspots, boundaries, and risky areas of the repository.

### Persona D. Technical PM / EM with technical inclination

* wants to understand feature structure and dependencies deeply enough to discuss them meaningfully.

---

## 6) Product principles

1. **Truly local-first**
   All clone data, indexes, caches, and reports stay on the user’s machine by default.

2. **Evidence-first**
   Every major conclusion must be grounded in files, symbols, or snippets. The app must not “sound convincing” without showing evidence.

3. **Readable over clever**
   Output must be clearer rather than “AI-written beautifully.” The goal is usefulness, not performance.

4. **Structured over free-form**
   Reports must have fixed sections so they are easy to scan and compare between runs.

5. **Progressive depth**
   Users should be able to view:

   * an ultra-short summary,
   * a standard summary,
   * a deep dive by feature / module / file.

6. **Explicit uncertainty**
   If the system can infer only part of the answer, it must show confidence and clearly state blind spots.

---

## 7) Benchmark research and ideas worth borrowing

## 7.1 Sourcegraph Cody

Cody emphasizes that answer quality depends on context retrieval. It combines keyword search, Sourcegraph search, and a code graph to retrieve context from the codebase.[R1]
**Ideas worth borrowing**:

* hybrid retrieval,
* repository-aware context,
* multi-source context,
* context window management based on question intent.

## 7.2 Aider repo map

Aider uses a compact “repo map” containing files, important classes, functions, and signatures so the LLM can understand the whole repository without loading the entire codebase.[R2]
**Ideas worth borrowing**:

* precomputed repo map,
* signature-first context,
* a backbone summary before injecting detailed snippets.

## 7.3 Continue

Continue uses embeddings for indexing and codebase awareness, and supports distinct model roles such as chat, embed, and reranker. Autocomplete and context selection can also depend on the current file, imports, and LSP definitions.[R3][R4]
**Ideas worth borrowing**:

* separate model roles,
* dedicated embedding / reranking models,
* structured context providers,
* avoiding total dependence on a single all-purpose model.

## 7.4 Repomix

Repomix packages a repository into an AI-friendly format with file summaries, directory structure, token counting, `.gitignore` awareness, and even compressed code views using Tree-sitter to reduce token usage.[R5]
**Ideas worth borrowing**:

* a canonical packed repository summary,
* token awareness,
* strong ignore rules,
* secret hygiene before sending context to a model.

## 7.5 Tree-sitter and AST-based tooling

Tree-sitter is a parser generator and incremental parsing library suitable for multi-language parsing and fast updates when files change.[R6]
ast-grep provides structural search, linting, and rewrite capabilities using ASTs across multiple languages.[R7]
**Ideas worth borrowing**:

* syntax-aware indexing,
* rule-based convention mining,
* structural pattern scanning instead of regex-only analysis.

## 7.6 LM Studio and Ollama

LM Studio provides a local server and endpoints compatible with OpenAI and Anthropic APIs.[R8]
Ollama exposes a local API by default at `http://localhost:11434/api`, with local access and no authentication by default.[R9]
**Ideas worth borrowing**:

* prioritize a local provider adapter that supports “OpenAI-compatible” APIs when possible,
* avoid lock-in to a single local runtime vendor.

### Benchmark conclusion

Most existing tools are:

* strong at chat / code assistance / editing,
* strong at “ask anything, get an answer,”
* but not deeply optimized for the use case of **onboarding + repository memory refresh + convention extraction + feature-to-file mapping**.

That is the opening where Repo CodeSpectra can win.

---

## 8) Differentiated value proposition

Repo CodeSpectra does not compete directly by being “better at AI coding.”
Repo CodeSpectra competes by answering these 6 onboarding questions extremely well:

1. **What does this repository do?**
2. **What are its architecture and boundaries?**
3. **What conventions does this team actually follow?**
4. **Where do the main features live?**
5. **Which files are worth reading first?**
6. **Which areas are risky or difficult to modify?**

In short:
**Cursor/Cody help you code with the repository. Repo CodeSpectra helps you understand the repository before you even dare to code in it.**

---

## 9) Scope of phase 1

## 9.1 Mandatory in-scope items

1. Local-first desktop app
2. Workspace management
3. LLM mode selection:

   * local
   * cloud BYOK
4. Code host connections:

   * GitHub
   * Bitbucket
5. Local clone / sync
6. Local codebase indexing
7. Structured codebase analysis report generation
8. Viewer to read the report and drill down by section
9. Export report as Markdown / JSON
10. Re-run analysis when the repository changes

## 9.2 Nice-to-have if capacity allows

1. Compare two scans
2. Important files radar
3. Glossary / domain term extraction
4. TODO / FIXME hotspot scan
5. Config / env / migration inventory
6. Quick mode vs Full mode

## 9.3 Out of scope for phase 1

1. code editing
2. auto commit / auto PR
3. AI agent that modifies code
4. detailed task planning from issues
5. PR review
6. live pair programming
7. collaborative cloud workspace

---

## 10) Core product functionality for phase 1

## 10.1 Repo ingestion

* authenticate with code host
* select repository / branch / commit / tag
* clone locally
* incremental sync
* support private repositories
* support submodule detection
* support shallow / partial clone for large repos

Git supports `--filter=blob:none` for partial clone and `--sparse` / sparse-checkout to reduce the number of materialized files initially.[R13]
=> Phase 1 does not have to enable this by default, but the design should prepare for very large repositories.

## 10.2 Local indexing pipeline

Suggested pipeline:

1. File discovery
2. Ignore filtering

   * `.gitignore`
   * binary files
   * generated files
   * build artifacts
   * vendor / dependencies
   * large lockfiles by rule
3. Language detection
4. Metadata extraction

   * file path
   * size
   * extension
   * checksum
5. Symbol extraction

   * class / function / interface / type / route / controller / service / command
6. Relationship extraction

   * import graph
   * file dependency
   * entrypoint candidates
   * config linkage
7. Chunking
8. Embeddings
9. Lexical index
10. Analysis artifact generation

## 10.3 Codebase analysis report

This is the heart of phase 1. The report should include at least these sections.

### A. Project Identity Card

* repository name
* domain / product area
* what this application is for
* inferred user/problem/business context from code/docs/config
* runtime type:

  * web app
  * backend service
  * mobile backend
  * monolith
  * monorepo
  * library
  * CLI
  * worker
  * cron / job system

### B. Architecture Overview

* main layers
* actual folder structure
* module boundaries
* main framework(s)
* entrypoints
* main services / adapters / controllers / jobs / routes
* external integrations
* config sources
* database / migration / queue / event hints

### C. Repo Structure Narrative

Not just a folder list. It must explain:

* which folders are domain
* which folders are infrastructure
* which folders are delivery layer
* which folders are shared/common
* which folders are test-only
* which folders are generated

### D. Coding Convention & Team Style

* naming conventions
* module naming
* error handling style
* dependency injection style
* logging style
* async / concurrency style
* testing style
* config / env style
* branching style of business logic
* whether the code is if/else-heavy or strategy/polymorphism-heavy
* class-based or functional
* file length/profile
* suspected anti-patterns the team avoids

### E. “Forbidden things” / Negative conventions

This is highly valuable and few tools do it well.
Examples:

* do not import directly from infra into domain
* do not call the repo layer from controllers
* do not place business logic inside route handlers
* do not use raw SQL in module X
* do not create services following pattern Y
* do not use mutable shared state in workers

This must be inferred from:

* structural rules inferred from the majority pattern,
* path boundaries,
* import patterns,
* outlier detection.

### F. Functionality / Feature Map

Each major functionality should include:

* inferred feature name
* related modules/files
* entrypoint
* supporting services
* data path
* related config
* related tests
* note on “if you want to understand this feature, read these in this order”

### G. Important Files Radar

It must answer:

* which file is the entrypoint
* which file is the backbone
* which file is critical configuration
* which file has high centrality
* which file is “dangerous to touch”
* which file should be read first

### H. Onboarding Reading Order

A kind of “repo reading playlist”:

1. read file A to understand the entrypoint,
2. then file B to understand wiring,
3. then file C to understand the domain,
4. then file D to understand persistence,
5. finish with tests / config / migrations.

### I. Glossary / Domain Terms

Extract frequently repeated business terms such as:

* invoice
* settlement
* wallet
* payout
* risk_check
* approval_chain
  ...
  and provide short explanations based on evidence.

### J. Risk / Complexity / Unknowns

* oversized modules
* god-object files
* heavy branching logic
* suspected circular imports
* TODO / FIXME hotspots
* generated code mixed with handwritten code
* weak coverage (if test layout can be inferred)
* risky configuration / scattered env usage

### K. Confidence & Evidence

Each section should include:

* confidence: high / medium / low
* supporting files
* snippets or symbol references
* known blind spots

---

## 11) Recommended output format

Each report section should use a schema like this:

```json
{
  "section_id": "architecture_overview",
  "title": "Architecture Overview",
  "summary": "...",
  "bullets": ["...", "..."],
  "evidence": [
    {"file": "src/app.ts", "symbol": "bootstrap", "reason": "entrypoint"},
    {"file": "src/routes/index.ts", "symbol": null, "reason": "route wiring"}
  ],
  "confidence": "medium",
  "unknowns": ["Unable to identify all queue consumers"],
  "followup_questions": ["Where was the payments module split from?"]
}
```

### Why a rigid schema is necessary

* easier UI rendering,
* easier export,
* easier testing,
* easier diffing between two runs,
* reduces long and unfocused output.

---

## 12) Suggested system architecture

## 12.1 High-level architecture

### Desktop shell

* Electron app
* local UI
* no remote untrusted content rendering
* enable hardening based on Electron’s security checklist.[R14]

### Local services inside the app

1. **Workspace Service**
2. **Secrets Service**
3. **Provider Service**
4. **Code Host Service**
5. **Repo Sync Service**
6. **Indexing Service**
7. **Analysis Service**
8. **Report Service**
9. **Export Service**

### Local storage

* SQLite for metadata
* file storage for clone/cache/report artifacts
* local vector index
* OS keychain for secrets/tokens/API keys

## 12.2 Recommended data model

* Workspace
* RepositoryConnection
* RepositorySnapshot
* AnalysisRun
* FileManifest
* SymbolRecord
* EdgeRecord
* ChunkRecord
* EmbeddingRecord
* ReportArtifact
* SectionArtifact
* SecretReference
* ConsentRecord

## 12.3 Suggested internal directory model

* `/workspaces/{workspaceId}/repos/{repoId}`
* `/workspaces/{workspaceId}/indexes/{repoId}/{snapshotId}`
* `/workspaces/{workspaceId}/reports/{analysisRunId}`
* SQLite DB stored separately
* secrets must not be stored as plaintext in the DB

---

## 13) Adapter strategy for model providers

## 13.1 A common abstraction is mandatory

Never let individual screens call different provider SDKs directly.
A standardized `LLMProviderAdapter` layer is required to normalize:

* list models
* validate credentials
* get capabilities
* generate text
* generate structured JSON
* embeddings (if the provider supports them)
* token estimation
* streaming
* cancellation

## 13.2 Capabilities matrix

Providers differ significantly, so capabilities must be represented explicitly:

* supports_structured_output
* supports_streaming
* supports_embeddings
* supports_tool_use
* max_context_known
* is_openai_compatible
* privacy_mode

## 13.3 Recommended strategic modes

### Local providers

* Ollama connector
* LM Studio connector
* generic adapter for other OpenAI-compatible local servers

LM Studio provides OpenAI-compatible and Anthropic-compatible endpoints.[R8]
Ollama exposes a local API by default on localhost.[R9]

### Cloud providers

* OpenAI
* Anthropic
* Gemini
* DeepSeek

DeepSeek explicitly documents compatibility with OpenAI SDK / base URL model adapters.[R20]
=> this is excellent for the abstraction design.

### Important decision

**Embeddings should not depend entirely on the primary chat provider.**
Recommended approach:

* default to local embeddings,
* generation can be local or cloud.

This helps:

* stabilize the indexing pipeline,
* avoid vendor lock-in,
* reduce costs,
* preserve the local-first model more effectively.

Continue also separates embedding roles and recommends local embedding models such as `nomic-embed-text` for local flows.[R3]

---

## 14) Adapter strategy for code hosts

## 14.1 GitHub

GitHub Docs clearly explain that GitHub Apps are often preferred over OAuth apps because they offer fine-grained permissions, better repository control, and shorter-lived tokens.[R11]
However, OAuth device flow is also available for headless or non-web applications.[R10]

### Practical proposal for phase 1

* **Repository discovery / metadata / permissions**: prioritize GitHub App or OAuth depending on implementation speed
* **Native desktop MVP**: OAuth device flow can be used first for faster delivery
* **Mid-term**: move toward GitHub App if a cleaner permission model is desired

## 14.2 Bitbucket

Bitbucket supports OAuth 2.0 authorization code flow for apps and integrations.[R12]
In addition, Bitbucket Cloud has been transitioning away from app passwords toward API tokens. New app passwords can no longer be created after 2025-09-09, and old app passwords will be disabled on 2026-06-09.[R21]
Bitbucket Support also states that API tokens can be used for Git commands and HTTPS clone operations.[R22]

### Practical proposal for phase 1

* Repo discovery: OAuth 2.0
* Git clone over HTTPS: API token or SSH
* The UI must clearly differentiate:

  * token used for API access,
  * token used for Git clone,
  * SSH path if the user wants a zero-password flow

## 14.3 Phase 1.5 / future

* GitLab
* Azure DevOps
* local folder import (without code host)
* “Open local folder” mode should be considered early, because onboarding a repository does not necessarily start through GitHub or Bitbucket.

---

## 15) Security and secret handling

## 15.1 Principles

* tokens / API keys must not be stored as plaintext in SQLite
* use OS keychain / secure storage
* the app should only load local UI content
* enforce contextIsolation / CSP / IPC validation for Electron.[R14]
* never print secrets in logs
* exports must not accidentally include secrets

## 15.2 Secret classes

1. Cloud LLM API key
2. Code host access token
3. Refresh token
4. Repository-specific HTTPS token
5. Optional SSH config reference (private key itself should not be stored)

## 15.3 Redaction / safe context layer

Before sending context to a cloud provider:

* block `.env`, secrets, credentials,
* ignore unnecessary generated lockfiles,
* mask obvious tokens,
* allow users to define denylist paths such as:

  * `secrets/**`
  * `certs/**`
  * `infra/terraform.tfvars`
  * `*.pem`

Repomix integrates secret checking into its packaging flow, which is a benchmark worth learning from.[R5]

---

## 16) Suggested UX flow (normal UI, functionality first)

The UI does not need to be fancy. It only needs to be clear, trustworthy, and low-friction.

## 16.1 Main screens

1. **Workspace Home**

   * repository list
   * sync/index status
   * recent runs

2. **Provider Setup**

   * Local LLM
   * Cloud BYOK
   * test connection
   * privacy badge

3. **Code Host Connections**

   * GitHub
   * Bitbucket
   * local folder import

4. **Repository Setup**

   * select repository
   * branch/tag/commit
   * sync settings
   * ignore settings

5. **Analysis Run Screen**

   * Quick scan / Full scan
   * Strict local / BYOK cloud
   * progress log
   * cancel / retry

6. **Report Viewer**

   * sidebar sections
   * top summary
   * evidence drawer
   * important files
   * reading order
   * export

7. **Settings**

   * cache path
   * data retention
   * secret management
   * privacy defaults

## 16.2 Important interactions

* The Evidence drawer must always show file paths and reasoning clearly
* Confidence badges must be visually prominent
* If a section has low confidence, provide a CTA such as “scan deeper” or “open related files”
* If cloud mode is used, show a badge stating “selected context leaves device”

---

## 17) Analysis execution modes

## 17.1 Quick Scan

Goal: produce usable results quickly
Includes:

* file manifest
* basic symbol index
* repo map
* project identity
* architecture sketch
* important files v1

## 17.2 Full Scan

Includes additional:

* chunking + embeddings
* feature map
* deeper convention mining
* risk / hotspot analysis
* onboarding reading order
* glossary

## 17.3 Compare Scan

* compare the current report with the previous report
* see where structure changed
* flag stale sections

---

## 18) Insight generation method to reduce hallucination

Repo CodeSpectra should not ask the model in one shot: “please describe this repository.”

Instead, the pipeline must be split:

1. **Static extraction first**

   * manifest
   * symbols
   * graph
   * config inventory
   * docs inventory

2. **Heuristic summaries**

   * folder roles
   * entrypoint candidates
   * high-centrality files
   * route/controller/service clusters

3. **Section-specific retrieval**

   * each section has different queries and context

4. **Structured generation**

   * force the model to return JSON matching a schema

5. **Evidence attachment**

   * reconnect files / symbols / snippets

6. **Confidence scoring**

   * based on number of evidence items, consistency between signals, and the degree of alignment between docs and code

### Example

* `architecture_overview` should use entrypoints + import graph + config + framework files
* `coding_conventions` should use AST pattern statistics + file sampling + naming statistics
* `feature_map` should use route files + controller/service clusters + docs + test names + domain terms

---

## 19) Heuristics and intelligence layers that should exist

## 19.1 File classification

Classify files into:

* source
* test
* config
* migration
* schema
* infra
* docs
* generated
* vendor
* asset
* secret-risk

## 19.2 Symbol extraction

Extract:

* class
* function
* method
* interface/type
* controller
* route
* service
* command
* job/worker
* repository/DAO
* event/consumer
* config object

## 19.3 Relationship graph

At minimum, it should contain:

* file import graph
* symbol-to-file mapping
* route-to-controller
* controller-to-service
* service-to-repository
* job-to-handler
* config-to-module
* test-to-target-file

## 19.4 Convention mining

Based on:

* folder naming frequency
* suffix/prefix patterns (`Service`, `Controller`, `UseCase`, `Handler`, `Repository`)
* import boundaries
* AST patterns
* exception handling shape
* dependency injection style
* framework decorators/annotations
* test naming shape

## 19.5 Important file scoring

File scores can be based on:

* indegree / outdegree
* entrypoint flag
* config flag
* symbol density
* route exposure
* file length
* churn (in later phases)
* mention frequency in docs/tests

---

## 20) High-value insights that should exist even in v1

This section enriches the original idea further.

### 20.1 “Read this repo in 20 minutes”

A very short onboarding section:

* 5 files to read first
* 3 modules to understand
* 2 places where people are likely to make mistakes

### 20.2 “Hidden rules”

Unwritten rules in the repository that are not formally documented.

### 20.3 “Feature slices”

Not just saying what each folder does, but saying:

* which route feature A goes through
* which service
* which repository
* which test

### 20.4 “Why this file matters”

For each important file, explain why it matters.

### 20.5 “What probably changed the most”

Later phases can add git history / hotspots.
Phase 1 only needs the architecture to support this later.

### 20.6 “What is likely generated vs hand-written”

Very useful in enterprise repositories with heavy code generation.

### 20.7 “Where business logic actually lives”

Many repos appear to place logic in services, but in practice the logic may live in controllers / jobs / query layers.
This section is highly useful.

### 20.8 “What not to touch casually”

A list of files/modules with large blast radius.

---

## 21) NFR - Non-functional requirements

### 21.1 Performance

* Quick scan of a medium-sized repo must produce the first useful result quickly
* Full scan can take longer but must show clear progress
* Incremental re-scan must be faster than the first full scan

### 21.2 Reliability

* analysis runs should support resume / retry
* the whole app must not crash if one parser fails
* one language parser failing must not collapse the entire pipeline

### 21.3 Privacy

* local storage only by default
* explicit consent when using cloud mode
* deleting a workspace should delete local artifacts based on user choice

### 21.4 Explainability

* every major section must include evidence
* confidence must be shown clearly
* unknowns must be visible

### 21.5 Extensibility

* easy to add new providers
* easy to add new code hosts
* easy to add new sections
* easy to add new parsers / language support

---

## 22) Suggested success metrics

## 22.1 Product metrics

* time-to-first-usable-summary
* time-to-first-full-report
* repo re-scan delta time
* export success rate
* provider connection success rate

## 22.2 User value metrics

* do new developers feel more confident after 20 minutes of reading the report?
* does the number of random file openings before finding the core decrease?
* do leads believe the report reflects the real structure accurately?
* do users return to the report after 1 week / 1 month?

## 22.3 Quality metrics

* section factual accuracy
* evidence coverage
* hallucination rate
* parse coverage by language
* feature map usefulness rating

---

## 23) Major risks and mitigation

## 23.1 Hallucination in summaries

**Mitigate with**:

* section-specific retrieval
* schema-based output
* confidence model
* evidence attachment
* static heuristics before LLM generation

## 23.2 Multi-language repositories that are too complex

**Mitigate with**:

* support a prioritized set of languages well in phase 1
* use manifest-only fallback for languages without deep parsing support

## 23.3 Enterprise repositories that are too large

**Mitigate with**:

* quick scan
* partial clone / sparse strategies
* incremental indexing
* strong ignore rules
* sample first, deep scan later

## 23.4 Secret leakage in cloud mode

**Mitigate with**:

* denylist
* masking obvious secrets
* evidence budget
* consent banner
* clearly separated Strict Local Mode

## 23.5 Provider fragmentation

**Mitigate with**:

* adapter layer
* capabilities matrix
* structured output contract
* local embeddings by default

---

## 24) Most practical MVP cut

If the goal is to launch a first version quickly while still delivering real value, the MVP only needs:

1. desktop shell,
2. local workspace,
3. GitHub + local folder import,
4. local LLM through Ollama or LM Studio,
5. repository clone / sync,
6. file manifest + symbol index,
7. project identity card,
8. architecture overview,
9. important files radar,
10. onboarding reading order,
11. Markdown export.

After that, add:

* cloud providers,
* Bitbucket,
* embeddings,
* deeper feature map,
* advanced convention mining.

However, because the original idea explicitly included BYOK cloud providers and GitHub/Bitbucket, the ticket plan below still includes them, but arranged with proper dependencies.

---

## 25) Proposed high-level roadmap

### Milestone A. Foundation

* product contract
* report schema
* desktop shell
* secrets/storage

### Milestone B. Connections

* local providers
* cloud providers
* GitHub
* Bitbucket

### Milestone C. Intelligence Backbone

* clone/sync
* file classification
* symbol extraction
* graph
* embeddings/retrieval

### Milestone D. Analysis UX

* project identity
* architecture
* conventions
* feature map
* important files
* viewer/export

### Milestone E. Hardening

* evaluation harness
* security
* packaging

---

## 26) Decisions I recommend locking early

1. **App first, not web-first**
2. **Desktop MVP with Electron**
3. **Privacy mode clearly split into Strict Local vs BYOK Cloud**
4. **Local embeddings by default**
5. **Report output must be structured JSON + Markdown render**
6. **Evidence & confidence are mandatory, not optional**
7. **Support GitHub first, with Bitbucket following closely**
8. **Open local folder mode should come early**
9. **Do not build code editing in phase 1**
10. **Feature map + convention mining are strong differentiators and should not be removed**

---

## 27) Future backlog after phase 1 is stable

1. Repo chat / ask mode
2. Task planner by issue / ticket
3. Impact analysis
4. Feature-based review checklist
5. “Explain this diff in repository context”
6. Team knowledge pack / onboarding pack export
7. Multi-repo workspace
8. Git history hotspot / ownership / churn map
9. PR review assist
10. ADR extraction / architecture drift detection

---

## 28) Short conclusion

This idea is **worth building** and has a clear market opening if positioned correctly:

* do not turn it into yet another AI IDE,
* turn it into a **repository understanding machine**.

The critical success factor is not flashy UI. It is:

* a strong indexing pipeline,
* strong retrieval,
* a strong structured report,
* clear evidence/confidence,
* and correct positioning: **help people understand a repository quickly, deeply, and systematically**.

If built correctly, Repo CodeSpectra becomes the first thing a developer opens when touching an unfamiliar codebase.

---

## Research references

* [R1] Sourcegraph, “Cody Context”
  [https://sourcegraph.com/docs/cody/core-concepts/context](https://sourcegraph.com/docs/cody/core-concepts/context)

* [R2] Aider, “Repository map”
  [https://aider.chat/docs/repomap.html](https://aider.chat/docs/repomap.html)

* [R3] Continue Docs, “Embed Role”
  [https://docs.continue.dev/advanced/model-roles/embeddings](https://docs.continue.dev/advanced/model-roles/embeddings)

* [R4] Continue Docs, “Context Providers” / “Context Selection”
  [https://docs.continue.dev/guides/build-your-own-context-provider](https://docs.continue.dev/guides/build-your-own-context-provider)
  [https://docs.continue.dev/ide-extensions/autocomplete/context-selection](https://docs.continue.dev/ide-extensions/autocomplete/context-selection)

* [R5] Repomix Docs / Site
  [https://repomix.com/](https://repomix.com/)
  [https://repomix.com/guide/](https://repomix.com/guide/)

* [R6] Tree-sitter, “Introduction”
  [https://tree-sitter.github.io/tree-sitter/index.html](https://tree-sitter.github.io/tree-sitter/index.html)

* [R7] ast-grep, “Introduction”
  [https://ast-grep.github.io/](https://ast-grep.github.io/)
  [https://ast-grep.github.io/guide/introduction.html](https://ast-grep.github.io/guide/introduction.html)

* [R8] LM Studio Docs, “Local LLM API Server” / “OpenAI Compatibility Endpoints”
  [https://lmstudio.ai/docs/developer/core/server](https://lmstudio.ai/docs/developer/core/server)
  [https://lmstudio.ai/docs/developer/openai-compat](https://lmstudio.ai/docs/developer/openai-compat)

* [R9] Ollama Docs, “API Introduction” / “Authentication”
  [https://docs.ollama.com/api](https://docs.ollama.com/api)
  [https://docs.ollama.com/api/authentication](https://docs.ollama.com/api/authentication)

* [R10] GitHub Docs, “Authorizing OAuth apps”
  [https://docs.github.com/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps](https://docs.github.com/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps)

* [R11] GitHub Docs, “Differences between GitHub Apps and OAuth apps”
  [https://docs.github.com/ja/apps/oauth-apps/building-oauth-apps/differences-between-github-apps-and-oauth-apps](https://docs.github.com/ja/apps/oauth-apps/building-oauth-apps/differences-between-github-apps-and-oauth-apps)

* [R12] CodeSpectrasian Developer, “Bitbucket OAuth 2.0”
  [https://developer.CodeSpectrasian.com/cloud/bitbucket/oauth-2/](https://developer.CodeSpectrasian.com/cloud/bitbucket/oauth-2/)

* [R13] Git Docs, “git clone” / “partial clone” / “sparse-checkout”
  [https://git-scm.com/docs/git-clone.html](https://git-scm.com/docs/git-clone.html)
  [https://git-scm.com/docs/partial-clone/2.36.0.html](https://git-scm.com/docs/partial-clone/2.36.0.html)
  [https://git-scm.com/docs/git-sparse-checkout/2.36.0.html](https://git-scm.com/docs/git-sparse-checkout/2.36.0.html)

* [R14] Electron Docs, “Security”
  [https://www.electronjs.org/docs/latest/tutorial/security](https://www.electronjs.org/docs/latest/tutorial/security)

* [R15] Tauri Docs, “Security”
  [https://v2.tauri.app/security/](https://v2.tauri.app/security/)

* [R16] OpenAI Help Center, “Best Practices for API Key Safety”
  [https://help.openai.com/en/articles/5112595-best-practices-for-api](https://help.openai.com/en/articles/5112595-best-practices-for-api)

* [R17] Google AI for Developers, “Using Gemini API keys”
  [https://ai.google.dev/gemini-api/docs/api-key](https://ai.google.dev/gemini-api/docs/api-key)

* [R18] OpenAI API Reference, “Responses”
  [https://platform.openai.com/docs/api-reference/responses/retrieve](https://platform.openai.com/docs/api-reference/responses/retrieve)

* [R19] Anthropic Docs, “Messages API”
  [https://docs.anthropic.com/en/api/messages](https://docs.anthropic.com/en/api/messages)

* [R20] DeepSeek API Docs, “Your First API Call”
  [https://api-docs.deepseek.com/](https://api-docs.deepseek.com/)

* [R21] CodeSpectrasian Support, “Revoke an App password”
  [https://support.CodeSpectrasian.com/bitbucket-cloud/docs/revoke-an-app-password/](https://support.CodeSpectrasian.com/bitbucket-cloud/docs/revoke-an-app-password/)

* [R22] CodeSpectrasian Support, “Using API tokens”
  [https://support.CodeSpectrasian.com/bitbucket-cloud/docs/using-api-tokens/](https://support.CodeSpectrasian.com/bitbucket-cloud/docs/using-api-tokens/)
