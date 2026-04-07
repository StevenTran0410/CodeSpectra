# CodeSpectra

CodeSpectra is a desktop app that analyzes a code repository and generates an evidence-backed onboarding report.

It is built for engineers joining unfamiliar codebases and teams that want fast technical context without handing full source code to third-party services by default.

## What CodeSpectra Does

- Imports a local folder or managed Git clone as a snapshot.
- Builds manifest, symbols, structural graph, and retrieval index.
- Runs a multi-agent analysis pipeline and outputs sections `A` to `K`.
- Streams section progress to the UI while analysis is running.
- Saves reports and supports export to Markdown.

## Analysis Sections

Each report is split into stable sections with fixed schemas:

- `A` Project Identity
- `B` Architecture Overview
- `C` Repository Structure
- `D` Coding Conventions
- `E` Forbidden Patterns / Negative Conventions
- `F` Feature Map
- `G` Important Files Radar
- `H` Onboarding Reading Order
- `I` Domain Glossary
- `J` Risk / Complexity
- `K` Evidence Auditor (meta-confidence across A-J)

## Runtime Architecture

```text
Electron (main + preload + renderer)
  -> local HTTP
Python FastAPI backend
  -> SQLite state
  -> retrieval + repo intelligence services
  -> LLM provider adapters (local + cloud)
  -> multi-agent orchestration
```

Key points:

- UI is Electron + React + TypeScript.
- Backend is Python (FastAPI) with optional native C++ acceleration for hotspots.
- Agent orchestration is Haystack AsyncPipeline with dependency-aware execution.
- Agents keep existing retrieval and model-provider integrations (custom services).

## Privacy Modes

- `Strict Local`: use local models only (for example Ollama / LM Studio).
- `BYOK Cloud`: use your own API key with cloud providers.

No cloud provider is used unless configured explicitly by the user.

## Current OSS State

What is production-ready in the repo today:

- Local repository setup and snapshot lifecycle.
- Manifest + ignore engine + language classification.
- Symbol extraction and structural graph generation.
- Retrieval pipeline with chunking and boundary-aware context expansion.
- A-K typed report sections with incremental section events.
- Report list/detail view and Markdown export.

## Quick Start

### Requirements

- Node.js 20+
- Python 3.11+
- `uv` package manager

### Install

```bash
npm install
cd backend
uv venv --python 3.11
uv pip install -e ".[dev]"
cd ..
```

### Run

```bash
npm run dev
```

The app starts Electron and Python backend together.

## Native Build Notes (Optional)

If you want native acceleration builds on Windows, install Visual Studio Build Tools with C++ workload and build the native module:

```bash
cd backend
python scripts/build_native_graph.py
```

If the native module is missing, affected hotspot paths may be slower.

## Project Layout

```text
backend/
  api/
  domain/
  infrastructure/
  shared/
src/
  main/
  preload/
  renderer/
repo_atlas_plan/
```

## Docs

- `COMMANDS.md` - run/build/troubleshooting commands
- `ANALYSIS_AGENTS.md` - current A-K pipeline and responsibilities
- `RAG_INDEXING_CHUNKING_NOTES.md` - retrieval/chunking behavior
- `KNOWN_LIMITATIONS.md` - current limitations and trade-offs

## License

Apache 2.0. See `LICENSE`.

