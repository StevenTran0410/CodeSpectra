# Repo CodeSpectra

> A local desktop app that reads a codebase and produces an evidence-backed onboarding report — powered by a fully local LLM or a bring-your-own-key cloud model.

**Author:** Steven Le Minh — steven0410leminh@gmail.com

---

## What it is

Repo CodeSpectra is built for one very specific engineering pain: walking into a large, unfamiliar codebase — or returning to one you once knew well and now barely remember.

It does not try to be an AI code editor or a chat assistant. It acts as a **codebase intelligence workbench**. You point it at a repository, it builds a structured understanding of the code, then produces a readable report that answers the questions every developer actually asks when onboarding:

- What does this system do, and how is it organized?
- Which files and symbols actually matter?
- Where should I start reading?
- What conventions does the team follow, and where are they broken?
- Where are the risky or complex parts?

The analysis runs entirely on your machine when you use a local model. No code leaves the device unless you explicitly configure a cloud provider and give consent.

---

## Architecture

```
┌──────────────────────────────────┐
│   Electron shell (Node.js)       │  Window management, IPC bridge,
│                                  │  native dialog, OS keychain access
└────────────┬─────────────────────┘
             │ HTTP (localhost)
┌────────────▼─────────────────────┐
│   Python backend (FastAPI)       │  All analysis logic, LLM routing,
│                                  │  file indexing, report generation
│   ├── domain/model_connector     │  Provider adapters (Ollama, LM Studio,
│   │       ├── ollama             │  OpenAI, Anthropic, Gemini, DeepSeek)
│   │       ├── lmstudio           │
│   │       ├── openai             │
│   │       ├── anthropic          │
│   │       └── gemini / deepseek  │
│   ├── domain/workspace           │  Workspace CRUD
│   ├── domain/local_repo          │  Local folder import + git metadata
│   └── infrastructure/db         │  SQLite via aiosqlite (WAL mode)
└──────────────────────────────────┘
             │ IPC
┌────────────▼─────────────────────┐
│   React renderer (Vite + TS)     │  Screens, Zustand stores, Tailwind UI
└──────────────────────────────────┘
```

The Electron main process is intentionally thin — it spawns the Python server, bridges IPC to HTTP, and handles anything that requires OS-level access (file dialogs, keychain). All domain logic lives in Python.

---

## Privacy modes

| Mode | What leaves the device |
|---|---|
| **Strict Local** (Ollama / LM Studio) | Nothing. All inference runs on your hardware. |
| **BYOK Cloud** (OpenAI / Anthropic / Gemini / DeepSeek) | Code context is sent to the provider you configured. Explicit consent required on first use. API keys stored in local DB, never logged. |

---

## Current state

### Done

| Ticket | Area | What was built |
|---|---|---|
| RPA-001 | Foundation | Product contract, privacy modes, domain model, state machine, open questions |
| RPA-002 | Foundation | Report schema, evidence model, confidence model, 3 sample JSON artifacts |
| RPA-010 | Platform | Electron + React + TypeScript desktop shell, workspace CRUD, IPC boundary, CSP |
| RPA-011 | Platform | SQLite storage schema, secure secrets design, settings service |
| RPA-012 | Platform | Job orchestration design, progress tracking, cancellation/resumability |
| RPA-020 | Providers | LLM provider abstraction layer, capability matrix, interface design |
| RPA-021 | Providers | Ollama adapter + LM Studio adapter, test connection, model listing, error mapping |
| RPA-022 | Providers | OpenAI, Anthropic, Gemini, DeepSeek adapters; BYOK consent flow; API key masking |
| RPA-025 | Code Hosts | Local folder import — native folder picker, git metadata reader, branch picker, size warnings |

### In progress / next

Ingestion pipeline (file manifest, symbol extraction via Tree-sitter), structural graph, analysis generation, and report viewer.

---

## Getting started

### Prerequisites

- Node.js 20+
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Ollama or LM Studio running locally (optional — required for Strict Local mode)

### Install and run

```bash
# Install Node dependencies
npm install

# Create Python virtual environment and install backend dependencies
cd backend
uv venv .venv
uv pip install -e ".[dev]"
cd ..

# Start the app (Electron + Python backend together)
npm run dev
```

The Python backend starts automatically when the Electron app launches. You can also run the backend standalone:

```bash
npm run dev:backend
```

See [`COMMANDS.md`](./COMMANDS.md) for the full reference including build and troubleshooting.

---

## Repository layout

```
├── backend/                  Python FastAPI backend
│   ├── api/                  Route handlers
│   ├── domain/               Business logic (model_connector, workspace, local_repo)
│   ├── infrastructure/db/    SQLite database + migrations
│   ├── shared/               Logger, error types
│   └── main.py               FastAPI app entry point
├── src/
│   ├── main/                 Electron main process
│   │   ├── api/              IPC handlers
│   │   └── infrastructure/   Python server manager, HTTP client
│   ├── preload/              Electron preload (IPC bridge)
│   └── renderer/             React app
│       ├── screens/          Page-level components
│       ├── store/            Zustand state stores
│       └── components/       Shared UI components
├── repo_atlas_plan/          Project planning (tickets, design docs, report samples)
└── COMMANDS.md               Start, build, troubleshoot
```

---

## License

Private — all rights reserved.
