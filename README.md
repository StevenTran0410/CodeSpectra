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
│   Python/C++ backend (FastAPI)   │  All analysis logic, LLM routing,
│                                  │  file indexing, report generation
│   ├── domain/model_connector     │  Provider adapters (Ollama, LM Studio,
│   │       ├── ollama             │  OpenAI, Anthropic, Gemini, DeepSeek)
│   │       ├── lmstudio           │
│   │       ├── openai             │
│   │       ├── anthropic          │
│   │       └── gemini / deepseek  │
│   ├── domain/workspace           │  Workspace CRUD
│   ├── domain/local_repo          │  Local folder import + git metadata
│   └── infrastructure/db          │  SQLite via aiosqlite (WAL mode)
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

## Analysis pipeline (current)

Current report generation is LLM-driven and split into dedicated agents:

- **Section agents A-J**: each section has its own dedicated LLM agent and fixed output schema.
- **Agent K (Auditor)**: reviews A-J outputs and reports confidence, weakest sections, and coverage.
- **Orchestration**: Haystack AsyncPipeline with dependency-aware execution and per-section progress events.
- **Runtime behavior**: base sections run in parallel, dependent sections wait for prerequisites, auditor runs last.

Implementation location:
- `backend/domain/analysis/orchestrator.py`
- `backend/domain/analysis/agent_pipeline.py`
- `backend/domain/analysis/agents/`
- `backend/domain/analysis/prompts.py`

Notes:
- Agents run on the provider/model selected by the user at analysis start.
- Each section agent fetches retrieval context directly from the retrieval service (no central broker layer).
- Section completion is streamed incrementally to the UI (`analysis:section_done`).
- Report payload is versioned (`version: 2`) with section map under `sections`.

---

## Getting started

### Prerequisites

- Node.js 20+
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Ollama or LM Studio running locally (optional — required for Strict Local mode)

### Native C/C++ build prerequisites (for hotspot modules)

If you only run the app, the list above is enough.

If you want to build native Python extensions (C/C++ acceleration for indexing/graph hotspot), install this on Windows:

- Visual Studio Build Tools (official download):
  [https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2026](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2026)
- Workload: **Desktop development with C++**
- Components: **MSVC v143**, **Windows 10/11 SDK**, **C++ CMake tools for Windows**

After installation, open **Developer PowerShell for VS** and verify:

```powershell
cl
where.exe cl
```

If `cl` shows Microsoft C/C++ compiler banner, your native toolchain is ready.

Example detected path:

`C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\Hostx86\x86\cl.exe`

For production-native builds, prefer x64 toolchain shell (`Hostx64\x64`) instead of x86.

Graph hotspot features (RPA-033) require the native module. This project does not use a pure-Python fallback for that path.

### Install and run

```bash
# Install Node dependencies
npm install

# Create Python virtual environment and install backend dependencies
cd backend
uv venv .venv
uv pip install -e ".[dev]"

# Build native graph module (required for structural graph hotspot)
python scripts/build_native_graph.py
cd ..

# Start the app (Electron + Python backend together)
npm run dev
```

The Python backend starts automatically when the Electron app launches. You can also run the backend standalone:

```bash
npm run dev:backend
```

See [`COMMANDS.md`](./COMMANDS.md) for the full reference including build and troubleshooting.

Deep indexing dependencies (`tree-sitter` + language parser packages) are installed by default with backend dependencies.

Local app state is stored under Electron `userData` (Windows default: `%APPDATA%\CodeSpectra\codespectra.db`).
Managed cloned repositories are stored under `%USERPROFILE%\CodeSpectra\repos`.

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

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.