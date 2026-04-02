# CodeSpectra — Commands Reference

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Node.js | ≥ 20 | https://nodejs.org |
| Python | ≥ 3.11 | https://python.org |
| uv | latest | `pip install uv` |

---

## First-time setup

```bash
# 1. Install Node.js dependencies
npm install

# 2. Create Python virtual environment and install backend dependencies
cd backend
uv venv --python 3.11
uv pip install -e ".[dev]"
cd ..
```

---

## Development

### Start the full app (Electron + Python backend together)

```bash
npm run dev
```

Electron spawns the Python backend automatically.
The app window will not open until the backend is ready.

### Start Python backend standalone (for API testing)

```bash
npm run dev:backend
```

Backend runs on `http://127.0.0.1:7868`.
Interactive API docs available at `http://127.0.0.1:7868/docs`.

---

## Build

```bash
# Type-check both main process and renderer
npm run typecheck

# Build production bundles (output → out/)
npm run build

# Preview production build
npm run preview
```

---

## Project structure

```
CodeSpectra/
├── backend/                  # Python analysis engine (FastAPI)
│   ├── main.py               # Entry point — spawned by Electron
│   ├── api/                  # HTTP route handlers
│   ├── domain/               # Business logic
│   │   ├── workspace/        # Workspace CRUD
│   │   └── model_connector/  # LLM provider adapters
│   │       ├── ollama/
│   │       └── lmstudio/
│   ├── infrastructure/
│   │   └── db/               # SQLite (aiosqlite) + migrations
│   ├── shared/               # Logger, errors
│   └── .venv/                # Python virtual environment (git-ignored)
│
└── src/
    ├── main/                 # Electron main process (thin shell)
    │   ├── api/              # IPC handlers — proxy calls to Python HTTP
    │   ├── infrastructure/
    │   │   └── python-server/ # Spawns + manages the Python process
    │   └── shared/           # Logger
    ├── preload/              # Context bridge (IPC ↔ renderer)
    └── renderer/             # React + TypeScript UI
```

---

## Troubleshooting

### App shows error dialog on startup
Python failed to start. Common causes:
- `backend/.venv` does not exist → run `cd backend && uv venv --python 3.11 && uv pip install -e ".[dev]"`
- Python 3.11 not on PATH → check `python --version`

### Port conflict when running `dev:backend`
Change the port: `python backend/main.py --port 8000`

### TypeScript errors after pulling changes
```bash
npm install
npx tsc --noEmit -p tsconfig.node.json
npx tsc --noEmit -p tsconfig.web.json
```
