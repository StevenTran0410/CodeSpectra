# CodeSpectra — Commands Reference

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Node.js | ≥ 20 | https://nodejs.org |
| Python | ≥ 3.11 | https://python.org |
| uv | latest | `pip install uv` |

---

## Native C/C++ toolchain (Windows, for native hotspot build)

Only required when building native Python extension modules (C/C++).
Not required for basic app run.

Install **Visual Studio Build Tools** from:

[https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2026](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2026)

Then select:

- Desktop development with C++
- MSVC v143 build tools
- Windows 10/11 SDK
- C++ CMake tools for Windows

Verify in **Developer PowerShell for VS**:

```powershell
cl
where.exe cl
Get-Command cl
```

Expected: Microsoft compiler banner + valid `cl.exe` path.
Example valid path:
`C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\Hostx86\x86\cl.exe`

Important: this path is x86 toolchain. For this project, prefer x64 shell/toolchain when building release-native modules.

If you are in Git Bash (MINGW64), do not assume it is usable for Python native builds.
Check explicitly:

```bash
gcc --version
g++ --version
which gcc
which g++
```

For this project, prefer **MSVC** on Windows to avoid Python ABI/linking mismatch issues.

---

## Full Native Build Flow (copy/paste)

Starting point:

`PS D:\Program Files\Python\CodeSpectra>`

### 1) Open x64 VS toolchain shell from current PowerShell

```powershell
cmd.exe /k "`"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=amd64 -host_arch=amd64"
```

You are now in a new `cmd` window with VS env loaded.

### 2) Verify x64 compiler in that cmd window

```cmd
cl
where.exe cl
```

Expected:
- `Microsoft (R) C/C++ Optimizing Compiler ... for x64`
- `...Hostx64\x64\cl.exe`

### 3) Build backend native graph module

```cmd
cd /d "D:\Program Files\Python\CodeSpectra\backend"

if not exist ".venv\Scripts\python.exe" uv venv --python 3.11
call ".venv\Scripts\activate.bat"

rem IMPORTANT: install deps into .venv, not global Python
uv pip install --python ".venv\Scripts\python.exe" -e ".[dev]"
uv pip install --python ".venv\Scripts\python.exe" setuptools
".venv\Scripts\python.exe" scripts\build_native_graph.py
```

If you accidentally run `pip install setuptools` and it says global path
like `C:\Users\...\Python311\...`, you installed to the wrong interpreter.
Use the exact commands above with `.venv\Scripts\python.exe`.

### 4) Verify native artifact exists

```cmd
dir "D:\Program Files\Python\CodeSpectra\backend\domain\structural_graph\_native_graph*.pyd"
```

### 5) Run app

```cmd
cd /d "D:\Program Files\Python\CodeSpectra"
npm run dev
```

---

## Rebuild Native Module (clean + build)

Run in x64 VS cmd:

```cmd
cd /d "D:\Program Files\Python\CodeSpectra\backend"
del /q "domain\structural_graph\_native_graph*.pyd"
rmdir /s /q build
call ".venv\Scripts\activate.bat"
uv pip install --python ".venv\Scripts\python.exe" setuptools
".venv\Scripts\python.exe" scripts\build_native_graph.py
```

---

## Quick Failure Checks

In x64 VS cmd:

```cmd
cl
where.exe cl
".venv\Scripts\python.exe" -c "import platform,struct; print(platform.machine(), struct.calcsize('P')*8)"
".venv\Scripts\python.exe" -c "import domain.structural_graph._native_graph as m; print('native ok')"
```

If last import fails, native build is not complete.

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

### Repository sources

Any git repository (GitHub, Bitbucket, GitLab, self-hosted, etc.) can be added via:
- **Open Local Folder** — point to an already-cloned directory on disk.
- **Clone from URL** — paste any HTTPS or SSH git URL; the app runs `git clone` using your system git credential manager or a configured SSH key.

No OAuth tokens or platform-specific setup required.

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

### Tree-sitter / deep index dependency

`tree-sitter` and `tree-sitter-languages` are now part of backend default dependencies.
If your existing virtual environment was created before this change, run:

```bash
cd backend
uv pip install -e .
cd ..
```

For packaged builds (EXE), these dependencies are bundled with the backend environment/binary when you build from an up-to-date environment.

### Native graph module build (optional, for performance)

Run this in x64 Developer PowerShell/Command Prompt:

```bash
cd backend
uv pip install -e ".[dev]"
python scripts/build_native_graph.py
cd ..
```

Expected artifact (Windows):
`backend/domain/structural_graph/_native_graph*.pyd`

If this module is missing, all graph features (scoring, neighbor expansion, Louvain clustering, cycle detection) fall back to pure-Python automatically. The native module provides a performance boost for large repositories.

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

## Data Location + Fresh Reset

CodeSpectra stores local app data in Electron `userData`:

- Windows (default): `%APPDATA%\CodeSpectra\`
- DB file: `%APPDATA%\CodeSpectra\codespectra.db`

Managed cloned repositories are stored at:

- `%USERPROFILE%\CodeSpectra\repos\`

### Fresh reset (Windows PowerShell)

Close the app first, then run:

```powershell
Remove-Item "$env:APPDATA\CodeSpectra\codespectra.db" -Force -ErrorAction SilentlyContinue
Remove-Item "$env:APPDATA\CodeSpectra\logs" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:USERPROFILE\CodeSpectra\repos" -Recurse -Force -ErrorAction SilentlyContinue
```

This resets local DB state and removes managed clone copies.
It does not delete repositories you imported directly from arbitrary folders.

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
