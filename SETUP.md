# CodeSpectra — Setup Guide

Step-by-step guide to get CodeSpectra running from a fresh clone on **Windows**.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the repository](#2-clone-the-repository)
3. [Install Node.js dependencies](#3-install-nodejs-dependencies)
4. [Set up Python backend](#4-set-up-python-backend)
5. [Build native C++ module (optional)](#5-build-native-c-module-optional)
6. [Run the app](#6-run-the-app)
7. [Configure an LLM provider](#7-configure-an-llm-provider)
8. [Run tests](#8-run-tests)
9. [Build for production](#9-build-for-production)
10. [Data locations](#10-data-locations)
11. [Fresh reset](#11-fresh-reset)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites

Install these **before** doing anything else.

### Required

| Tool | Version | Install | Verify |
|------|---------|---------|--------|
| **Node.js** | >= 20 | https://nodejs.org | `node --version` |
| **Python** | >= 3.11 | https://python.org | `python --version` |
| **uv** | latest | `pip install uv` | `uv --version` |
| **Git** | any recent | https://git-scm.com | `git --version` |

### Optional (for local LLM inference)

| Tool | Purpose | Install |
|------|---------|---------|
| **Ollama** | Run local models (Llama, Mistral, etc.) | https://ollama.com |
| **LM Studio** | GUI for local models | https://lmstudio.ai |

These are only needed if you want **Strict Local** mode (no code leaves your machine). You can skip them and use a cloud provider (OpenAI, Anthropic, etc.) instead.

---

## 2. Clone the repository

```bash
git clone https://github.com/StevenTran0410/CodeSpectra.git
cd CodeSpectra
```

---

## 3. Install Node.js dependencies

```bash
npm install
```

This installs Electron, React, Vite, Tailwind, and all frontend dependencies.

---

## 4. Set up Python backend

### 4.1 Create virtual environment

```bash
cd backend
uv venv --python 3.11
```

This creates a `.venv` folder inside `backend/`.

### 4.2 Activate the virtual environment

**PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
```

**cmd:**
```cmd
.venv\Scripts\activate.bat
```

### 4.3 Install Python dependencies

```bash
uv pip install -e ".[dev]"
```

This installs:
- FastAPI, Uvicorn, aiosqlite, Pydantic, httpx
- Haystack AI (LLM orchestration)
- Tree-sitter + language grammars (Python, JS, TS, Go, Java, Rust, C, C++)
- pytest, ruff (dev tools)

### 4.4 Go back to project root

```bash
cd ..
```

---

## 5. Build native C++ module (optional)

> This step is **optional**. The app runs fine without it — all graph features (dependency scoring, neighbor expansion, Louvain clustering, cycle detection) have a pure-Python fallback. The native C++ module only provides a **performance boost** for large repositories.

### 5.1 Install Visual Studio Build Tools

1. Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2026
2. Install with these components:
   - Desktop development with C++
   - MSVC v143 build tools
   - Windows 10/11 SDK
   - C++ CMake tools for Windows

### 5.2 Build the module

Open **x64 Native Tools Command Prompt for VS** (not regular PowerShell), then:

```cmd
cd /d "path\to\CodeSpectra\backend"
call ".venv\Scripts\activate.bat"
uv pip install --python ".venv\Scripts\python.exe" setuptools
".venv\Scripts\python.exe" scripts\build_native_graph.py
```

### 5.3 Verify

```cmd
dir "backend\domain\structural_graph\_native_graph*.pyd"
```

You should see a `_native_graph*.pyd` file. If it exists, the native module is ready.

---

## 6. Run the app

### Full app (Electron + Python backend)

```bash
npm run dev
```

What happens:
1. Electron launches and spawns the Python backend automatically
2. Python backend finds a free port and starts FastAPI
3. Backend prints `BACKEND_READY:<port>` to signal readiness
4. App window opens — ready to use

### Backend only (for API testing / debugging)

```bash
npm run dev:backend
```

- Backend runs on `http://127.0.0.1:7868`
- Interactive API docs at `http://127.0.0.1:7868/docs`
- Useful for testing API endpoints without the Electron shell

---

## 7. Configure an LLM provider

CodeSpectra needs an LLM to generate analysis reports. Configure one in the app UI after launching.

### Option A: Local model (no API key needed)

1. Install and start [Ollama](https://ollama.com) or [LM Studio](https://lmstudio.ai)
2. Pull a model (e.g., `ollama pull llama3.1`)
3. In the app, go to Settings and select **Ollama** or **LM Studio** as provider
4. The app auto-detects models running on localhost

### Option B: Cloud provider (bring your own API key)

1. Get an API key from one of:
   - [OpenAI](https://platform.openai.com/api-keys)
   - [Anthropic](https://console.anthropic.com/)
   - [Google Gemini](https://aistudio.google.com/apikey)
   - [DeepSeek](https://platform.deepseek.com/)
2. In the app, go to Settings, select the provider, and paste your API key
3. The app will ask for **consent** on first use (code context is sent to the provider)

> API keys are stored in the local SQLite database only. They are never logged or sent anywhere other than the provider you configured.

---

## 8. Run tests

```bash
cd backend
pytest
```

Tests use `pytest-asyncio` with `asyncio_mode = "auto"` — all async tests run automatically.

To also run the linter:

```bash
ruff check .
ruff format --check .
```

---

## 9. Build for production

### Type-check

```bash
npm run typecheck
```

### Build production bundles

```bash
npm run build
```

### Package as installer

```bash
npm run package:win
```

Output goes to the `release/` directory as an `.exe` (NSIS installer).

---

## 10. Data locations

| Data | Path |
|------|------|
| **Database** | `%APPDATA%\CodeSpectra\codespectra.db` |
| **Logs** | `%APPDATA%\CodeSpectra\logs\` |
| **Cloned repos** | `%USERPROFILE%\CodeSpectra\repos\` |

---

## 11. Fresh reset

Close the app first, then run in PowerShell:

```powershell
Remove-Item "$env:APPDATA\CodeSpectra\codespectra.db" -Force -ErrorAction SilentlyContinue
Remove-Item "$env:APPDATA\CodeSpectra\logs" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:USERPROFILE\CodeSpectra\repos" -Recurse -Force -ErrorAction SilentlyContinue
```

> This resets all app state. It does **not** delete repositories you imported from local folders — only managed clones.

---

## 12. Troubleshooting

### App shows error dialog on startup

Python failed to start. Check:

```powershell
# Does the venv exist?
dir backend\.venv\

# Is Python 3.11+ available?
python --version

# Are dependencies installed?
cd backend; uv pip install -e ".[dev]"; cd ..
```

### Port conflict when running `dev:backend`

Change the port:

```bash
python backend/main.py --port 8000
```

### TypeScript errors after pulling changes

```bash
npm install
npm run typecheck
```

### Native module import fails

```bash
cd backend
python -c "import domain.structural_graph._native_graph as m; print('native ok')"
```

If this fails, rebuild the native module (see [Step 5](#5-build-native-c-module-optional)).

### Tree-sitter not found

If your venv was created before tree-sitter was added as a dependency:

```bash
cd backend
uv pip install -e .
```

---

## Quick reference

```bash
npm run dev              # Full app (Electron + backend)
npm run dev:backend      # Backend only on port 7868
npm run build            # Build production bundles
npm run typecheck        # TypeScript type-check
npm run package:win      # Package Windows installer
cd backend && pytest     # Run Python tests
```
