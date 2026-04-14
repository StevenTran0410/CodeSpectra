@echo off
setlocal enabledelayedexpansion

set "PORT=8787"
set "HOST=127.0.0.1"
set "PYTHON=%~dp0backend\.venv\Scripts\python.exe"

:: ─────────────────────────────────────────────────────────────────────────────
::  CodeSpectra MCP Server - Windows launcher
:: ─────────────────────────────────────────────────────────────────────────────

cls
echo.
echo   +==============================================================+
echo   :                                                              :
echo   :            CodeSpectra - MCP Server                          :
echo   :                                                              :
echo   +==============================================================+
echo.

:: Check venv Python
if not exist "%PYTHON%" (
    echo   [ERROR] Python venv not found at:
    echo          %PYTHON%
    echo.
    echo   Run these commands first:
    echo.
    echo     cd backend
    echo     uv venv .venv
    echo     uv pip install -e ".[dev]"
    echo.
    pause
    exit /b 1
)

echo   +--------------------------------------------------------------+
echo   :                                                              :
echo   :   MCP endpoint:  http://%HOST%:%PORT%/mcp                    :
echo   :                                                              :
echo   :   -- Claude Code ------------------------------------------- :
echo   :                                                              :
echo   :   claude mcp add --transport http codespectra                :
echo   :     http://%HOST%:%PORT%/mcp                                 :
echo   :                                                              :
echo   :   Or add to .mcp.json:                                       :
echo   :                                                              :
echo   :   {                                                          :
echo   :     "mcpServers": {                                          :
echo   :       "codespectra": {                                       :
echo   :         "url": "http://%HOST%:%PORT%/mcp"                    :
echo   :       }                                                      :
echo   :     }                                                        :
echo   :   }                                                          :
echo   :                                                              :
echo   :   -- Available tools -------------------------------------- :
echo   :                                                              :
echo   :   setup_project      Index a local project                   :
echo   :   retrieve_context   Semantic + BM25 code search             :
echo   :   ask_codebase       Q+A with multi-round retrieval          :
echo   :   deep_research      Multi-hop graph-aware research          :
echo   :                                                              :
echo   :   Press Ctrl+C to stop the server.                           :
echo   :                                                              :
echo   +--------------------------------------------------------------+
echo.

:: Start backend using venv Python directly
cd /d "%~dp0backend"
"%PYTHON%" main.py --port %PORT%

:: If server exits (error or Ctrl+C), pause so user can read output
echo.
echo   Server stopped.
pause
