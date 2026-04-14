#!/usr/bin/env bash
set -euo pipefail

PORT=8787
HOST=127.0.0.1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─────────────────────────────────────────────────────────────────────────────
#  CodeSpectra MCP Server — macOS / Linux launcher
# ─────────────────────────────────────────────────────────────────────────────

clear
echo ""
echo "  +==============================================================+"
echo "  :                                                              :"
echo "  :            CodeSpectra - MCP Server                          :"
echo "  :                                                              :"
echo "  +==============================================================+"
echo ""

# Check venv
PYTHON="$SCRIPT_DIR/backend/.venv/bin/python3"
if [ ! -f "$PYTHON" ]; then
    echo "  [ERROR] Python venv not found at backend/.venv"
    echo "         Run these commands first:"
    echo ""
    echo "           cd backend"
    echo "           uv venv .venv"
    echo "           uv pip install -e '.[dev]'"
    echo ""
    exit 1
fi

echo "  +--------------------------------------------------------------+"
echo "  :                                                              :"
echo "  :   MCP endpoint:  http://$HOST:$PORT/mcp                     :"
echo "  :                                                              :"
echo "  :   -- Claude Code ------------------------------------------- :"
echo "  :                                                              :"
echo "  :   claude mcp add --transport http codespectra \\              :"
echo "  :     http://$HOST:$PORT/mcp                                   :"
echo "  :                                                              :"
echo "  :   -- Available tools -------------------------------------- :"
echo "  :                                                              :"
echo "  :   setup_project      Index a local project                   :"
echo "  :   retrieve_context   Semantic + BM25 code search             :"
echo "  :   ask_codebase       Q&A with multi-round retrieval          :"
echo "  :   deep_research      Multi-hop graph-aware research          :"
echo "  :                                                              :"
echo "  :   Press Ctrl+C to stop the server.                           :"
echo "  :                                                              :"
echo "  +--------------------------------------------------------------+"
echo ""

# Start backend using venv Python directly
cd "$SCRIPT_DIR/backend"
"$PYTHON" main.py --port "$PORT"
