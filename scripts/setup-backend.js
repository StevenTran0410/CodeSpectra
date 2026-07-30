#!/usr/bin/env node
/**
 * Postinstall hook: sets up the Python backend venv alongside `npm install`.
 *
 * Idempotent (uv skips already-satisfied packages) and tolerant of failure —
 * never exits non-zero, so a frontend-only `npm install` never breaks because
 * of a missing/misconfigured Python toolchain. On any problem, it prints a
 * warning pointing to SETUP.md and lets npm install continue normally.
 *
 * Does NOT install the `reranker` extra (torch/transformers/bitsandbytes,
 * multi-GB GPU download) — that stays an explicit, separate opt-in via the
 * app's Settings UI, never bundled into routine `npm install`.
 */

const { spawnSync } = require("child_process");
const path = require("path");
const fs = require("fs");

const BACKEND_DIR = path.join(__dirname, "..", "backend");
const VENV_DIR = path.join(BACKEND_DIR, ".venv");

function run(cmd, args, cwd) {
  const result = spawnSync(cmd, args, { cwd, stdio: "inherit", shell: true });
  return result.status === 0;
}

function warnAndExit(message) {
  console.warn(`\n[setup-backend] ${message}`);
  console.warn("[setup-backend] Skipping backend setup — see SETUP.md section 4 to set it up manually.\n");
  process.exit(0);
}

function main() {
  if (!fs.existsSync(BACKEND_DIR)) {
    warnAndExit("backend/ directory not found, skipping.");
    return;
  }

  if (!run("uv", ["--version"], BACKEND_DIR)) {
    warnAndExit("`uv` not found on PATH (required: `pip install uv`). Backend venv was not set up automatically.");
    return;
  }

  if (!fs.existsSync(VENV_DIR)) {
    console.log("[setup-backend] No backend/.venv found — creating it (uv venv --python 3.11)...");
    if (!run("uv", ["venv", "--python", "3.11"], BACKEND_DIR)) {
      warnAndExit("Failed to create backend/.venv. Backend setup incomplete.");
      return;
    }
  }

  console.log("[setup-backend] Installing backend dependencies (uv pip install -e \".[dev,analysis]\")...");
  console.log("[setup-backend] This is fast if already installed (uv skips satisfied packages).");
  if (!run("uv", ["pip", "install", "-e", ".[dev,analysis]"], BACKEND_DIR)) {
    warnAndExit("Failed to install backend dependencies. Backend setup incomplete.");
    return;
  }

  console.log("[setup-backend] Backend ready.");
  console.log(
    "[setup-backend] Note: the GPU reranker extra (torch/bitsandbytes, multi-GB) is NOT installed here — " +
      "enable it from the app's Settings panel when you have a compatible GPU."
  );
}

main();
