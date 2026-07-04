# agent_eval_harness (AEH) — Agentic Evaluation Harness

CS-261 Phase 0: instrumentation adapters, trace capture, result store, and CLI
runner. An agentic system that evaluates OTHER agentic systems — see
`../repo_atlas_plan/tickets/CS-260_agentic_eval_harness_epic_architecture.md`
for the epic contract and `../repo_atlas_plan/aeh_prior_art_and_toolchain.md`
for the toolchain/prior-art research behind this design.

This package is fully independent of `backend/` — own `pyproject.toml`, own
`.venv`, own SQLite store (`aeh.db`). It talks to CodeSpectra's backend only
over REST (never a Python import across the folder boundary).

## Setup

```powershell
cd agent_eval_harness
python -m uv venv --python 3.11
python -m uv pip install --python ".venv\Scripts\python.exe" -e ".[dev]"
```

## Running the offline test suite

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

All 23 tests run fully offline via `FakeLLMClient` — no network calls, no
CodeSpectra backend required.

## CLI usage

```powershell
$env:AEH_DATA_DIR = "$PWD\.aeh_data"

.venv\Scripts\python.exe -m agent_eval_harness.cli run `
    --target "test_targets.multi_agent.pipeline:build_pipeline" `
    --map "test_targets\multi_agent\system_map.yaml" `
    --query "Can I get a refund and also change my shipping address?"
```

Toggle a planted defect (see `test_targets/multi_agent/README.md` for all 6):

```powershell
$env:AEH_DEFECT_PLANNER_OVERPACK = "1"
```

Tier-2 boundary-wrapper fallback demo (T1 only):

```powershell
.venv\Scripts\python.exe -m agent_eval_harness.cli run `
    --target "test_targets.linear_rag.pipeline:build_pipeline" `
    --map "test_targets\linear_rag\system_map.yaml" `
    --query "What is the vacation policy?" `
    --tier 2
```

## Live runs (real LLM, via CodeSpectra's configured provider)

AEH has no provider/API-key config of its own — it reuses whatever provider
you already configured in CodeSpectra, via a passthrough endpoint
(`backend/api/external.py`, mounted at `/api/external/*`).

**Manual smoke-test recipe** (not covered by the automated suite — requires a
running backend + a real provider):

1. Set a shared secret and start CodeSpectra's backend:
   ```powershell
   $env:CODESPECTRA_EXTERNAL_TOKEN = "some-long-random-token"
   cd ..\backend
   .venv\Scripts\python.exe main.py --port 7868
   ```
2. Confirm a provider is configured and discoverable:
   ```powershell
   curl.exe http://127.0.0.1:7868/api/external/llm/providers -H "Authorization: Bearer some-long-random-token"
   ```
3. Run AEH against it — either `.aeh/config.yaml` in this directory:
   ```yaml
   backend_url: "http://127.0.0.1:7868"
   backend_token: "some-long-random-token"
   provider_id: "<a provider_id from step 2>"
   ```
   or equivalent CLI flags: `--provider-id ... --backend-url ... --backend-token ...`.

## Layout

- `agent_eval_harness/` — the importable package (instrumentation adapters,
  mapping engine, store, LLM seam, CLI, runner, reporting).
- `test_targets/` — AEH's own unit-test fixtures (T1 `linear_rag`, T2
  `multi_agent`), never the thing being evaluated — see CS-260 §6.
- `tests/` — the offline test suite.
