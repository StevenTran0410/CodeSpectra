"""AEH Stage 4 eval route — thin wrapper that loads .aeh/run_eval.py by path (".aeh" isn't a
valid dotted package segment) and calls its driver directly, in-process. No LLM/network calls
of its own; the driver's own provider calls use this repo's own already-configured provider
service."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

_RUN_EVAL_PATH = Path(__file__).resolve().parent.parent / ".aeh" / "run_eval.py"


def _load_run_eval():
    spec = importlib.util.spec_from_file_location("aeh_run_eval", _RUN_EVAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {_RUN_EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@router.post("/run-eval")
async def run_eval(verify: bool = False) -> dict:
    try:
        run_eval_module = _load_run_eval()
        return await run_eval_module.run(verify=verify)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
