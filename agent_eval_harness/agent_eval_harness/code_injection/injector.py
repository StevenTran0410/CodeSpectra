"""Orchestrates a Stage 4 injection into CodeSpectra's own backend/: prepares an isolated
worktree (never the caller's working directory), writes the 4 new artifacts, and applies the
2 marker-wrapped hooks main.py needs — tracer registration before any haystack import, and
the new eval route registered alongside the existing routers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agent_eval_harness.code_injection.file_writer import apply_marker_block, write_generated_file
from agent_eval_harness.code_injection.injection_target import WorktreeInjectionTarget
from agent_eval_harness.code_injection.wiring import build_wiring_for_codespectra
from agent_eval_harness.mapping.system_map import SystemMap

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

_TRACER_HOOK = (
    'import os\n'
    'os.environ.setdefault("HAYSTACK_CONTENT_TRACING_ENABLED", "true")\n'
)
_ROUTER_IMPORT_HOOK = "from api.aeh_eval import router as aeh_eval_router\n"
_ROUTER_INCLUDE_HOOK = '    app.include_router(aeh_eval_router, prefix="/aeh")\n'


@dataclass
class InjectionResult:
    worktree_path: Path
    branch_name: str
    files_written: list[str]


def inject_into_codespectra_backend(
    repo_root: Path, system_map: SystemMap, plan_id: str
) -> InjectionResult:
    target = WorktreeInjectionTarget(repo_root=repo_root, plan_id=plan_id)
    worktree = target.prepare()
    backend_root = worktree / "backend"

    files_written: list[str] = []

    write_generated_file(backend_root / ".aeh" / "tracer.py", (_TEMPLATES_DIR / "tracer.py").read_text(encoding="utf-8"))
    files_written.append("backend/.aeh/tracer.py")

    write_generated_file(backend_root / ".aeh" / "run_eval.py", (_TEMPLATES_DIR / "run_eval.py").read_text(encoding="utf-8"))
    files_written.append("backend/.aeh/run_eval.py")

    write_generated_file(backend_root / "api" / "aeh_eval.py", (_TEMPLATES_DIR / "aeh_eval.py").read_text(encoding="utf-8"))
    files_written.append("backend/api/aeh_eval.py")

    wiring = build_wiring_for_codespectra(system_map, plan_id)
    wiring_path = backend_root / ".aeh" / "wiring.json"
    wiring_path.parent.mkdir(parents=True, exist_ok=True)
    wiring_path.write_text(json.dumps(wiring, indent=2), encoding="utf-8")
    files_written.append("backend/.aeh/wiring.json")

    main_py = backend_root / "main.py"
    apply_marker_block(main_py, "aeh_tracer", _TRACER_HOOK, anchor="import argparse", where="before")
    apply_marker_block(
        main_py, "aeh_eval_import", _ROUTER_IMPORT_HOOK,
        anchor="from api.external import router as external_router", where="after",
    )
    apply_marker_block(
        main_py, "aeh_eval_route", _ROUTER_INCLUDE_HOOK,
        anchor='app.include_router(external_router, prefix="/api/external")', where="after",
    )
    files_written.append("backend/main.py (3 marker-wrapped hooks)")

    return InjectionResult(worktree_path=worktree, branch_name=target.branch_name, files_written=files_written)
