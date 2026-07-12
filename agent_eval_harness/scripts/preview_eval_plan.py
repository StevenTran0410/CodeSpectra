"""Preview AEH_EVAL_PLAN.md's real content for a session, without touching the target repo
or requiring an eval branch — calls the exact same functions the eval-plan route
(ui/server.py::create_eval_plan) uses, just skips the git/branch requirements and doesn't
write dataset .jsonl files into the target repo.

IMPORTANT: the live app's aeh.db/maps/ live under Electron's userData path
(app.getPath('userData'), set as AEH_DATA_DIR by src/main/infrastructure/aeh-server/server.ts),
NOT under agent_eval_harness/ — that's just wherever a bare `uv run` happens to default to.
On Windows that's normally %APPDATA%\\CodeSpectra. Point AEH_DATA_DIR there or you'll be
reading a stale/empty local db instead of the real session data.

Usage (run from agent_eval_harness/, same cwd the AEH server itself uses):
  # PowerShell:
  $env:AEH_DATA_DIR = "$env:APPDATA\CodeSpectra"; uv run python scripts/preview_eval_plan.py [session_id]
  # bash:
  AEH_DATA_DIR="$APPDATA/CodeSpectra" uv run python scripts/preview_eval_plan.py [session_id]

If session_id is omitted, uses the most recently created session that has both a
map_path and a plan_report_path set.

Output: writes AEH_EVAL_PLAN.preview.md next to this script and prints its path.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import aiosqlite

from agent_eval_harness.code_injection.plan_renderer import render_eval_plan_md
from agent_eval_harness.code_injection.wiring import build_wiring_for_codespectra
from agent_eval_harness.config import AEHConfig
from agent_eval_harness.datasets.fulfillment import export_dataset
from agent_eval_harness.mapping.system_map import load_system_map
from agent_eval_harness.metrics.suite import load_suite
from agent_eval_harness.planning.report import load_plan_report
from agent_eval_harness.store import repository
from agent_eval_harness.store.database import close_db, init_db


async def _find_latest_session_id() -> tuple[str | None, bool]:
    """Returns (session_id, has_plan_report). Prefers a session with a plan_report_path;
    falls back to any session with at least map_path + plan_path, flagging the report as
    missing so the caller can render the thin legacy view instead of erroring out."""
    # Same AEH_DATA_DIR / "aeh.db" resolution init_db() uses (store/database.py:308) —
    # this is a read-only lookup query, not a second connection manager.
    db_path = Path(os.getenv("AEH_DATA_DIR", ".")) / "aeh.db"
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id FROM expansion_sessions "
            "WHERE map_path IS NOT NULL AND plan_report_path IS NOT NULL "
            "ORDER BY created_at DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        if row:
            return row["id"], True
        cursor = await db.execute(
            "SELECT id FROM expansion_sessions "
            "WHERE map_path IS NOT NULL AND plan_path IS NOT NULL "
            "ORDER BY created_at DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return (row["id"], False) if row else (None, False)


def _get_repo_root() -> Path:
    config = AEHConfig.load()
    if not config.backend_source_path:
        raise SystemExit("backend_source_path not configured — set it in .aeh/config.yaml")
    backend_path = Path(config.backend_source_path)
    if not backend_path.is_dir():
        raise SystemExit(f"backend_source_path does not exist: {backend_path}")
    return backend_path.parent


async def main() -> None:
    await init_db()
    try:
        if len(sys.argv) > 1:
            session_id, has_report = sys.argv[1], True  # trust an explicit id; checked below
        else:
            session_id, has_report = await _find_latest_session_id()
        if not session_id:
            raise SystemExit("No session found with both map_path and plan_path set.")

        sess = await repository.get_expansion_session(session_id)
        if not sess:
            raise SystemExit(f"Session {session_id!r} not found.")
        if not sess.get("map_path"):
            raise SystemExit(f"Session {session_id!r} has no map_path — Stage 2 not complete.")
        if not sess.get("plan_path"):
            raise SystemExit(f"Session {session_id!r} has no plan_path — generate a plan first.")
        has_report = has_report and bool(sess.get("plan_report_path"))

        print(f"Using session: {session_id}")
        repo_root = _get_repo_root()
        system_map = load_system_map(sess["map_path"])
        suite = load_suite(sess["plan_path"])
        if has_report:
            plan_report = load_plan_report(sess["plan_report_path"])
            print(f"Agents in plan_report: {len(plan_report.agents)}")
        else:
            plan_report = None
            print(
                "WARNING: this session has no plan_report_path (predates it, or Stage 3's "
                "plan was never regenerated with a saved report). Rendering the thin legacy "
                "view — regenerate the plan in Stage 3 to see the rich per-agent sections."
            )

        datasets_to_gate_ids: dict[str, list[str]] = {}
        for entry in suite.entries:
            if entry.dataset and entry.dataset.ref:
                datasets_to_gate_ids.setdefault(entry.dataset.ref, []).append(entry.id)

        dataset_summaries: list[dict] = []
        for dataset_id, gate_ids in datasets_to_gate_ids.items():
            cases = await export_dataset(dataset_id)
            dataset_summaries.append({
                "dataset_id": dataset_id,
                "kind": cases[0].kind if cases else "",
                "case_count": len(cases),
                "gate_ids": gate_ids,
                "example_case": cases[0].model_dump() if cases else None,
            })

        wiring = build_wiring_for_codespectra(system_map, session_id)
        branch_name = sess.get("eval_branch_name") or f"aeh/eval-{session_id}"
        md_content = render_eval_plan_md(
            system_map, wiring, dataset_summaries, session_id, branch_name,
            plan_report=plan_report, repo_root=repo_root,
        )

        out_path = Path(__file__).resolve().parent / "AEH_EVAL_PLAN.preview.md"
        out_path.write_text(md_content, encoding="utf-8")
        print(f"Wrote preview to: {out_path}")
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
