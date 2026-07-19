"""Standalone OFFLINE Stage-3 runner + dataset dumper for AEH.

Runs the REAL Stage 3 (plan generation + dataset fulfillment) for a few randomly-chosen
already-extracted agents from a persisted Stage-1/2 session, then dumps the generated
datasets (input + gold + labels) to JSON so you can eyeball Stage-3 output.

Design (no Electron GUI, no backend HTTP server):
  * LLM  -> the FSOFT provider row already in codespectra.db (gpt-5.4-mini, reasoning=medium),
           driven directly through the backend's ProviderConfigService. The API key stays in
           the DB; this script never reads/prints it.
  * repo -> snapshot.local_path read straight from codespectra.db (no get_snapshot REST call).
  * embeddings -> skipped (qa_testset degrades to needs_human; synthetic_agent_io skips dedup).
  * Stage 1/2 artifacts -> loaded from aeh.db + on-disk maps/sidecars.

Run from the repo root, inside the agent_eval_harness uv env:
    uv run python agent_eval_harness/scripts/run_stage3_smoke.py --list
    uv run python agent_eval_harness/scripts/run_stage3_smoke.py [--session ID] [--k 4]
        [--agents id1,id2,...] [--seed 0] [--all-kinds] [--smoke-only]

NOTE: fulfillment writes NEW (versioned, non-destructive) dataset rows into the real aeh.db.
To keep the real DB untouched, copy aeh.db elsewhere and set AEH_DATA_DIR to that folder.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sqlite3
import sys
from pathlib import Path

# --- paths + env (must be set before AEH/backend DB modules read them) ---
_REPO = Path(__file__).resolve().parents[2]  # scripts -> agent_eval_harness -> <repo>
_APPDATA = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
_DATA_DIR = os.path.join(_APPDATA, "CodeSpectra")
os.environ.setdefault("AEH_DATA_DIR", _DATA_DIR)          # aeh.db
os.environ.setdefault("CODESPECTRA_DATA_DIR", _DATA_DIR)  # codespectra.db (provider row lives here)
_CSDB = os.path.join(_DATA_DIR, "codespectra.db")

# backend/ must be importable for the direct LLM path; insert explicitly (do NOT trust .aeh config).
for p in (str(_REPO / "backend"), str(_REPO / "agent_eval_harness")):
    if p not in sys.path:
        sys.path.insert(0, p)

# --- FSOFT provider defaults (config ids, NOT secrets) ---
FSOFT_PROVIDER_ID = "a4b16186-c964-4338-b96c-53af776b5fdc"
FSOFT_MODEL_ID = "GPT-5.4-mini"
DEFAULT_EFFORT = "medium"


def _resolve_fsoft_provider(explicit_id: str | None) -> tuple[str, str]:
    """Pick the FSOFT row specifically (NOT any gpt-5 provider — OpenAI's gpt-5-chat-latest
    must not win). Priority: explicit id > 'fsoft' in name > known FSOFT id > exact model id."""
    con = sqlite3.connect(_CSDB)
    con.row_factory = sqlite3.Row
    try:
        rows = list(con.execute("SELECT id, model_id, display_name FROM provider_configs"))
    finally:
        con.close()
    by_id = {r["id"]: r for r in rows}
    if explicit_id and explicit_id in by_id:
        r = by_id[explicit_id]
        return r["id"], r["model_id"] or FSOFT_MODEL_ID
    for r in rows:  # display_name = "API-Fsoft"
        if "fsoft" in (r["display_name"] or "").lower():
            return r["id"], r["model_id"] or FSOFT_MODEL_ID
    if FSOFT_PROVIDER_ID in by_id:
        r = by_id[FSOFT_PROVIDER_ID]
        return r["id"], r["model_id"] or FSOFT_MODEL_ID
    for r in rows:
        if (r["model_id"] or "").lower() == FSOFT_MODEL_ID.lower():
            return r["id"], r["model_id"]
    print("[warn] no FSOFT provider row matched; using known id constant")
    return FSOFT_PROVIDER_ID, FSOFT_MODEL_ID


class FsoftLLMClient:
    """Satisfies the AEH LLMClient Protocol by delegating to the backend ProviderConfigService,
    which loads the FSOFT key from codespectra.db internally and auto-retries dropping
    reasoning_effort/temperature if the gateway rejects them. Never touches the raw key."""

    def __init__(self, provider_id: str, model_id: str, default_effort: str = DEFAULT_EFFORT) -> None:
        from domain.model_connector.service import ProviderConfigService
        self._svc = ProviderConfigService()
        self._pid, self._mid, self._effort = provider_id, model_id, default_effort

    async def complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False, reasoning_effort=None):
        from agent_eval_harness.llm.client import LLMResponse
        from domain.model_connector.types import ChatMessage, ChatRequest
        effort = reasoning_effort if reasoning_effort is not None else self._effort  # per-call wins
        resp = await self._svc.chat(ChatRequest(
            provider_id=self._pid, model_id=self._mid,
            messages=[ChatMessage(role=m.role, content=m.content) for m in messages],
            max_completion_tokens=max_tokens, temperature=temperature,
            reasoning_effort=effort, json_mode=json_mode, stream=False,
        ))
        return LLMResponse(
            content=resp.content, model=resp.model_id,
            prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
            token_source="measured" if resp.prompt_tokens is not None else "estimated",
        )

    async def aclose(self) -> None:  # service opens/closes its own adapter per call
        pass


def _snapshot_local_path(snapshot_id: str) -> Path:
    """Resolve the target repo root from codespectra.db (replaces client.get_snapshot)."""
    con = sqlite3.connect(_CSDB)
    try:
        row = con.execute(
            "SELECT local_path FROM repo_snapshots WHERE id=?", (snapshot_id,)
        ).fetchone()
    finally:
        con.close()
    override = os.environ.get("AEH_REPO_ROOT")
    if override:
        return Path(override)
    if not row or not row[0]:
        raise SystemExit(f"snapshot {snapshot_id} has no local_path in codespectra.db")
    return Path(row[0])


async def _pick_ready_session(repository, session_id: str | None) -> dict:
    """Newest completed expansion session that has map + agent_flows + >=1 agent_knowledge row."""
    db = repository.get_db()
    async with db.execute(
        "SELECT id FROM expansion_sessions "
        "WHERE status='completed' AND map_path IS NOT NULL AND agent_flows_path IS NOT NULL "
        "ORDER BY created_at DESC"
    ) as cur:
        rows = await cur.fetchall()
    candidates = [session_id] if session_id else [r[0] for r in rows]
    for sid in candidates:
        sess = await repository.get_expansion_session(sid)
        if not sess or not sess.get("map_path") or not sess.get("agent_flows_path"):
            continue
        if not Path(sess["agent_flows_path"]).exists() or not Path(sess["map_path"]).exists():
            continue
        if not await repository.list_agent_knowledge(sid):
            continue
        return sess
    raise SystemExit("No Stage-3-ready expansion session found (need map + agent_flows + agent_knowledge).")


async def _list_sessions(repository) -> None:
    db = repository.get_db()
    async with db.execute(
        "SELECT id, created_at FROM expansion_sessions "
        "WHERE status='completed' AND map_path IS NOT NULL AND agent_flows_path IS NOT NULL "
        "ORDER BY created_at DESC"
    ) as cur:
        rows = await cur.fetchall()
    print(f"{'session_id':<40} {'created_at':<22} agents  knowledge")
    for sid, created in rows:
        sess = await repository.get_expansion_session(sid)
        try:
            from agent_eval_harness.mapping.agent_flow import load_agent_flow_map
            n_agents = len(load_agent_flow_map(sess["agent_flows_path"]).agents)
        except Exception:
            n_agents = "?"
        n_know = len(await repository.list_agent_knowledge(sid))
        ready = "READY" if n_know else "no-knowledge"
        print(f"{sid:<40} {str(created):<22} {str(n_agents):>6}  {n_know:>4}  {ready}")


def _write_plan_files(suite, report, scratch: Path, map_src: str):
    """Write a scratch map copy + plan.yaml + <mapstem>_plan_report.yaml (the name fulfill_plan derives)."""
    import yaml
    scratch.mkdir(parents=True, exist_ok=True)
    map_scratch = scratch / "map.yaml"
    map_scratch.write_text(Path(map_src).read_text("utf-8"), "utf-8")
    plan_path = scratch / "plan.yaml"
    from agent_eval_harness.datasets.fulfillment import _save_suite
    _save_suite(suite, plan_path)  # canonical writer fulfill_plan reads back
    report_path = map_scratch.with_name("map_plan_report.yaml")
    try:
        from agent_eval_harness.planning.report import save_plan_report
        save_plan_report(report, report_path)
    except Exception:
        report_path.write_text(yaml.safe_dump(report.model_dump(mode="json"), sort_keys=False, allow_unicode=True), "utf-8")
    return plan_path, map_scratch


async def _dump_datasets(repository, fulfil_result: dict, out_path: Path) -> None:
    """Read back every fulfilled dataset with get_dataset_cases (NOT export_dataset — it drops synthetic)."""
    dump = {"generated_at_note": "synthetic (pre-review) Stage-3 output", "datasets": []}
    for group_key, info in sorted(fulfil_result.items()):
        did = info.get("dataset_id")
        entry = {"group": group_key, "status": info.get("status"), "reason": info.get("reason"), "dataset_id": did, "cases": []}
        if did:
            meta = await repository.get_dataset_metadata(did)
            entry["kind"] = meta.get("kind") if meta else None
            for c in await repository.get_dataset_cases(did):
                inp = json.loads(c["input_json"]) if c.get("input_json") else {}
                labels = json.loads(c["labels_json"]) if c.get("labels_json") else {}
                gold = json.loads(c["expected_json"]) if c.get("expected_json") else None
                entry["cases"].append({
                    "id": c["id"], "provenance": c["provenance"],
                    "agent_id": labels.get("agent_id"), "archetype": labels.get("archetype"),
                    "shape": inp.get("shape"),
                    "input": {k: v for k, v in inp.items() if k != "kind"},
                    "gold": gold, "labels": labels,
                })
        dump["datasets"].append(entry)
    out_path.write_text(json.dumps(dump, indent=2, ensure_ascii=False), "utf-8")

    print("\n=== STAGE 3 OUTPUT SUMMARY ===")
    for d in dump["datasets"]:
        n = len(d["cases"])
        print(f"  {d['group']:<45} {d['status']:<12} {d.get('kind') or '':<20} {n} case(s)"
              + (f"  ({d['reason']})" if d.get("reason") else ""))
        if d["cases"]:
            s = d["cases"][0]
            print(f"      e.g. shape={s['shape']!r} input_fields={sorted(k for k in s['input'])}"
                  f" gold_keys={sorted((s['gold'] or {}).keys()) if isinstance(s['gold'], dict) else s['gold']}")
    print(f"\nFull dump written to: {out_path}")


async def run(args) -> None:
    from agent_eval_harness.store.database import init_db as aeh_init_db, close_db as aeh_close_db
    from agent_eval_harness.store import repository

    # backend DB (provider row) — needed for the direct LLM path.
    from infrastructure.db.database import init_db as backend_init_db, close_db as backend_close_db
    await backend_init_db()
    await aeh_init_db()
    llm = None
    try:
        if args.list:
            await _list_sessions(repository)
            return

        provider_id, model_id = _resolve_fsoft_provider(args.provider_id)
        llm = FsoftLLMClient(provider_id, model_id, default_effort=args.effort)

        # smoke-test the LLM before any expensive Stage-3 work.
        from agent_eval_harness.llm.client import LLMMessage
        r = await llm.complete([LLMMessage(role="user", content="Reply with exactly: OK")], max_tokens=512)
        print(f"[llm smoke] {model_id} (effort={args.effort}) -> {r.content!r}")
        if args.smoke_only:
            return

        sess = await _pick_ready_session(repository, args.session)
        session_id = sess["id"]
        print(f"[session] {session_id}  map={sess['map_path']}")

        from agent_eval_harness.mapping.system_map import load_system_map
        from agent_eval_harness.mapping.agent_flow import load_agent_flow_map, build_source_by_component, AgentFlowMap
        system_map = load_system_map(sess["map_path"])
        full_flow = load_agent_flow_map(sess["agent_flows_path"])

        # choose agents
        if args.agents:
            want = {a.strip() for a in args.agents.split(",") if a.strip()}
            chosen = [a for a in full_flow.agents if a.id in want]
            if not chosen:
                raise SystemExit(f"none of --agents {sorted(want)} found; available: {[a.id for a in full_flow.agents]}")
        else:
            rng = random.Random(args.seed)
            chosen = rng.sample(full_flow.agents, min(args.k, len(full_flow.agents)))
        chosen_ids = [a.id for a in chosen]
        print(f"[agents] {chosen_ids}")
        sub_flow = AgentFlowMap(
            target_system_id=full_flow.target_system_id, agents=chosen,
            entry_agent_ids=[a.id for a in chosen if a.id in full_flow.entry_agent_ids],
            unassigned_component_ids=[],
        )

        local_path = _snapshot_local_path(sess["snapshot_id"])
        if not local_path.exists():
            raise SystemExit(f"repo root does not exist on disk: {local_path}")
        abs_files = [local_path / (p["file"] if isinstance(p, dict) else p) for p in sess["accepted"]]
        harvest_files = abs_files + [local_path / p for p in sess.get("boundary", [])]
        source_by_component = build_source_by_component(abs_files, system_map)

        conventions = None
        try:
            from agent_eval_harness.config import AEHConfig
            conventions = AEHConfig.load().contract_conventions
        except Exception:
            pass

        # --- STAGE 3a: plan generation (in-memory, no writes) ---
        from agent_eval_harness.planning.agentic_planner import generate_plan_agentic
        print("[stage 3a] generating plan (analyst + gate design + reconcile + critic)...")
        suite, report = await generate_plan_agentic(
            system_map, sub_flow, source_by_component, sess["accepted_edges"], llm,
            files=harvest_files, files_root=local_path,
            project_context=None, conventions=conventions,
        )
        for ar in report.agents:
            if ar.agent_id in chosen_ids:
                ar.eval_enabled = True  # else synthetic_agent_io groups are skipped in fulfillment

        # HARD CAP the case count (default 20 -> args.cases) so a smoke run stays tiny.
        capped = 0
        for entry in suite.entries:
            req = entry.dataset.required if entry.dataset else None
            if isinstance(req, dict) and req.get("kind"):
                req["min_cases"] = args.cases  # authoritative per-run override
                capped += 1
        print(f"[cap] min_cases <= {args.cases} on {capped} dataset entr(ies)")

        scratch = Path(_DATA_DIR) / "_stage3_offline" / session_id
        plan_path, map_scratch = _write_plan_files(suite, report, scratch, sess["map_path"])

        # --- STAGE 3b: dataset fulfillment (writes versioned rows into aeh.db) ---
        from agent_eval_harness.datasets.fulfillment import fulfill_plan
        only_kind = None if args.all_kinds else "synthetic_agent_io"
        print(f"[stage 3b] fulfilling datasets (only_kind={only_kind}, embeddings off)...")
        result = await fulfill_plan(
            str(plan_path), str(map_scratch), sess["snapshot_id"], str(local_path),
            provider_id, model_id, llm,
            embedding_client=None, codespectra_client=None,
            only_agent_ids=chosen_ids, only_kind=only_kind, session_id=session_id,
        )

        out_path = scratch / "stage3_output.json"
        await _dump_datasets(repository, result, out_path)
    finally:
        if llm is not None:
            await llm.aclose()
        await aeh_close_db()
        await backend_close_db()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run AEH Stage 3 offline for a few agents and dump the output.")
    ap.add_argument("--session", default=None, help="expansion session id (default: newest ready one)")
    ap.add_argument("--k", type=int, default=3, help="number of random agents (default 3)")
    ap.add_argument("--cases", type=int, default=5, help="hard cap on cases per dataset (default 5)")
    ap.add_argument("--agents", default=None, help="comma-separated agent ids (overrides --k/--seed)")
    ap.add_argument("--seed", type=int, default=0, help="random seed for agent choice")
    ap.add_argument("--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high"])
    ap.add_argument("--provider-id", default=None, help="override FSOFT provider row id")
    ap.add_argument("--all-kinds", action="store_true", help="fulfill every dataset kind (else synthetic_agent_io only)")
    ap.add_argument("--smoke-only", action="store_true", help="just test the LLM call and exit")
    ap.add_argument("--list", action="store_true", help="list Stage-3-ready sessions and exit")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
