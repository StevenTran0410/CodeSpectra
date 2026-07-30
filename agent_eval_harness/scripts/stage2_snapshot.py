"""Stage-2 regression harness: snapshot what Stage 2 produced, then diff two snapshots.

Purpose: before extending Stage 2 to new frameworks, capture the current per-candidate result;
after the change, re-run Stage 2 and diff — anything that silently disappeared shows up as a
regression instead of being noticed weeks later.

    python scripts/stage2_snapshot.py snapshot -o before.json
    # ... change code, re-run Stage 2 in the app ...
    python scripts/stage2_snapshot.py snapshot -o after.json
    python scripts/stage2_snapshot.py diff before.json after.json     # exit 1 on regression

The snapshot deliberately drops ids, paths and timestamps — they change every run and would drown
the diff. What it keeps is the shape of the result: which components/agents/edges exist and how
they are classified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

_PROVIDER_BOUNDARIES_PATH = (
    Path(__file__).parent.parent / "agent_eval_harness" / "discovery" / "provider_boundaries.yaml"
)


def _boundary_vocab_hash() -> str | None:
    """Content hash of provider_boundaries.yaml, so a diff can tell 'the boundary vocabulary itself
    changed' apart from 'the code changed' when a component's makes_model_call verdict shifts."""
    if not _PROVIDER_BOUNDARIES_PATH.exists():
        return None
    return hashlib.sha256(_PROVIDER_BOUNDARIES_PATH.read_bytes()).hexdigest()[:12]


def _resolve_db(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    # Same resolution the server uses (store/database.py).
    return Path(os.getenv("AEH_DATA_DIR", ".")) / "aeh.db"


def _open_readonly(db: Path) -> tuple[sqlite3.Connection, tempfile.TemporaryDirectory]:
    """The app holds aeh.db open in WAL mode, so read from a private copy (db + -wal + -shm);
    copying the db alone would silently miss everything still sitting in the WAL."""
    tmp = tempfile.TemporaryDirectory(prefix="aeh-snap-")
    dst = Path(tmp.name) / db.name
    for suffix in ("", "-wal", "-shm"):
        src = db.with_name(db.name + suffix)
        if src.exists():
            shutil.copy2(src, dst.with_name(dst.name + suffix))
    conn = sqlite3.connect(dst)
    conn.row_factory = sqlite3.Row
    return conn, tmp


def _jload(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _yload(path: str | None) -> dict | None:
    if not path or not Path(path).exists():
        return None
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _map_summary(m: dict | None) -> dict | None:
    if not m:
        return None
    comps = m.get("components") or []
    return {
        "framework": m.get("framework"),
        "component_count": len(comps),
        "discrepancy_count": len(m.get("discrepancies") or []),
        "components": sorted(
            (
                {
                    "id": c.get("id"),
                    "role": c.get("role"),
                    "role_source": c.get("role_source"),
                    "entry_kind": c.get("entry_kind"),
                    "entry_point": c.get("entry_point"),
                    "file": c.get("file"),
                    "is_tool": bool(c.get("is_tool")),
                    "is_library_object": bool(c.get("is_library_object")),
                    "upstream": sorted(c.get("upstream") or []),
                    "downstream": sorted(c.get("downstream") or []),
                    "constraint_count": len(c.get("constraints") or []),
                }
                for c in comps
            ),
            key=lambda c: c["id"] or "",
        ),
    }


def _flow_summary(a: dict | None) -> dict | None:
    if not a:
        return None
    agents = a.get("agents") or []
    return {
        "agent_count": len(agents),
        "entry_agent_ids": sorted(a.get("entry_agent_ids") or []),
        "unassigned_component_ids": sorted(a.get("unassigned_component_ids") or []),
        "agents": sorted(
            (
                {
                    "id": ag.get("id"),
                    "role": ag.get("role"),
                    "component_ids": sorted(ag.get("component_ids") or []),
                    "parent_agent": ag.get("parent_agent"),
                    # AgentFlow's own field names (mapping/agent_flow.py) — NOT *_agent_ids.
                    "upstream_agents": sorted(ag.get("upstream_agents") or []),
                    "downstream_agents": sorted(ag.get("downstream_agents") or []),
                }
                for ag in agents
            ),
            key=lambda x: x["id"] or "",
        ),
    }


def snapshot(db: Path, session_id: str | None) -> dict:
    conn, tmp = _open_readonly(db)
    try:
        if session_id is None:
            row = conn.execute(
                "SELECT id FROM discovery_sessions ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if row is None:
                sys.exit("No discovery sessions in the database.")
            session_id = row["id"]

        out: dict[str, Any] = {
            "schema": 1, "candidates": [], "boundary_vocab_hash": _boundary_vocab_hash(),
        }
        cands = conn.execute(
            "SELECT * FROM discovery_candidates WHERE session_id = ? ORDER BY name", (session_id,)
        ).fetchall()
        if not cands:
            sys.exit(f"Discovery session {session_id} has no candidates.")

        for c in cands:
            wb = _jload(c["wiring_block_json"], None)
            # Latest completed expansion wins: re-running Stage 2 appends a row, never updates.
            exp = conn.execute(
                "SELECT * FROM expansion_sessions WHERE candidate_id = ? AND status = 'completed' "
                "ORDER BY created_at DESC LIMIT 1",
                (c["id"],),
            ).fetchone()

            entry: dict[str, Any] = {
                "name": c["name"],
                "verdict": c["verdict"],
                "frameworks": _jload(c["frameworks_json"], []),
                "community_id": c["community_id"],
                "wiring": None
                if not wb
                else {
                    "framework": wb.get("framework"),
                    "node_aliases": sorted(n["alias"] for n in wb.get("nodes") or []),
                    "edges": sorted(
                        [e["src"], e["dst"]] for e in wb.get("edges") or []
                    ),
                },
                "matched_file_count": len(_jload(c["matched_files_json"], [])),
                "expansion": None,
            }

            if exp is not None:
                accepted = _jload(exp["accepted_json"], [])
                entry["expansion"] = {
                    # Recorded so a diff can tell "the roles got worse" apart from "this is a
                    # different, freshly-expanded session that was never enriched".
                    "session_id": exp["id"],
                    "created_at": exp["created_at"],
                    "stop_reason": exp["stop_reason"],
                    "accepted_files": sorted(
                        a["file"] for a in accepted if isinstance(a, dict) and a.get("file")
                    ),
                    "boundary_files": sorted(_jload(exp["boundary_json"], [])),
                    "accepted_edges": sorted(
                        [e["src"], e["dst"]] for e in _jload(exp["accepted_edges_json"], [])
                    ),
                    "map": _map_summary(_yload(exp["map_path"])),
                    "agent_flow": _flow_summary(_yload(exp["agent_flows_path"])),
                }
            out["candidates"].append(entry)
        return out
    finally:
        conn.close()
        tmp.cleanup()


# ---- diff ----

REGRESSION = "REGRESSION"
INFO = "info"


def _by_name(snap: dict) -> dict[str, dict]:
    return {c["name"]: c for c in snap["candidates"]}


def _diff_set(label: str, before: list, after: list, findings: list) -> None:
    b, a = {json.dumps(x, sort_keys=True) for x in before}, {json.dumps(x, sort_keys=True) for x in after}
    for gone in sorted(b - a):
        findings.append((REGRESSION, f"{label}: LOST {gone}"))
    for new in sorted(a - b):
        findings.append((INFO, f"{label}: new {new}"))


def diff(before: dict, after: dict) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    if before.get("boundary_vocab_hash") != after.get("boundary_vocab_hash"):
        findings.append((
            INFO,
            f"provider_boundaries.yaml changed: {before.get('boundary_vocab_hash')} -> "
            f"{after.get('boundary_vocab_hash')} -- any makes_model_call shift below may be this, not code",
        ))
    bc, ac = _by_name(before), _by_name(after)

    for name in sorted(set(bc) - set(ac)):
        findings.append((REGRESSION, f"candidate LOST: {name}"))
    for name in sorted(set(ac) - set(bc)):
        findings.append((INFO, f"candidate new: {name}"))

    for name in sorted(set(bc) & set(ac)):
        b, a = bc[name], ac[name]
        p = f"[{name}]"
        for field in ("verdict", "frameworks", "community_id"):
            if b[field] != a[field]:
                findings.append((INFO, f"{p} {field}: {b[field]} -> {a[field]}"))

        bw, aw = b["wiring"], a["wiring"]
        if bool(bw) != bool(aw):
            findings.append((REGRESSION, f"{p} wiring block {'lost' if bw else 'appeared'}"))
        elif bw and aw:
            if bw["framework"] != aw["framework"]:
                findings.append((REGRESSION, f"{p} wiring framework: {bw['framework']} -> {aw['framework']}"))
            _diff_set(f"{p} wiring node", bw["node_aliases"], aw["node_aliases"], findings)
            _diff_set(f"{p} wiring edge", bw["edges"], aw["edges"], findings)

        be, ae = b["expansion"], a["expansion"]
        if be and not ae:
            findings.append((REGRESSION, f"{p} expansion session LOST"))
            continue
        if ae and not be:
            findings.append((INFO, f"{p} expansion session NEW (was never expanded before)"))
            continue
        if not be or not ae:
            continue

        same_session = be.get("session_id") == ae.get("session_id")
        if not same_session:
            # Comparing two different expansion runs is legitimate, but every downstream delta then
            # includes "Stage 2 was re-run", not just "the code changed" — say so once, loudly.
            findings.append((
                INFO,
                f"{p} DIFFERENT expansion session: {be.get('session_id')} "
                f"({be.get('created_at')}) -> {ae.get('session_id')} ({ae.get('created_at')}); "
                f"differences below include re-running Stage 2, not only your change",
            ))
        if be["stop_reason"] != ae["stop_reason"]:
            findings.append((INFO, f"{p} stop_reason: {be['stop_reason']} -> {ae['stop_reason']}"))
        _diff_set(f"{p} accepted file", be["accepted_files"], ae["accepted_files"], findings)
        _diff_set(f"{p} accepted edge", be["accepted_edges"], ae["accepted_edges"], findings)

        bm, am = be["map"], ae["map"]
        if bm and am:
            bcomp = {c["id"]: c for c in bm["components"]}
            acomp = {c["id"]: c for c in am["components"]}
            for gone in sorted(set(bcomp) - set(acomp)):
                findings.append((REGRESSION, f"{p} component LOST: {gone}"))
            for new in sorted(set(acomp) - set(bcomp)):
                findings.append((INFO, f"{p} component new: {new}"))
            for cid in sorted(set(bcomp) & set(acomp)):
                x, y = bcomp[cid], acomp[cid]
                # Role classification is LLM-driven and varies run to run, so only losing a role
                # (something -> unknown) counts as a regression; any other change is informational.
                if x["role"] != y["role"]:
                    lost = y["role"] == "unknown" and x["role"] != "unknown"
                    # A fresh expansion session starts every role at 'unknown' until enrichment
                    # runs, so role loss there says nothing about the code — only a same-session
                    # re-enrich can prove a real regression.
                    sev = REGRESSION if (lost and same_session) else INFO
                    note = "" if (same_session or not lost) else "  (new session - enrich it first)"
                    findings.append((sev, f"{p} component {cid} role: {x['role']} -> {y['role']}{note}"))
                for field in ("entry_kind", "entry_point", "file", "is_tool"):
                    if x[field] != y[field]:
                        findings.append((INFO, f"{p} component {cid} {field}: {x[field]} -> {y[field]}"))
                for field in ("upstream", "downstream"):
                    if x[field] and not y[field]:
                        findings.append((REGRESSION, f"{p} component {cid} {field} emptied: {x[field]}"))
                    elif x[field] != y[field]:
                        findings.append((INFO, f"{p} component {cid} {field}: {x[field]} -> {y[field]}"))

        bf, af = be["agent_flow"], ae["agent_flow"]
        if bf and af:
            bag = {x["id"]: x for x in bf["agents"]}
            aag = {x["id"]: x for x in af["agents"]}
            for gone in sorted(set(bag) - set(aag)):
                findings.append((REGRESSION, f"{p} agent LOST: {gone}"))
            for new in sorted(set(aag) - set(bag)):
                # An invented agent renames every downstream metric just as losing one does -- the
                # agent-set symmetric difference is the regression, not only the shrinking half of it.
                findings.append((REGRESSION, f"{p} agent INVENTED: {new}"))
            for aid in sorted(set(bag) & set(aag)):
                x, y = bag[aid], aag[aid]
                # An agent losing every inter-agent link means the DAG collapsed into isolated
                # nodes — the exact failure that would silently starve Stage 3 of flow context.
                for field in ("upstream_agents", "downstream_agents"):
                    if x[field] and not y[field]:
                        findings.append((REGRESSION, f"{p} agent {aid} {field} emptied: {x[field]}"))
                    elif x[field] != y[field]:
                        findings.append((INFO, f"{p} agent {aid} {field}: {x[field]} -> {y[field]}"))
                if x["component_ids"] != y["component_ids"]:
                    findings.append((INFO, f"{p} agent {aid} components: {x['component_ids']} -> {y['component_ids']}"))
            _diff_set(f"{p} entry agent", bf["entry_agent_ids"], af["entry_agent_ids"], findings)
            if len(af["unassigned_component_ids"]) > len(bf["unassigned_component_ids"]):
                findings.append((
                    INFO,
                    f"{p} unassigned components: {len(bf['unassigned_component_ids'])} -> "
                    f"{len(af['unassigned_component_ids'])}",
                ))
    return findings


def _print_snapshot(snap: dict) -> None:
    for c in snap["candidates"]:
        w, e = c["wiring"], c["expansion"]
        wtxt = f"{w['framework']} {len(w['node_aliases'])}n/{len(w['edges'])}e" if w else "no wiring"
        print(f"  {c['name'][:44]:46} {c['verdict']:10} {wtxt}")
        if e is None:
            print("      stage 2: NOT RUN")
            continue
        m, f = e["map"], e["agent_flow"]
        print(
            f"      stage 2: {len(e['accepted_files'])} accepted, {len(e['boundary_files'])} boundary, "
            f"{len(e['accepted_edges'])} edges, stop={e['stop_reason']}"
        )
        if m:
            roles: dict[str, int] = {}
            for comp in m["components"]:
                roles[comp["role"]] = roles.get(comp["role"], 0) + 1
            print(f"      map    : {m['component_count']} components, roles={roles}")
        if f:
            linked = sum(1 for ag in f["agents"] if ag["upstream_agents"] or ag["downstream_agents"])
            print(
                f"      flow   : {f['agent_count']} agents ({linked} linked in the DAG), "
                f"{len(f['entry_agent_ids'])} entry, {len(f['unassigned_component_ids'])} unassigned"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot", help="capture the current Stage-2 result for every candidate")
    s.add_argument("--db", help="path to aeh.db (default: $AEH_DATA_DIR/aeh.db)")
    s.add_argument("--session", help="discovery session id (default: most recent)")
    s.add_argument("-o", "--out", help="write JSON here (default: stdout summary only)")

    d = sub.add_parser("diff", help="compare two snapshots; exits 1 if anything regressed")
    d.add_argument("before")
    d.add_argument("after")

    args = ap.parse_args()

    if args.cmd == "snapshot":
        snap = snapshot(_resolve_db(args.db), args.session)
        _print_snapshot(snap)
        if args.out:
            Path(args.out).write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nwrote {args.out}")
        return 0

    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    findings = diff(before, after)
    regressions = [f for f in findings if f[0] == REGRESSION]
    for sev, msg in findings:
        print(f"{'!!' if sev == REGRESSION else '  '} {msg}")
    print(f"\n{len(regressions)} regression(s), {len(findings) - len(regressions)} informational change(s)")
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
