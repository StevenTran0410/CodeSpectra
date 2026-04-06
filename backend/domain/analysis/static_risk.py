"""Static risk/complexity detectors — heuristic analysis of the repo without LLM.

All functions query the SQLite DB and return serialisable dataclasses.
Results are fed as pre-computed context to RiskComplexityAgent.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import aiosqlite

from shared.logger import logger

# ── tunables ──────────────────────────────────────────────────────────────────
_GOD_OBJECT_LINE_THRESHOLD = 300   # estimated lines (size_bytes / 40)
_GOD_OBJECT_SYMBOL_THRESHOLD = 25  # symbols per file
_SCC_HIGH_MIN = 3                  # SCC size for "high" severity circular import
_TODO_PATTERN = re.compile(r"\b(TODO|FIXME|HACK|XXX|NOSONAR)\b")
_TODO_MODULE_HIGH = 8              # annotations per module → high
_TODO_MODULE_MED = 3               # annotations per module → medium
_TEST_DIR_PATTERN = re.compile(r"(^|/)tests?(/|$)", re.IGNORECASE)
_TEST_FILE_SUFFIX = re.compile(r"(test_|_test|\.test\.|\.spec\.)", re.IGNORECASE)


@dataclass
class RiskFinding:
    category: str        # god_object|circular_import|todo_hotspot|test_gap|blast_radius|config_risk
    severity: str        # high|medium|low
    title: str
    rationale: str
    evidence: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class RiskReport:
    findings: list[RiskFinding]

    def as_context_text(self) -> str:
        """Compact text block fed into LLM system prompt."""
        if not self.findings:
            return "No static risk findings."
        lines: list[str] = ["=== Static Risk Findings ==="]
        for f in self.findings:
            ev = ", ".join(f.evidence[:3]) or "n/a"
            lines.append(
                f"[{f.severity.upper()}][{f.category}] {f.title}\n"
                f"  {f.rationale}\n"
                f"  evidence: {ev}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "findings": [
                {
                    "category": f.category,
                    "severity": f.severity,
                    "title": f.title,
                    "rationale": f.rationale,
                    "evidence": f.evidence,
                    "details": f.details,
                }
                for f in self.findings
            ]
        }


# ── helpers ───────────────────────────────────────────────────────────────────

def _module(rel_path: str) -> str:
    parts = PurePosixPath(rel_path.replace("\\", "/")).parts
    return parts[0] if len(parts) > 1 else "<root>"


def _is_test_file(rel_path: str) -> bool:
    return bool(_TEST_DIR_PATTERN.search(rel_path) or _TEST_FILE_SUFFIX.search(rel_path))


def _tarjan_scc(graph: dict[str, set[str]]) -> list[list[str]]:
    """Iterative Tarjan SCC. Returns list of components with ≥ 2 nodes."""
    index_counter = [0]
    stack: list[str] = []
    on_stack: set[str] = set()
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    sccs: list[list[str]] = []

    def strongconnect(v: str) -> None:
        index[v] = lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in graph.get(v, set()):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            scc: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == v:
                    break
            if len(scc) >= 2:
                sccs.append(scc)

    for node in list(graph.keys()):
        if node not in index:
            strongconnect(node)
    return sccs


# ── detectors ─────────────────────────────────────────────────────────────────

async def detect_god_objects(
    snapshot_id: str, db: aiosqlite.Connection
) -> list[RiskFinding]:
    # file size → estimated lines
    async with db.execute(
        "SELECT rel_path, size_bytes FROM manifest_files WHERE snapshot_id=?",
        (snapshot_id,),
    ) as cur:
        files = {r["rel_path"]: r["size_bytes"] for r in await cur.fetchall()}

    # symbol count per file
    async with db.execute(
        "SELECT rel_path, COUNT(*) as cnt FROM code_symbols WHERE snapshot_id=? GROUP BY rel_path",
        (snapshot_id,),
    ) as cur:
        sym_counts = {r["rel_path"]: r["cnt"] for r in await cur.fetchall()}

    findings: list[RiskFinding] = []
    for rel_path, size_bytes in files.items():
        if _is_test_file(rel_path):
            continue
        est_lines = size_bytes // 40
        sym_cnt = sym_counts.get(rel_path, 0)
        if est_lines >= _GOD_OBJECT_LINE_THRESHOLD and sym_cnt >= _GOD_OBJECT_SYMBOL_THRESHOLD:
            severity = "high" if est_lines >= 600 or sym_cnt >= 50 else "medium"
            findings.append(RiskFinding(
                category="god_object",
                severity=severity,
                title=f"Oversized file: {rel_path}",
                rationale=(
                    f"~{est_lines} estimated lines, {sym_cnt} symbols. "
                    "Likely does too much — high maintenance and change risk."
                ),
                evidence=[rel_path],
                details={"estimated_lines": est_lines, "symbol_count": sym_cnt},
            ))
    findings.sort(key=lambda f: (0 if f.severity == "high" else 1, -f.details.get("estimated_lines", 0)))
    return findings[:10]


async def detect_circular_imports(
    snapshot_id: str, db: aiosqlite.Connection
) -> list[RiskFinding]:
    async with db.execute(
        "SELECT src_path, dst_path FROM structural_graph_edges "
        "WHERE snapshot_id=? AND is_external=0",
        (snapshot_id,),
    ) as cur:
        rows = await cur.fetchall()

    if not rows:
        return []

    graph: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        graph[r["src_path"]].add(r["dst_path"])

    try:
        sccs = _tarjan_scc(dict(graph))
    except Exception as e:
        logger.warning("Tarjan SCC failed: %s", e)
        return []

    findings: list[RiskFinding] = []
    for scc in sorted(sccs, key=lambda s: -len(s))[:8]:
        severity = "high" if len(scc) >= _SCC_HIGH_MIN else "medium"
        files_short = scc[:5]
        findings.append(RiskFinding(
            category="circular_import",
            severity=severity,
            title=f"Circular import cycle ({len(scc)} files)",
            rationale=(
                f"Strongly-connected component of {len(scc)} files detected. "
                "Increases build/test fragility and complicates refactoring. "
                "(Confidence: approximate — dynamic imports not captured.)"
            ),
            evidence=files_short,
            details={"cycle_size": len(scc), "files": scc[:8]},
        ))
    return findings


async def detect_todo_hotspots(
    snapshot_id: str, db: aiosqlite.Connection
) -> list[RiskFinding]:
    async with db.execute(
        "SELECT rel_path, content FROM retrieval_chunks WHERE snapshot_id=?",
        (snapshot_id,),
    ) as cur:
        rows = await cur.fetchall()

    module_counts: dict[str, int] = defaultdict(int)
    module_files: dict[str, set[str]] = defaultdict(set)

    for r in rows:
        count = len(_TODO_PATTERN.findall(r["content"] or ""))
        if count:
            mod = _module(r["rel_path"])
            module_counts[mod] += count
            module_files[mod].add(r["rel_path"])

    findings: list[RiskFinding] = []
    for mod, count in sorted(module_counts.items(), key=lambda x: -x[1]):
        if count < _TODO_MODULE_MED:
            continue
        severity = "high" if count >= _TODO_MODULE_HIGH else "medium" if count >= _TODO_MODULE_MED else "low"
        evidence = sorted(module_files[mod])[:5]
        findings.append(RiskFinding(
            category="todo_hotspot",
            severity=severity,
            title=f"TODO/FIXME hotspot: {mod}/ ({count} annotations)",
            rationale=(
                f"Module '{mod}' has {count} TODO/FIXME/HACK annotations across "
                f"{len(module_files[mod])} files. Signals deferred work and potential instability."
            ),
            evidence=evidence,
            details={"module": mod, "annotation_count": count, "file_count": len(module_files[mod])},
        ))
    return findings[:8]


async def detect_test_coverage_shape(
    snapshot_id: str, db: aiosqlite.Connection
) -> list[RiskFinding]:
    async with db.execute(
        "SELECT rel_path FROM manifest_files WHERE snapshot_id=? AND category='source'",
        (snapshot_id,),
    ) as cur:
        source_files = [r["rel_path"] for r in await cur.fetchall()]

    async with db.execute(
        "SELECT rel_path FROM manifest_files WHERE snapshot_id=?",
        (snapshot_id,),
    ) as cur:
        all_files = [r["rel_path"] for r in await cur.fetchall()]

    test_files = {f for f in all_files if _is_test_file(f)}

    # Group source files by top-level module
    module_source: dict[str, list[str]] = defaultdict(list)
    for f in source_files:
        if not _is_test_file(f):
            module_source[_module(f)].append(f)

    # Which modules have test coverage
    modules_with_tests: set[str] = set()
    for tf in test_files:
        modules_with_tests.add(_module(tf))

    total_test = len(test_files)
    total_source = len(source_files)
    overall_ratio = total_test / total_source if total_source else 0

    findings: list[RiskFinding] = []

    # Modules with no tests
    for mod, src_files in module_source.items():
        if len(src_files) < 3:
            continue
        if mod not in modules_with_tests:
            severity = "high" if len(src_files) >= 10 else "medium"
            findings.append(RiskFinding(
                category="test_gap",
                severity=severity,
                title=f"No test coverage: {mod}/ ({len(src_files)} source files)",
                rationale=f"Module '{mod}' has {len(src_files)} source files but zero test files found.",
                evidence=src_files[:4],
                details={"module": mod, "source_file_count": len(src_files)},
            ))

    # Overall ratio warning
    if overall_ratio < 0.15 and total_source >= 10:
        findings.append(RiskFinding(
            category="test_gap",
            severity="medium",
            title=f"Low overall test ratio ({total_test}/{total_source} = {overall_ratio:.0%})",
            rationale="Less than 15% of source files have corresponding test files (structural estimate).",
            evidence=[],
            details={"test_files": total_test, "source_files": total_source, "ratio": round(overall_ratio, 3)},
        ))

    findings.sort(key=lambda f: (0 if f.severity == "high" else 1 if f.severity == "medium" else 2))
    return findings[:8]


async def detect_blast_radius(
    snapshot_id: str, db: aiosqlite.Connection
) -> list[RiskFinding]:
    async with db.execute(
        "SELECT top_central_files FROM structural_graph_summaries WHERE snapshot_id=?",
        (snapshot_id,),
    ) as cur:
        row = await cur.fetchone()

    if not row:
        return []

    try:
        top_files: list = json.loads(row["top_central_files"] or "[]")
    except Exception:
        return []

    if not top_files:
        return []

    file_list: list[str] = []
    for item in top_files[:10]:
        if isinstance(item, str):
            file_list.append(item)
        elif isinstance(item, dict):
            file_list.append(str(item.get("path") or item.get("file") or item.get("name") or ""))

    file_list = [f for f in file_list if f]
    if not file_list:
        return []

    return [RiskFinding(
        category="blast_radius",
        severity="medium",
        title=f"High blast-radius files ({len(file_list)} identified)",
        rationale=(
            "These files appear in many import paths. "
            "Changes here ripple across the most of the codebase — "
            "should be touched carefully and tested thoroughly."
        ),
        evidence=file_list[:8],
        details={"files": file_list[:10]},
    )]


async def detect_config_env_risk(
    snapshot_id: str, db: aiosqlite.Connection
) -> list[RiskFinding]:
    """Detect env-var spread and potential hardcoded secrets heuristics."""
    _ENV_PATTERN = re.compile(r"os\.environ|process\.env|getenv|dotenv|\.env", re.IGNORECASE)
    _HARDCODE_PATTERN = re.compile(
        r"(password|secret|api_key|token|bearer)\s*=\s*['\"][^'\"]{8,}['\"]",
        re.IGNORECASE,
    )

    async with db.execute(
        "SELECT rel_path, content FROM retrieval_chunks WHERE snapshot_id=?",
        (snapshot_id,),
    ) as cur:
        rows = await cur.fetchall()

    env_files: set[str] = set()
    hardcode_files: set[str] = set()

    for r in rows:
        content = r["content"] or ""
        if _ENV_PATTERN.search(content):
            env_files.add(r["rel_path"])
        if _HARDCODE_PATTERN.search(content):
            hardcode_files.add(r["rel_path"])

    findings: list[RiskFinding] = []

    if len(env_files) >= 8:
        findings.append(RiskFinding(
            category="config_risk",
            severity="medium",
            title=f"Env var references spread across {len(env_files)} files",
            rationale=(
                "Configuration is read from many locations rather than a single config layer. "
                "Makes deployment and environment changes risky."
            ),
            evidence=sorted(env_files)[:6],
            details={"env_ref_count": len(env_files)},
        ))

    if hardcode_files:
        findings.append(RiskFinding(
            category="config_risk",
            severity="high",
            title=f"Possible hardcoded secrets in {len(hardcode_files)} file(s)",
            rationale=(
                "Heuristic pattern matched credential-like assignments. "
                "Verify these are not real secrets committed to source. (Confidence: low — may be test fixtures.)"
            ),
            evidence=sorted(hardcode_files)[:5],
            details={"suspicious_files": len(hardcode_files)},
        ))

    return findings


# ── entry point ───────────────────────────────────────────────────────────────

async def run_risk_analysis(snapshot_id: str, db: aiosqlite.Connection) -> RiskReport:
    """Run all static risk detectors and return a consolidated RiskReport."""
    findings: list[RiskFinding] = []
    detectors = [
        ("god_objects", detect_god_objects),
        ("circular_imports", detect_circular_imports),
        ("todo_hotspots", detect_todo_hotspots),
        ("test_coverage", detect_test_coverage_shape),
        ("blast_radius", detect_blast_radius),
        ("config_risk", detect_config_env_risk),
    ]
    for name, fn in detectors:
        try:
            results = await fn(snapshot_id, db)
            findings.extend(results)
        except Exception as e:
            logger.warning("Static risk detector '%s' failed: %s", name, e)

    # Sort: high first, then medium, then low
    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: (order.get(f.severity, 3), f.category))
    return RiskReport(findings=findings)
