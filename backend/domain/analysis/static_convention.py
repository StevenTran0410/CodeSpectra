"""Static convention miners — heuristic analysis of coding patterns without LLM.

Queries the DB (code_symbols, manifest_files) and returns serialisable dataclasses
that are fed as pre-computed context to ConventionIntelligenceAgent.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import aiosqlite

from shared.logger import logger

# ── known patterns ────────────────────────────────────────────────────────────

_CLASS_SUFFIXES = [
    "Service",
    "Controller",
    "Repository",
    "Handler",
    "Manager",
    "Factory",
    "Builder",
    "Provider",
    "Middleware",
    "Adapter",
    "Decorator",
    "Observer",
    "Strategy",
    "Validator",
    "Serializer",
    "Util",
    "Utils",
    "Helper",
    "Mixin",
]

_FOLDER_ROLES: dict[str, str] = {
    "api": "API layer",
    "controllers": "controllers",
    "controller": "controller",
    "services": "services",
    "service": "service",
    "domain": "domain logic",
    "repositories": "data access",
    "repository": "data access",
    "models": "data models",
    "schemas": "schemas/serialization",
    "utils": "utilities",
    "helpers": "helpers",
    "middleware": "middleware",
    "handlers": "handlers",
    "routes": "routing",
    "components": "UI components",
    "hooks": "React hooks",
    "stores": "state management",
    "screens": "UI screens",
}

_CAMEL_RE = re.compile(r"^[a-z][a-zA-Z0-9]*$")
_PASCAL_RE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
_SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SCREAMING_RE = re.compile(r"^[A-Z][A-Z0-9_]+$")

_TEST_PATTERN = re.compile(r"(test_|_test|\.test\.|\.spec\.)", re.IGNORECASE)
_TEST_DIR = re.compile(r"(^|/)tests?(/|$)", re.IGNORECASE)


@dataclass
class ConventionSignal:
    signal: str  # naming_suffix|naming_style|folder_role|style_signal|anti_pattern
    title: str
    description: str
    confidence: str  # high|medium|low
    evidence: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class ConventionReport:
    signals: list[ConventionSignal]

    def as_context_text(self) -> str:
        if not self.signals:
            return "No static convention signals found."
        lines = ["=== Static Convention Signals ==="]
        for s in self.signals:
            ev = ", ".join(s.evidence[:3]) or "n/a"
            lines.append(
                f"[{s.confidence.upper()}][{s.signal}] {s.title}\n"
                f"  {s.description}\n"
                f"  evidence: {ev}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "signals": [
                {
                    "signal": s.signal,
                    "title": s.title,
                    "description": s.description,
                    "confidence": s.confidence,
                    "evidence": s.evidence,
                    "details": s.details,
                }
                for s in self.signals
            ]
        }


# ── helpers ───────────────────────────────────────────────────────────────────


def _detect_naming_style(names: list[str]) -> str | None:
    camel = sum(1 for n in names if _CAMEL_RE.match(n))
    pascal = sum(1 for n in names if _PASCAL_RE.match(n))
    snake = sum(1 for n in names if _SNAKE_RE.match(n))
    total = len(names)
    if total == 0:
        return None
    dominant = max(camel, pascal, snake)
    if dominant / total < 0.5:
        return None
    if dominant == camel:
        return "camelCase"
    if dominant == pascal:
        return "PascalCase"
    return "snake_case"


def _is_test_file(rel_path: str) -> bool:
    return bool(_TEST_DIR.search(rel_path) or _TEST_PATTERN.search(rel_path))


# ── miners ────────────────────────────────────────────────────────────────────


async def mine_naming_conventions(
    snapshot_id: str, db: aiosqlite.Connection
) -> list[ConventionSignal]:
    async with db.execute(
        "SELECT name, kind, rel_path FROM code_symbols WHERE snapshot_id=?",
        (snapshot_id,),
    ) as cur:
        symbols = await cur.fetchall()

    signals: list[ConventionSignal] = []

    # --- class suffix conventions ---
    class_names = [r["name"] for r in symbols if r["kind"] in ("class", "Class")]
    suffix_counts: Counter[str] = Counter()
    for name in class_names:
        for suffix in _CLASS_SUFFIXES:
            if name.endswith(suffix):
                suffix_counts[suffix] += 1
                break

    dominant_suffixes = [(s, c) for s, c in suffix_counts.most_common(5) if c >= 2]
    if dominant_suffixes:
        desc_parts = [f"{s} ({c}×)" for s, c in dominant_suffixes[:3]]
        signals.append(
            ConventionSignal(
                signal="naming_suffix",
                title="Class naming suffix conventions observed",
                description=f"Dominant class suffixes: {', '.join(desc_parts)}. "
                f"These are the team's architectural role markers.",
                confidence="high" if dominant_suffixes[0][1] >= 5 else "medium",
                evidence=[],
                details={"dominant_suffixes": [s for s, _ in dominant_suffixes]},
            )
        )

    # --- function naming style ---
    func_names = [
        r["name"]
        for r in symbols
        if r["kind"] in ("function", "method") and not r["name"].startswith("_")
    ]
    func_style = _detect_naming_style(func_names[:300])
    if func_style:
        signals.append(
            ConventionSignal(
                signal="naming_style",
                title=f"Function naming: {func_style}",
                description=f"Function/method names predominantly use {func_style} style.",
                confidence="high" if len(func_names) >= 20 else "medium",
                evidence=[],
                details={"style": func_style, "sample_size": len(func_names)},
            )
        )

    # --- file naming patterns ---
    async with db.execute(
        "SELECT rel_path FROM manifest_files WHERE snapshot_id=? AND category='source'",
        (snapshot_id,),
    ) as cur:
        src_files = [r["rel_path"] for r in await cur.fetchall()]

    file_names = [
        PurePosixPath(f.replace("\\", "/")).stem for f in src_files if not _is_test_file(f)
    ]
    file_style = _detect_naming_style(file_names[:300])
    if file_style:
        signals.append(
            ConventionSignal(
                signal="naming_style",
                title=f"File naming: {file_style}",
                description=f"Source file names predominantly use {file_style} style.",
                confidence="medium",
                evidence=[],
                details={"style": file_style, "sample_size": len(file_names)},
            )
        )

    return signals


async def mine_folder_roles(snapshot_id: str, db: aiosqlite.Connection) -> list[ConventionSignal]:
    async with db.execute(
        "SELECT rel_path FROM manifest_files WHERE snapshot_id=? AND category='source'",
        (snapshot_id,),
    ) as cur:
        rows = await cur.fetchall()

    folder_files: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        parts = PurePosixPath(r["rel_path"].replace("\\", "/")).parts
        for part in parts[:-1]:
            role = _FOLDER_ROLES.get(part.lower())
            if role:
                folder_files[f"{part}/ ({role})"].append(r["rel_path"])

    signals: list[ConventionSignal] = []
    for folder_label, files in sorted(folder_files.items(), key=lambda x: -len(x[1])):
        if len(files) < 2:
            continue
        signals.append(
            ConventionSignal(
                signal="folder_role",
                title=f"Folder-role pattern: {folder_label}",
                description=f"{len(files)} source files under this role folder. "
                f"The team uses folder-based architectural separation.",
                confidence="high" if len(files) >= 5 else "medium",
                evidence=files[:4],
                details={"file_count": len(files)},
            )
        )

    return signals[:6]


async def mine_class_function_ratio(
    snapshot_id: str, db: aiosqlite.Connection
) -> list[ConventionSignal]:
    async with db.execute(
        "SELECT kind, language, COUNT(*) as cnt FROM code_symbols WHERE snapshot_id=? "
        "GROUP BY kind, language",
        (snapshot_id,),
    ) as cur:
        rows = await cur.fetchall()

    by_lang: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        lang = (r["language"] or "unknown").lower()
        kind = (r["kind"] or "").lower()
        by_lang[lang][kind] += r["cnt"]

    signals: list[ConventionSignal] = []
    for lang, counts in by_lang.items():
        if lang in ("unknown", ""):
            continue
        classes = counts.get("class", 0)
        functions = counts.get("function", 0) + counts.get("method", 0)
        total = classes + functions
        if total < 10:
            continue
        if classes == 0:
            style = "purely functional"
            desc = f"{lang}: no classes found, {functions} functions — purely functional style."
        elif functions == 0:
            style = "class-heavy (no standalone functions)"
            desc = f"{lang}: {classes} classes, no standalone functions."
        else:
            ratio = classes / total
            if ratio > 0.4:
                style = "class-heavy OOP"
                desc = (
                    f"{lang}: {classes} classes vs {functions} functions "
                    f"({ratio:.0%} class ratio) — OOP style."
                )
            elif ratio < 0.1:
                style = "functional / procedural"
                desc = (
                    f"{lang}: {classes} classes vs {functions} functions "
                    f"({ratio:.0%} class ratio) — functional style."
                )
            else:
                continue  # mixed, not interesting

        signals.append(
            ConventionSignal(
                signal="style_signal",
                title=f"{lang.capitalize()}: {style}",
                description=desc,
                confidence="medium",
                evidence=[],
                details={"language": lang, "classes": classes, "functions": functions},
            )
        )

    return signals[:4]


async def detect_import_boundary_violations(
    snapshot_id: str, db: aiosqlite.Connection
) -> list[ConventionSignal]:
    """Detect files that import from unexpected layers (e.g. domain importing from api)."""
    # Heuristic: if a file in domain/ imports from api/, that is a boundary violation.
    LAYER_ORDER = ["infrastructure", "domain", "service", "api", "controllers", "routes"]

    async with db.execute(
        "SELECT src_path, dst_path FROM structural_graph_edges "
        "WHERE snapshot_id=? AND is_external=0",
        (snapshot_id,),
    ) as cur:
        rows = await cur.fetchall()

    def _layer(path: str) -> int:
        parts = PurePosixPath(path.replace("\\", "/")).parts
        for i, layer in enumerate(LAYER_ORDER):
            if any(layer in p.lower() for p in parts):
                return i
        return -1

    violations: list[tuple[str, str]] = []
    for r in rows:
        src_layer = _layer(r["src_path"])
        dst_layer = _layer(r["dst_path"])
        if src_layer >= 0 and dst_layer >= 0 and dst_layer > src_layer + 1:
            violations.append((r["src_path"], r["dst_path"]))

    if not violations:
        return []

    ev_files = list({v[0] for v in violations[:10]})
    return [
        ConventionSignal(
            signal="anti_pattern",
            title=f"Import boundary violations ({len(violations)} detected)",
            description=(
                f"Files in lower-level layers import from higher-level layers. "
                f"Example: {violations[0][0]} → {violations[0][1]}. "
                f"This couples layers that should be unidirectional."
            ),
            confidence="medium",
            evidence=ev_files[:5],
            details={
                "violation_count": len(violations),
                "examples": [{"from": v[0], "to": v[1]} for v in violations[:5]],
            },
        )
    ]


# ── entry point ───────────────────────────────────────────────────────────────


async def run_convention_analysis(snapshot_id: str, db: aiosqlite.Connection) -> ConventionReport:
    """Run all convention miners and return a ConventionReport."""
    signals: list[ConventionSignal] = []
    miners = [
        ("naming", mine_naming_conventions),
        ("folders", mine_folder_roles),
        ("class_fn_ratio", mine_class_function_ratio),
        ("import_boundaries", detect_import_boundary_violations),
    ]
    for name, fn in miners:
        try:
            results = await fn(snapshot_id, db)
            signals.extend(results)
        except Exception as e:
            logger.warning("Convention miner '%s' failed: %s", name, e)

    return ConventionReport(signals=signals)
