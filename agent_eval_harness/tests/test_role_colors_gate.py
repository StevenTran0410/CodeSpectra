"""CS-300 AC4: cross-language gate — ROLE_COLORS (TypeScript) must stay set-equal to
VALID_ROLES (Python). `Record<AEHRole, string>` + tsc catches a MISSING key at compile time,
but tsc cannot see Python, and this repo's own B1 (worker shipped backend-only, 12/13
components rendered grey 'unknown') proves a drifted vocabulary ships silently otherwise.
This test is the mechanical catch for that bug class — not code review."""
from __future__ import annotations

import re
from pathlib import Path

from agent_eval_harness.mapping.builder.roles import VALID_ROLES

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROLE_COLORS_TSX = _REPO_ROOT / "src" / "renderer" / "src" / "screens" / "aeh" / "AgentSubGraphPanel.tsx"


def _parse_role_colors_keys(path: Path) -> set[str]:
    """Extract the object-literal keys of `export const ROLE_COLORS: Record<AEHRole, string> = {...}`."""
    content = path.read_text(encoding="utf-8")
    match = re.search(r"ROLE_COLORS:\s*Record<AEHRole,\s*string>\s*=\s*\{(.*?)\}", content, re.DOTALL)
    assert match, f"Could not find the ROLE_COLORS object literal in {path}"
    body = match.group(1)

    keys: set[str] = set()
    for line in body.splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith("//"):
            continue
        key_part = line.split(":", 1)[0].strip()
        key = key_part[1:-1] if key_part[:1] in ("'", '"') else key_part
        if key:
            keys.add(key)
    return keys


def test_role_colors_keys_are_set_equal_to_valid_roles():
    keys = _parse_role_colors_keys(_ROLE_COLORS_TSX)

    # A silently-empty parse would pass a subset/equality check for the wrong reason — a gate
    # that can silently stop gating is not a gate.
    assert keys, f"Parsed 0 keys from ROLE_COLORS in {_ROLE_COLORS_TSX} — the gate found nothing to check"

    assert keys == VALID_ROLES, (
        f"ROLE_COLORS keys and VALID_ROLES have drifted.\n"
        f"  in ROLE_COLORS but not VALID_ROLES: {sorted(keys - VALID_ROLES)}\n"
        f"  in VALID_ROLES but not ROLE_COLORS: {sorted(VALID_ROLES - keys)}"
    )


def test_aeh_role_union_in_electron_d_ts_is_set_equal_to_valid_roles():
    """Second independent channel: the AEHRole TS union itself, so a drift between the union
    and ROLE_COLORS (not just between ROLE_COLORS and Python) is also caught."""
    electron_d_ts = _REPO_ROOT / "src" / "renderer" / "src" / "types" / "electron.d.ts"
    content = electron_d_ts.read_text(encoding="utf-8")
    match = re.search(r"export type AEHRole =\s*((?:\s*\|\s*'[^']+')+)", content)
    assert match, f"Could not find `export type AEHRole = ...` in {electron_d_ts}"

    roles = set(re.findall(r"'([^']+)'", match.group(1)))
    assert roles, f"Parsed 0 members from the AEHRole union in {electron_d_ts}"
    assert roles == VALID_ROLES, (
        f"AEHRole union and VALID_ROLES have drifted.\n"
        f"  in AEHRole but not VALID_ROLES: {sorted(roles - VALID_ROLES)}\n"
        f"  in VALID_ROLES but not AEHRole: {sorted(VALID_ROLES - roles)}"
    )
