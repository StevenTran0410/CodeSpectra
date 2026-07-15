"""Makes the CodeSpectra backend package importable from inside the AEH process — AEH and backend are separate packages with separate venvs, but the AEH venv has every transitive dep the agents need, so a plain sys.path insertion is enough."""
from __future__ import annotations

import sys
from pathlib import Path

from agent_eval_harness.config import AEHConfig

_bridged = False


def default_backend_path() -> Path:
    """This repo's own backend/ — pass explicitly in tests, don't fall through to
    AEHConfig.load() (that config may point at a disposable clone for live-run safety)."""
    # this file -> injection -> agent_eval_harness(pkg) -> agent_eval_harness(repo) -> monorepo root
    return Path(__file__).resolve().parents[3] / "backend"


def ensure_backend_importable(backend_source_path: str | None = None) -> Path:
    """Safe to call repeatedly — skips the sys.path insert if the path is already present."""
    global _bridged
    root = Path(
        backend_source_path
        or AEHConfig.load().backend_source_path
        or default_backend_path()
    ).resolve()
    if not (root / "domain").is_dir():
        raise FileNotFoundError(
            f"backend source not found at {root} (no domain/ package). "
            "Set backend_source_path in .aeh/config.yaml or CODESPECTRA_BACKEND_SRC."
        )
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    _bridged = True
    return root
