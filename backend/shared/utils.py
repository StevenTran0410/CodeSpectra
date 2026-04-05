"""Shared utility helpers used across domain services."""
import uuid
from datetime import datetime, timezone
from pathlib import Path


def new_id() -> str:
    """Generate a new random UUID string."""
    return str(uuid.uuid4())


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def read_utf8_lenient(path: Path) -> str:
    """Read UTF-8 text and never raise (returns empty on failure)."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
