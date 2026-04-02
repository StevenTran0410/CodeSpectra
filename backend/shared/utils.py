"""Shared utility helpers used across domain services."""
import uuid
from datetime import datetime, timezone


def new_id() -> str:
    """Generate a new random UUID string."""
    return str(uuid.uuid4())


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
