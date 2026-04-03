"""In-process path lock — prevents concurrent git ops on the same working dir."""
import asyncio
from contextlib import asynccontextmanager

_locks: dict[str, asyncio.Lock] = {}


def _get(path: str) -> asyncio.Lock:
    if path not in _locks:
        _locks[path] = asyncio.Lock()
    return _locks[path]


@asynccontextmanager
async def acquire(path: str):
    """Async context manager that holds an exclusive lock on a repo path."""
    lock = _get(path)
    async with lock:
        yield
