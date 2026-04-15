"""Shared HTTP utilities for FastAPI routers."""
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import HTTPException

F = TypeVar("F", bound=Callable[..., Any])


def handle_value_error(func: F) -> F:
    """Decorator that converts ValueError -> HTTP 400 and RuntimeError -> HTTP 503.

    Apply to any async or sync FastAPI endpoint that raises ValueError for
    bad-input conditions. Replaces the boilerplate::

        try:
            return await service.method(...)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    """
    if _is_async(func):
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

        return async_wrapper  # type: ignore[return-value]
    else:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

        return sync_wrapper  # type: ignore[return-value]


def _is_async(func: Callable[..., Any]) -> bool:
    import asyncio
    return asyncio.iscoroutinefunction(func)
