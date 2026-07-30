"""Process-level setup the target needs before any of its agents can run.

This file is yours: run_eval.py calls into it and never overwrites it, so a newer plan can
ship a newer run_eval.py without erasing what you write here.
"""
from typing import Any


def init_target(config: dict[str, Any]) -> Any:
    """Prepare this process so the target's agents can run. May be sync or async."""
    # === IMPLEMENT THIS ===
    #   Call whatever the target requires once per process before its agents work —
    #   typically a database/connection initialiser, plus any config or registry bootstrap.
    #   Copy what the target's own server entrypoint does at startup; services that look
    #   stateless still resolve a shared connection on first use and raise if it was never
    #   opened.
    #   Point it at the same data directory the harness uses: derive paths from
    #   `.aeh/wiring.json` rather than assuming a location, since a target often ships a
    #   second, empty database inside its source tree.
    #   Leave this returning None only if you have confirmed the target needs no setup.
    # === END IMPLEMENT THIS ===
    return None


def teardown_target(config: dict[str, Any]) -> Any:
    """Release what init_target opened. May be sync or async; run_eval.py awaits either."""
    # === IMPLEMENT THIS ===
    #   Mirror init_target: whatever the target's entrypoint closes on shutdown, close here.
    #   This is not optional tidiness. A driver that opened a connection pool and never closed
    #   it can hang after the last case finishes: a non-daemon worker thread keeps the process
    #   alive, and CPython joins those threads BEFORE running atexit handlers, so an atexit
    #   teardown cannot rescue it and `Thread.daemon` cannot be set after start(). The symptom
    #   reads as "the eval is very slow" rather than as a leak.
    #   Leave this as None only if init_target opened nothing.
    # === END IMPLEMENT THIS ===
    return None
