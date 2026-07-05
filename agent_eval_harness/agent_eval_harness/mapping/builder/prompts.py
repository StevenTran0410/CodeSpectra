"""LLM system prompts for the map builder.

Centralized so a prompt change is one diff, one file, one thing to audit when the
role taxonomy (CS-260 §3) evolves.
"""
from __future__ import annotations

# v1 — role taxonomy per CS-260 §3 (7 concrete roles + unknown)
ROLE_CLASSIFICATION_SYSTEM = """You classify a single source-code component into exactly \
one role from this taxonomy. Return JSON only: {"role": "<role>", "confidence": <0.0-1.0>, \
"reasoning": "<one sentence>"}.

Roles:
- input_guard.rule: deterministic input filtering (length, format, gibberish) — no LLM call
- input_guard.llm: LLM-based admission control (topic, jailbreak, policy)
- orchestrator: decomposes intent, routes to sub-agents, manages retries
- retrieval_agent: agentic retrieval — picks tools, gathers context
- tool: leaf capability invoked by an agent (no sub-agents of its own)
- validator: judges sufficiency/quality of intermediate results, can trigger retry
- writer: produces final grounded output from what it is given
- unknown: none of the above fit, or you are not confident

If uncertain, prefer a lower confidence score and let the caller apply the threshold —
never guess a specific role just to avoid "unknown"."""

# v1 — constraint mining: extract machine-checkable limits from prompt-embedded text
CONSTRAINT_EXTRACTION_SYSTEM = """A source code component contains one or more string \
literals that look like LLM prompts. Read them for machine-checkable numeric limits \
(max items per call, retry caps, allowed route counts). Return JSON only: a list of \
objects {"name": "<snake_case_name>", "value": <number>, "quote": "<the exact substring \
of the literal that states the limit>"}. If no such limit is stated, return an empty list \
[]. Never invent a limit that is not explicitly stated in the text."""
