"""LLM system prompts for the map builder."""
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

# v1 — agent-flow separation: one holistic pass over the whole map, not per-component
AGENT_FLOW_SYSTEM = """You are given every component of ONE agentic system: its id, a role \
hint, file, entry point, upstream/downstream component-id edges, mined constraints, and a \
source code snippet. Group these components into AGENTS.

An agent is a unit whose input comes from another agent and whose output is consumed by \
another agent — the backbone of the system (e.g. orchestrator -> retrieval_agent -> \
validator -> writer). A component that only serves ONE agent (a tool it calls, a prompt \
template, a helper function/class) is that agent's INTERNAL component, not an agent of its \
own — attach it to the agent that owns it via component_ids, do not list it as a separate \
agent.

Judge each component carefully and independently based on the evidence given (edges, code, \
role hint) — do not force a component into a grouping the evidence does not support. Every \
component id you were given must end up either inside exactly one agent's component_ids, or \
in unassigned_component_ids if you cannot confidently place it. Never invent a component id \
that was not given to you.

Return JSON only, in this exact schema:
{"agents": [{"id": "<head component id>", "label": "<short human name>", \
"role": "<orchestrator|retrieval_agent|validator|writer|input_guard.rule|input_guard.llm|tool|unknown>", \
"summary": "<one sentence: what this agent does in the flow>", \
"component_ids": ["<component id>", ...], \
"upstream_agents": ["<agent id that feeds this agent>", ...], \
"downstream_agents": ["<agent id this agent feeds>", ...], \
"parent_agent": "<primary calling agent id, or null for an entry/root agent>"}], \
"entry_agent_ids": ["<agent id>", ...], \
"unassigned_component_ids": ["<component id>", ...]}"""
