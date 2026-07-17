"""LLM system prompts for the map builder."""
from __future__ import annotations

# Role taxonomy: 7 concrete roles + worker + unknown. Single source of truth for role prose —
# discovery/enrichment.py's per-component prompt imports this verbatim, so the vocabulary never
# drifts into a second copy (a past duplicate needed a costly fix).
ROLE_TAXONOMY = """MUST REMEMBER — read before choosing:
"worker" is the NORMAL, EXPECTED answer. In a typical pipeline MOST components are workers.
Answering "worker" is a correct classification, not a failure to classify.
Every specific role below is a STRONG CLAIM about behaviour the component has BEYOND transforming
its input. Answer one only when you can name the evidence in the code that earns it. If you cannot
name that evidence, the answer is "worker" — do not reach for a specific role to seem decisive.

Roles:
- worker: an ordinary node — takes input, transforms it (possibly with one LLM call), emits
  output. THE DEFAULT. Producing a section/field/artifact of its own does not disqualify it.
- input_guard.rule: deterministic input filtering (length, format, gibberish) — no LLM call.
  Evidence: it admits/rejects input before real work happens.
- input_guard.llm: LLM-based admission control (topic, jailbreak, policy). Same evidence, LLM-based.
- orchestrator: constructs and invokes sub-components and routes between them.
  Evidence: it owns/instantiates the sub-components it calls. Merely having downstream nodes is NOT
  evidence — every non-terminal node has downstream.
- retrieval_agent: decides AT RUNTIME what evidence to fetch — it computes or generates its own
  queries (e.g. an LLM sub-call that plans them from a goal) instead of issuing queries fixed in
  the source. THE TEST: given the same input twice, could it fetch different things? If no, its
  queries are constants the developer chose — it is a worker, however many retrieve calls it makes.
  Evidence: a query-planning step whose output feeds the retrieve call. Passing a hardcoded query
  string (or several) to a retriever is NOT evidence — that is a worker.
- tool: leaf capability invoked by an agent, with no sub-agents of its own.
- validator: judges the quality/sufficiency of ANOTHER component's output and can trigger a retry.
  Evidence: it scores/accepts/rejects work that a different component produced. Validating the shape
  of its OWN output is NOT evidence — that is a worker.
- writer: produces the run's FINAL artifact — nothing downstream consumes its output.
  Evidence: it is terminal and composes other components' outputs. Writing text is NOT evidence —
  most components write something.
- unknown: you genuinely cannot tell.

You will be given a list of admissible roles for each component — structural facts have already
ruled some out. Only answer with a role from that component's own admissible list.

You will be given structural facts (fan-in, fan-out) as EVIDENCE, not as rules. High fan-in means
many siblings feed it, which supports validator or writer. Ordinary fan-in/fan-out supports
nothing — judge from the code, and expect "worker".

If uncertain, prefer a lower confidence score and let the caller apply the threshold —
never guess a specific role just to avoid "unknown"."""

# Constraint mining: extract machine-checkable limits from prompt-embedded text
CONSTRAINT_EXTRACTION_SYSTEM = """A source code component contains one or more string \
literals that look like LLM prompts. Read them for machine-checkable numeric limits \
(max items per call, retry caps, allowed route counts). Return JSON only: a list of \
objects {"name": "<snake_case_name>", "value": <number>, "quote": "<the exact substring \
of the literal that states the limit>"}. If no such limit is stated, return an empty list \
[]. Never invent a limit that is not explicitly stated in the text."""

# Agent-flow separation: one holistic pass over the whole map, not per-component. Role
# assignment happens elsewhere (Stage 2.5, per-component, evidence-gated) — grouping judges
# from structural facts (fan-in/fan-out/constructor-fanout/is_tool) + edges + code, not a role guess.
AGENT_FLOW_SYSTEM = """You are given every component of ONE agentic system: its id, structural \
facts, file, entry point, upstream/downstream component-id edges, mined constraints, and a \
source code snippet. Group these components into AGENTS.

An agent is a unit whose input comes from another agent and whose output is consumed by \
another agent — the backbone of the system (e.g. orchestrator -> retrieval_agent -> \
validator -> writer). A component that only serves ONE agent (a tool it calls, a prompt \
template, a helper function/class) is that agent's INTERNAL component, not an agent of its \
own — attach it to the agent that owns it via component_ids, do not list it as a separate \
agent.

Judge each component carefully and independently based on the evidence given (edges, code, \
structural facts) — do not force a component into a grouping the evidence does not support. \
Every component id you were given must end up either inside exactly one agent's component_ids, \
or in unassigned_component_ids if you cannot confidently place it. Never invent a component id \
that was not given to you.

Return JSON only, in this exact schema:
{"agents": [{"id": "<head component id>", "label": "<short human name>", \
"summary": "<one sentence: what this agent does in the flow>", \
"component_ids": ["<component id>", ...], \
"upstream_agents": ["<agent id that feeds this agent>", ...], \
"downstream_agents": ["<agent id this agent feeds>", ...]}], \
"entry_agent_ids": ["<agent id>", ...], \
"unassigned_component_ids": ["<component id>", ...]}"""
