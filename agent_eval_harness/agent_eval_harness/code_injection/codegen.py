"""Codegen: generates per-agent dispatch modules and retrieval stubs from CompileFacts.

All generated code is deterministic and auditable (grep-clean for target literals).
"""
from __future__ import annotations

import hashlib
import logging
import textwrap
from typing import Any

from agent_eval_harness.code_injection.facts import AgentFacts, Resolved
from agent_eval_harness.datasets.archetype_vocabulary import EVIDENCE_CASE_KEY

logger = logging.getLogger("agent_eval_harness.code_injection.codegen")


def generate_dispatch_module(agent_facts: AgentFacts) -> tuple[str, str]:
    """Generate the dispatch/<agent_id>.py module for one agent; returns (module_code, sha256)."""
    agent_id = agent_facts.agent_id
    invocation_mode = agent_facts.invocation_mode

    if invocation_mode == "in_harness":
        code = _generate_in_harness_dispatch(agent_facts)
    elif invocation_mode == "per_agent_route":
        code = _generate_route_dispatch(agent_facts)
    elif invocation_mode == "pipeline_entry":
        code = _generate_unsupported_dispatch(agent_facts)
    else:
        code = _generate_unsupported_dispatch(agent_facts)

    sha = hashlib.sha256(code.encode()).hexdigest()
    return code, sha


IMPLEMENT_MARKER = "=== IMPLEMENT THIS ==="


def _generate_in_harness_dispatch(agent_facts: AgentFacts) -> str:
    """Generate in-harness dispatch: real kwargs + a specified IMPLEMENT-THIS region."""
    agent_id = agent_facts.agent_id
    location = agent_facts.location
    kwargs_code = _build_kwargs_code(agent_facts.case_binding, agent_facts)

    loc_file, entry_method, entry_line = "<see REFERENCE.md card>", "run", "?"
    if location and isinstance(location, Resolved):
        loc = location.value
        loc_file, entry_method, entry_line = loc.file, loc.entry_method, loc.entry_line
    deps = ", ".join(_constructor_deps(agent_facts)) or "(none harvested)"
    # Eight spaces so textwrap.dedent leaves it aligned with the surrounding comment block.
    stub_note = (
        "        #      Replace its retrieval dependency with `retrieval` built above — never a\n"
        "        #      live client, or the agent reads the real repo instead of this case.\n"
        if agent_facts.has_retrieval_signal else ""
    )
    stub_setup = (
        "\n        # This agent retrieves internally, so the case's evidence reaches it only here.\n"
        "        from retrieval_stub import make_retrieval_stub\n"
        f'        retrieval = make_retrieval_stub(case["input"].get("{EVIDENCE_CASE_KEY}"))\n'
        if agent_facts.has_retrieval_signal else ""
    )
    unusable_note = (
        "\n        # Ignored by design — the entry method has no parameter for these, so the agent\n"
        f"        # builds that context itself: {', '.join(agent_facts.unconsumed_case_fields)}"
        if agent_facts.unconsumed_case_fields else ""
    )

    return textwrap.dedent(f'''
    """Dispatch module for agent: {agent_id}

    Location: {loc_file}:{entry_line} ({entry_method})
    Invocation mode: in_harness
    """
    from typing import Any


    async def invoke_agent(case: dict[str, Any], config: dict[str, Any]) -> Any:
        """Invoke {agent_id} in-process and return its raw output."""
        # --- generated from the harvested case_binding — do not edit ---
        {kwargs_code}
        # A field this case does not carry resolves to None on purpose: synthetic cases
        # have no real snapshot, and the retrieval stub below makes that irrelevant.
        # --- end generated ---{unusable_note}
{stub_setup}
        # {IMPLEMENT_MARKER}
        #   1. Import the agent class defined at {loc_file}:{entry_line}
        #   2. Construct it with its constructor dependencies: {deps}
{stub_note}        #   3. Call its `{entry_method}(**kwargs)` entry method (await it if it is async)
        #   4. Return the raw result unchanged (no schema massaging here)
        #   Read credentials/provider from `config` only — never hardcode any secret.
        # === END IMPLEMENT THIS ===
        raise NotImplementedError(
            "dispatch.{agent_id}.invoke_agent not implemented yet — fill the IMPLEMENT THIS block"
        )


    def skip_case(case: dict[str, Any]) -> bool:
        """Return True if this case should be skipped."""
        return False
    ''').strip()


def _generate_route_dispatch(agent_facts: AgentFacts) -> str:
    """Generate route dispatch: real kwargs + a specified IMPLEMENT-THIS region."""
    agent_id = agent_facts.agent_id
    route = agent_facts.route
    route_value = route.value if route and isinstance(route, Resolved) else "<route not harvested — see RECON>"
    kwargs_code = _build_kwargs_code(agent_facts.case_binding, agent_facts)

    return textwrap.dedent(f'''
    """Dispatch module for agent: {agent_id}

    Route: {route_value}
    Invocation mode: per_agent_route
    """
    import logging
    from typing import Any

    logger = logging.getLogger("dispatch.{agent_id}")


    async def invoke_agent(case: dict[str, Any], config: dict[str, Any]) -> Any:
        """Invoke {agent_id} over its HTTP route and return the parsed output."""
        # --- generated from the harvested case_binding — do not edit ---
        {kwargs_code}
        # --- end generated ---

        # {IMPLEMENT_MARKER}
        #   Send this request to the running target and return its parsed output:
        #     method : POST
        #     url    : config["base_url"] + "{route_value}"
        #     body   : JSON object built from `kwargs` above
        #     auth   : config.get("auth_header") — omit if the target needs none
        #     return : the response payload holding this agent's output
        #   Use the HTTP client the target project already depends on.
        #   Read base_url/auth/provider from `config` (run_config.json) — never hardcode a secret.
        #   Raise on a non-2xx response so the case is recorded as an error, not a pass.
        # === END IMPLEMENT THIS ===
        raise NotImplementedError(
            "dispatch.{agent_id}.invoke_agent not implemented yet — fill the IMPLEMENT THIS block"
        )


    def skip_case(case: dict[str, Any]) -> bool:
        """Return True if this case should be skipped."""
        return False
    ''').strip()


def _generate_unsupported_dispatch(agent_facts: AgentFacts) -> str:
    """Generate dispatch that skips the agent — no invocation contract was harvested."""
    agent_id = agent_facts.agent_id

    return textwrap.dedent(f'''
    """Dispatch module for agent: {agent_id}

    Invocation mode: UNSUPPORTED — no invocation contract could be harvested.
    Every case is skipped and recorded as skipped_unsupported. Do NOT invent an
    invocation here; report it back instead so the plan can be regenerated.
    """
    from typing import Any


    async def invoke_agent(case: dict[str, Any], config: dict[str, Any]) -> Any:
        """Not invocable — skip_case() returns True so this is never called."""
        raise NotImplementedError("{agent_id} has no supported invocation contract")


    def skip_case(case: dict[str, Any]) -> bool:
        """This agent cannot be invoked — skip all cases."""
        return True
    ''').strip()


def _constructor_deps(agent_facts: AgentFacts) -> list[str]:
    """Constructor dependency names harvested for this agent, if any."""
    roles = agent_facts.constructor_dep_roles
    if roles and isinstance(roles, Resolved) and roles.value:
        return [getattr(v, "dep", str(v)) for v in roles.value]
    return list(agent_facts.constructor_deps)


def _build_kwargs_code(case_binding: Any, agent_facts: AgentFacts) -> str:
    """Generate Python code to build kwargs dict from case_binding.

    case_binding is Resolved[dict[str, str]] where values are like:
    - "case:$.input.query" → kwargs["query"] = case["input"]["query"]
    - "config:provider" → kwargs["provider"] = config.get("provider")
    - "const:default_model" → kwargs["model"] = "default_model"
    """
    if not case_binding or not isinstance(case_binding, Resolved):
        return "kwargs = {}"

    binding_dict = case_binding.value or {}
    if not binding_dict:
        return "kwargs = {}"

    lines = ["kwargs = {}"]
    for kwarg_name, binding_expr in binding_dict.items():
        if binding_expr.startswith("case:"):
            # Read the last hop with .get() — a missing optional field must not KeyError the whole run.
            parts = _json_path_parts(binding_expr[5:])
            if not parts:
                continue
            prefix = "".join(f'["{p}"]' for p in parts[:-1])
            lines.append(f'kwargs["{kwarg_name}"] = case{prefix}.get("{parts[-1]}")')

        elif binding_expr.startswith("config:"):
            key = binding_expr[7:]
            lines.append(f'kwargs["{kwarg_name}"] = config.get("{key}")')

        elif binding_expr.startswith("const:"):
            lit = binding_expr[6:]
            lines.append(f'kwargs["{kwarg_name}"] = "{lit}"')

        else:
            logger.warning(f"Unknown binding prefix in {kwarg_name}={binding_expr}")

    return "\n        ".join(lines)


def _json_path_parts(path: str) -> list[str]:
    """Split a JSONPath like "$.input.query" into ["input", "query"]."""
    if not path.startswith("$"):
        path = "$." + path
    return [p for p in path[1:].strip(".").split(".") if p]


def generate_retrieval_stub(agent_facts: AgentFacts) -> str:
    """Skeleton for the retrieval stub; its interface is the target's, so it is not invented.

    Guessing the method shape here once left agents running on zero evidence while still reporting success.
    """
    if not agent_facts.has_retrieval_signal:
        return ""

    deps = ", ".join(_constructor_deps(agent_facts)) or "the retrieval dependency"
    return textwrap.dedent(f'''
    """Retrieval stub: serves one case's own evidence in place of the real index.

    The class below carries the evidence; you give it the target's real interface.
    """
    from typing import Any


    class RetrievalStub:
        """Stands in for the target's retrieval dependency during evaluation."""

        def __init__(self, evidence: Any = None):
            self.evidence = evidence

        # {IMPLEMENT_MARKER}
        #   Give this class the SAME public method(s) as the real retrieval dependency
        #   ({deps}) — same names, same signature, same return type, and async if the real
        #   one is async. Read them from the target's source; do not guess.
        #   Each method must build its return value from `self.evidence` (this case's
        #   own data) and must never touch the real index, network, or database.
        #
        #   `self.evidence` is deliberately partial: it carries the meaning-bearing fields
        #   only (which file, what text, how relevant). Fill whatever else the target's
        #   type requires — ids, indexes, token counts, line numbers, mode/section/budget —
        #   with neutral defaults. Missing bookkeeping is expected, not a conflict.
        #
        #   Return the SAME evidence for every query. A case carries one evidence set, not
        #   one per query, so this harness measures how the agent reasons over given
        #   evidence — not how well it retrieves. Do not fabricate query-specific results.
        #
        #   A mismatch here can be swallowed by the caller's error handling and leave the
        #   agent running on empty evidence while still looking successful — so after
        #   writing it, assert one real call returns evidence-derived content.
        # === END IMPLEMENT THIS ===


    def make_retrieval_stub(evidence: Any = None) -> RetrievalStub:
        """Build the stub for one case; called by every dispatch module that needs it."""
        return RetrievalStub(evidence)
    ''').strip()
