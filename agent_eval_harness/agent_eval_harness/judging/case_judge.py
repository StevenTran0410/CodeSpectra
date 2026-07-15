"""Lightweight, ad-hoc case-level scoring for Stage 5's results view — compares an
ingested case's real output against its dataset gold `expected` value. Not the formal
gate/RunSource="ingested" scoring pipeline (that needs span-level component matching this
doesn't); this is a cheaper, always-applicable pass: one LLM call per case does both the
overall semantic-match judgment and (for list-valued fields) synonym-aware item matching —
the LLM only classifies which items mean the same thing, every count/division is plain
Python, and fields it doesn't confidently answer fall back to exact-string matching."""
from __future__ import annotations

import json
from typing import Any

from agent_eval_harness.llm.client import LLMClient, LLMMessage

SEMANTIC_MATCH_SYSTEM = (
    "You are a strict but fair evaluator comparing one AI agent's actual output against a "
    "human-reviewed expected (gold) output, for a single evaluation case. The case can come "
    "from any kind of software-analysis agent (architecture, conventions, risk, onboarding, "
    "and many others) covering any codebase or tech stack — do not assume a specific domain "
    "or field names ahead of time; judge whatever content you are actually given.\n\n"
    "SCORING RULES:\n"
    "1. Judge meaning, never exact text or shape. The two sides may use different field "
    "names, different nesting (e.g. a plain string on one side vs a list of objects on the "
    "other), or different ordering — none of that matters on its own. Only compare the "
    "real-world facts and conclusions each side is actually making.\n"
    "2. Missing pieces of the expected content lower the score, roughly in proportion to how "
    "much of the expected content is absent from the actual output.\n"
    "3. A genuine factual contradiction (actual states something that conflicts with "
    "expected, not just something additional) is worse than a simple omission and should be "
    "penalized more heavily.\n"
    "4. Extra correct information in the actual output that expected doesn't mention is NOT a "
    "penalty by itself — only extra information that is wrong or misleading counts against "
    "the score.\n"
    "5. Paraphrases, synonyms, reordering, and equivalent naming (different but equivalent "
    "ways of naming the same technology, file, or concept) all count as a match.\n\n"
    "Use this scale as an anchor, then use judgment for in-between cases:\n"
    "  0.9-1.0 — same facts/conclusions throughout; only wording, order, or shape differ\n"
    "  0.7-0.89 — mostly aligned; minor omissions or harmless extra detail\n"
    "  0.4-0.69 — partial overlap; a meaningful share of expected content is missing or wrong\n"
    "  0.1-0.39 — mostly different; only incidental overlap with expected\n"
    "  0.0-0.09 — no meaningful overlap, or actual is about a different subject entirely\n\n"
    "If a 'List fields to match' section is given below, also decide, for each such field, "
    "which actual-list item refers to the same real-world thing as which expected-list item — "
    "same rule as above: match on meaning, not wording, and a pair must refer to the same "
    "concept, not merely a related, broader, or narrower one. Only report index pairs; do not "
    "count, total, or compute precision/recall yourself — that is done afterward from your "
    "pairs, deterministically. Omit a field from field_matches if you are unsure, and never "
    "let one index appear in more than one pair for the same field.\n\n"
    "Output ONLY a single JSON object — no markdown code fences, no text before or after it: "
    '{"score": <0.0-1.0>, "notes": "<one or two sentences: what matches, what genuinely '
    'differs>", "field_matches": {"<field_name>": [[<expected_index>, <result_index>], ...], '
    '...}}'
)


def _collect_list_fields(result: Any, expected: Any) -> dict[str, tuple[list, list]]:
    """Fields present as a list on both sides, with at least one non-empty side — the only
    shape precision/recall (exact or LLM-assisted) can honestly be computed for."""
    if not isinstance(result, dict) or not isinstance(expected, dict):
        return {}
    fields: dict[str, tuple[list, list]] = {}
    for key, expected_val in expected.items():
        if not isinstance(expected_val, list):
            continue
        result_val = result.get(key)
        if not isinstance(result_val, list):
            continue
        if not expected_val and not result_val:
            continue
        fields[key] = (expected_val, result_val)
    return fields


def _format_list_fields_section(fields: dict[str, tuple[list, list]]) -> str:
    if not fields:
        return ""
    lines = ["\n\nList fields to match (0-based indices):"]
    for field_name, (expected_list, result_list) in fields.items():
        expected_items = ", ".join(f"[{i}] {v!r}" for i, v in enumerate(expected_list))
        result_items = ", ".join(f"[{i}] {v!r}" for i, v in enumerate(result_list))
        lines.append(f"{field_name}:\n  expected: {expected_items}\n  actual:   {result_items}")
    return "\n".join(lines)


def _judge_user_prompt(result: Any, expected: Any, fields: dict[str, tuple[list, list]]) -> str:
    return (
        f"Expected (gold) output:\n{json.dumps(expected, indent=2, default=str)}\n\n"
        f"Actual output:\n{json.dumps(result, indent=2, default=str)}"
        f"{_format_list_fields_section(fields)}"
    )


def _exact_match_precision_recall(expected_list: list, result_list: list) -> dict[str, Any]:
    expected_set = {str(v).strip().lower() for v in expected_list}
    result_set = {str(v).strip().lower() for v in result_list}
    intersection = expected_set & result_set
    precision = len(intersection) / len(result_set) if result_set else 0.0
    recall = len(intersection) / len(expected_set) if expected_set else 0.0
    return {"precision": precision, "recall": recall, "matched_via": "exact"}


def _validate_field_matches(
    raw_matches: Any, fields: dict[str, tuple[list, list]]
) -> dict[str, dict[str, Any]]:
    """Turns the judge's raw index pairs into precision/recall. The LLM only classifies which
    pairs mean the same thing; every count and division below is plain Python, never trusted
    from the model. Pairs are dropped (not trusted) if out-of-range or if either index was
    already used elsewhere in the same field, enforcing a one-to-one match. A field the judge
    didn't return valid pairs for falls back to exact-string matching rather than scoring 0."""
    raw_matches = raw_matches if isinstance(raw_matches, dict) else {}
    out: dict[str, dict[str, Any]] = {}
    for field_name, (expected_list, result_list) in fields.items():
        pairs = raw_matches.get(field_name)
        if not isinstance(pairs, list):
            out[field_name] = _exact_match_precision_recall(expected_list, result_list)
            continue
        used_expected: set[int] = set()
        used_result: set[int] = set()
        for pair in pairs:
            if not (isinstance(pair, list) and len(pair) == 2):
                continue
            e_idx, r_idx = pair
            if not (isinstance(e_idx, int) and isinstance(r_idx, int)):
                continue
            if not (0 <= e_idx < len(expected_list) and 0 <= r_idx < len(result_list)):
                continue
            if e_idx in used_expected or r_idx in used_result:
                continue
            used_expected.add(e_idx)
            used_result.add(r_idx)
        match_count = len(used_expected)
        out[field_name] = {
            "precision": match_count / len(result_list) if result_list else 0.0,
            "recall": match_count / len(expected_list) if expected_list else 0.0,
            "matched_via": "llm",
        }
    return out


async def judge_case_semantic_match(
    llm_client: LLMClient, result: Any, expected: Any
) -> dict[str, Any]:
    fields = _collect_list_fields(result, expected)
    item_count = sum(len(e) + len(r) for e, r in fields.values())
    response = await llm_client.complete(
        [
            LLMMessage(role="system", content=SEMANTIC_MATCH_SYSTEM),
            LLMMessage(role="user", content=_judge_user_prompt(result, expected, fields)),
        ],
        max_tokens=400 + item_count * 12,
        json_mode=True,
    )
    try:
        parsed = json.loads(response.content)
        score = float(parsed.get("score", 0.0))
        notes = parsed.get("notes", "")
        field_matches = _validate_field_matches(parsed.get("field_matches"), fields)
    except (json.JSONDecodeError, TypeError, ValueError):
        score, notes = 0.0, "judge response was not valid JSON — treated as unscored"
        field_matches = {
            name: _exact_match_precision_recall(exp, res) for name, (exp, res) in fields.items()
        }
    tokens = (response.prompt_tokens or 0) + (response.completion_tokens or 0)
    return {
        "score": max(0.0, min(1.0, score)),
        "notes": notes,
        "model": response.model,
        "tokens": tokens,
        "field_matches": field_matches,
    }


def compute_field_precision_recall(result: Any, expected: Any) -> dict[str, dict[str, Any]]:
    """Exact-string (case-insensitive) set-overlap precision/recall, with no LLM involved.
    Used as the judge's per-field fallback, and kept standalone for callers that want a free,
    deterministic score without an LLM call at all."""
    fields = _collect_list_fields(result, expected)
    return {
        name: _exact_match_precision_recall(exp, res) for name, (exp, res) in fields.items()
    }


AGENT_SUMMARY_SYSTEM = (
    "You are summarizing evaluation results for one AI agent across many test cases, for an "
    "arbitrary kind of software-analysis agent — do not assume any specific domain or agent "
    "type. You will be given, for each case, the semantic-match score another judge already "
    "assigned plus that judge's notes on what matched and what genuinely differed from the "
    "expected (gold) output. Do not re-judge individual cases or second-guess their scores — "
    "synthesize the pattern across all of them.\n"
    "Identify:\n"
    "1. Recurring strengths — what this agent consistently gets right across multiple cases.\n"
    "2. Recurring weaknesses — what it consistently gets wrong or omits, especially a pattern "
    "that repeats across several cases rather than a one-off mistake in a single case.\n"
    "3. Anything structurally suspicious — for example many cases describing content that "
    "looks unrelated to what each case was asking about, which points to a pipeline or wiring "
    "problem upstream rather than the agent itself being low quality. Only call this out if "
    "the notes actually support it.\n"
    "Be concrete: reference roughly how many of the given cases exhibit each pattern you "
    "name. Do not invent detail beyond what the case notes support, and do not simply restate "
    "every individual case — synthesize.\n"
    'Return JSON only: {"insight": "<3-6 sentences covering strengths, weaknesses, and any '
    'structural pattern you notice>"}'
)


def _agent_summary_prompt(case_summaries: list[dict[str, Any]]) -> str:
    lines = [f"{len(case_summaries)} scored case(s) for this agent:"]
    for i, c in enumerate(case_summaries):
        lines.append(f"[{i}] score={c['score']:.2f} — {c['notes']}")
    return "\n".join(lines)


async def summarize_agent_judgments(
    llm_client: LLMClient, case_summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    """One extra LLM call that reads already-computed per-case scores+notes for a single agent
    (not raw case content — those were already distilled by the per-case judge) and produces
    one holistic insight: recurring strengths, recurring weaknesses, structural red flags."""
    response = await llm_client.complete(
        [
            LLMMessage(role="system", content=AGENT_SUMMARY_SYSTEM),
            LLMMessage(role="user", content=_agent_summary_prompt(case_summaries)),
        ],
        max_tokens=500,
        json_mode=True,
    )
    try:
        parsed = json.loads(response.content)
        insight = str(parsed.get("insight", ""))
    except (json.JSONDecodeError, TypeError, ValueError):
        insight = "summary response was not valid JSON — treated as unavailable"
    tokens = (response.prompt_tokens or 0) + (response.completion_tokens or 0)
    return {"insight": insight, "model": response.model, "tokens": tokens}
