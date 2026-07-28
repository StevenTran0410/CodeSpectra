"""Scoring: case-judge, classifier, metric-registry coverage, DeepEval adapter, and RAGAS/DeepEval judge tests."""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel

from agent_eval_harness.judging.case_judge import (
    compute_field_precision_recall,
    compute_scalar_field_matches,
    judge_case_semantic_match,
    summarize_agent_judgments,
)
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.llm.ragas_adapter import stub_missing_langchain_community_vertexai
from agent_eval_harness.metrics.assertions.registry import ASSERTIONS, ensure_assertions_imported
from agent_eval_harness.metrics.registry import METRIC_REGISTRY, MetricSpec

# Must run before any pytest.importorskip("ragas")/import ragas anywhere below, to patch the missing optional dependency.
stub_missing_langchain_community_vertexai()


# Case judge — semantic-match LLM judging + field precision/recall scoring.
@pytest.mark.asyncio
async def test_judge_case_semantic_match_parses_valid_response() -> None:
    client = FakeLLMClient(LLMResponse(
        content=json.dumps({"score": 0.85, "notes": "domain and purpose match; runtime_type differs"}),
        model="gpt-5.4-mini",
        prompt_tokens=200,
        completion_tokens=30,
    ))

    result = await judge_case_semantic_match(
        client, {"domain": "mobile app"}, {"domain": "workforce management"}
    )

    assert result["score"] == 0.85
    assert "runtime_type" in result["notes"]
    assert result["model"] == "gpt-5.4-mini"
    assert result["tokens"] == 230


@pytest.mark.asyncio
async def test_judge_case_semantic_match_clamps_out_of_range_score() -> None:
    client = FakeLLMClient(LLMResponse(
        content=json.dumps({"score": 1.7, "notes": "over-confident judge"}), model="m",
    ))

    result = await judge_case_semantic_match(client, {}, {})

    assert result["score"] == 1.0


@pytest.mark.asyncio
async def test_judge_case_semantic_match_handles_malformed_json() -> None:
    client = FakeLLMClient(LLMResponse(content="not json at all", model="m"))

    result = await judge_case_semantic_match(client, {"a": 1}, {"a": 1})

    assert result["score"] == 0.0
    assert "unparseable" not in result["notes"]  # exact wording isn't load-bearing, just no crash
    assert isinstance(result["notes"], str)


def test_precision_recall_on_matching_list_field() -> None:
    result = {"tech_stack": ["TypeScript", "react-native", "Expo"]}
    expected = {"tech_stack": ["TypeScript", "React Native", "Expo", "Jest"]}

    scores = compute_field_precision_recall(result, expected)

    assert scores["tech_stack"]["precision"] == pytest.approx(2 / 3)  # TypeScript, Expo matched (case-insensitive)
    assert scores["tech_stack"]["recall"] == pytest.approx(2 / 4)


def test_precision_recall_skips_non_list_fields() -> None:
    result = {"domain": "mobile app", "confidence": "high"}
    expected = {"domain": "mobile app", "confidence": "high"}

    scores = compute_field_precision_recall(result, expected)

    assert scores == {}


def test_precision_recall_skips_field_missing_on_result_side() -> None:
    result = {"tech_stack": ["TypeScript"]}
    expected = {"tech_stack": ["TypeScript"], "evidence_files": ["README.md"]}

    scores = compute_field_precision_recall(result, expected)

    assert "evidence_files" not in scores
    assert "tech_stack" in scores


def test_precision_recall_perfect_match() -> None:
    result = {"evidence_files": ["a.py", "b.py"]}
    expected = {"evidence_files": ["a.py", "b.py"]}

    scores = compute_field_precision_recall(result, expected)

    assert scores["evidence_files"]["precision"] == 1.0
    assert scores["evidence_files"]["recall"] == 1.0


@pytest.mark.asyncio
async def test_judge_uses_llm_index_pairs_to_catch_synonyms() -> None:
    # "Postgres" vs "PostgreSQL" would NOT match on exact string overlap — index-pair matching is the whole point of this path.
    client = FakeLLMClient(LLMResponse(content=json.dumps({
        "score": 0.9,
        "notes": "tech stack matches modulo naming",
        "field_matches": {"tech_stack": [[0, 1], [1, 0]]},
    }), model="m"))

    result = await judge_case_semantic_match(
        client,
        {"tech_stack": ["Postgres", "React"]},
        {"tech_stack": ["React Native", "PostgreSQL"]},
    )

    assert result["field_matches"]["tech_stack"]["precision"] == pytest.approx(2 / 2)
    assert result["field_matches"]["tech_stack"]["recall"] == pytest.approx(2 / 2)
    assert result["field_matches"]["tech_stack"]["matched_via"] == "llm"


@pytest.mark.asyncio
async def test_judge_drops_out_of_range_and_duplicate_index_pairs() -> None:
    # index 5 is out of range, and result-index 0 is reused for two pairs — only the first legitimate, non-reused pair should ever be counted.
    client = FakeLLMClient(LLMResponse(content=json.dumps({
        "score": 0.5,
        "notes": "partial",
        "field_matches": {"tech_stack": [[0, 0], [1, 5], [2, 0]]},
    }), model="m"))

    result = await judge_case_semantic_match(
        client,
        {"tech_stack": ["TypeScript"]},
        {"tech_stack": ["TypeScript", "React", "Jest"]},
    )

    pr = result["field_matches"]["tech_stack"]
    assert pr["precision"] == pytest.approx(1 / 1)  # only [0,0] survives validation
    assert pr["recall"] == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_judge_falls_back_to_exact_match_when_field_omitted() -> None:
    # judge answers score/notes but forgets field_matches entirely for tech_stack.
    client = FakeLLMClient(LLMResponse(content=json.dumps({
        "score": 0.7, "notes": "ok", "field_matches": {},
    }), model="m"))

    result = await judge_case_semantic_match(
        client,
        {"tech_stack": ["TypeScript", "Jest"]},
        {"tech_stack": ["typescript", "Jest", "Expo"]},
    )

    pr = result["field_matches"]["tech_stack"]
    assert pr["matched_via"] == "exact"
    assert pr["precision"] == pytest.approx(2 / 2)  # typescript, jest matched case-insensitively
    assert pr["recall"] == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_judge_falls_back_to_exact_match_on_malformed_json() -> None:
    client = FakeLLMClient(LLMResponse(content="not json at all", model="m"))

    result = await judge_case_semantic_match(
        client,
        {"tech_stack": ["TypeScript"]},
        {"tech_stack": ["TypeScript"]},
    )

    pr = result["field_matches"]["tech_stack"]
    assert pr["matched_via"] == "exact"
    assert pr["precision"] == 1.0
    assert pr["recall"] == 1.0


def test_f1_balances_precision_and_recall() -> None:
    # 2 of 3 actual items are gold, covering 2 of 4 gold items — precision alone would hide the half-missing gold, so F1 must sit below both.
    result = {"tech_stack": ["TypeScript", "Expo", "Vite"]}
    expected = {"tech_stack": ["TypeScript", "Expo", "Jest", "Webpack"]}

    pr = compute_field_precision_recall(result, expected)["tech_stack"]

    assert pr["precision"] == pytest.approx(2 / 3)
    assert pr["recall"] == pytest.approx(2 / 4)
    assert pr["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))


def test_empty_gold_list_is_unscorable_not_zero() -> None:
    # Gold asserts nothing exists; there is no recall to compute, so the field must come back unscored rather than a 0.0 that drags averages down.
    result = {"frameworks": ["Node.js", "CommonJS"]}
    expected = {"frameworks": []}

    pr = compute_field_precision_recall(result, expected)["frameworks"]

    assert pr["f1"] is None
    assert pr["recall"] is None
    assert pr["unscorable"] == "gold_empty"


def test_empty_result_against_real_gold_still_scores_zero() -> None:
    # The mirror case is a genuine miss and must NOT be excused as unscorable.
    result = {"external_integrations": []}
    expected = {"external_integrations": ["LDAP", "SMTP"]}

    pr = compute_field_precision_recall(result, expected)["external_integrations"]

    assert pr["recall"] == 0.0
    assert pr["f1"] == 0.0
    assert "unscorable" not in pr


@pytest.mark.asyncio
async def test_one_coarse_gold_item_may_cover_several_actual_items() -> None:
    # Gold groups the three routers into one entry; the agent lists each separately — a granularity difference, not three wrong answers.
    client = FakeLLMClient(LLMResponse(content=json.dumps({
        "score": 0.9,
        "notes": "same routers, listed individually",
        "field_matches": {"main_services": [[0, 0], [0, 1], [0, 2]]},
    }), model="m"))

    result = await judge_case_semantic_match(
        client,
        {"main_services": ["health route", "auth route", "students route"]},
        {"main_services": ["the health/auth/students routers"]},
    )

    pr = result["field_matches"]["main_services"]
    assert pr["recall"] == pytest.approx(1 / 1)
    assert pr["precision"] == pytest.approx(3 / 3)
    assert pr["f1"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_scalar_field_disagreement_is_scored() -> None:
    # `confidence` is scalar, so no precision/recall metric can see it — yet claiming "high" where gold says "low" is exactly the overconfidence worth catching.
    client = FakeLLMClient(LLMResponse(content=json.dumps({
        "score": 0.8, "notes": "ok", "field_matches": {},
    }), model="m"))

    result = await judge_case_semantic_match(
        client, {"confidence": "medium"}, {"confidence": "low"}
    )

    assert result["scalar_matches"]["confidence"]["score"] == 0.0
    assert result["scalar_matches"]["confidence"]["actual"] == "medium"


def test_scalar_matching_skips_long_free_text() -> None:
    # A mermaid diagram is scalar but has no single right answer — exact comparison would report a meaningless 0 on every case.
    long_text = "graph TD\n" + "  A --> B\n" * 20
    matches = compute_scalar_field_matches(
        {"confidence": "high", "mermaid_diagram": long_text},
        {"confidence": "high", "mermaid_diagram": "graph TD\n  X --> Y"},
    )

    assert "mermaid_diagram" not in matches
    assert matches["confidence"]["score"] == 1.0


@pytest.mark.asyncio
async def test_summarize_agent_judgments_parses_valid_response() -> None:
    client = FakeLLMClient(LLMResponse(
        content=json.dumps({"insight": "Consistently strong on frameworks, weak on evidence_files."}),
        model="deepseek-v4-flash", prompt_tokens=300, completion_tokens=40,
    ))
    case_summaries = [
        {"score": 0.9, "notes": "close match"},
        {"score": 0.2, "notes": "wrong domain entirely"},
    ]

    result = await summarize_agent_judgments(client, case_summaries)

    assert result["insight"] == "Consistently strong on frameworks, weak on evidence_files."
    assert result["model"] == "deepseek-v4-flash"
    assert result["tokens"] == 340
    # both cases' score+notes must actually reach the model, not just the count
    sent_prompt = client.calls[0][1].content
    assert "0.90" in sent_prompt and "close match" in sent_prompt
    assert "0.20" in sent_prompt and "wrong domain entirely" in sent_prompt


@pytest.mark.asyncio
async def test_summarize_agent_judgments_handles_malformed_json() -> None:
    client = FakeLLMClient(LLMResponse(content="not json at all", model="m"))

    result = await summarize_agent_judgments(client, [{"score": 0.5, "notes": "n/a"}])

    assert "not valid JSON" in result["insight"]


# Classifier scorer — confusion-matrix accuracy over a stub guard component.
async def _seed_dataset(dataset_id: str, cases: list[dict]) -> None:
    """Insert raw case dicts directly into dataset_cases."""
    from agent_eval_harness.store.database import get_db

    db = get_db()
    rows = [
        (
            str(uuid.uuid4()),
            dataset_id,
            json.dumps(c["input"]),
            json.dumps(c["expected"]),
            json.dumps(c.get("labels", {})),
            "handwritten",
        )
        for c in cases
    ]
    await db.executemany(
        "INSERT INTO dataset_cases "
        "(id, dataset_id, input_json, expected_json, labels_json, provenance) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    await db.commit()


class _StubGuardComponent:
    """Stub guard component that returns 'pass' for long queries, 'reject' for short."""

    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    async def run_async(self, query: str) -> dict:
        if len(query) >= 5:
            return {"verdict": "pass"}
        return {"verdict": "reject"}


@pytest.fixture
async def _init_db(tmp_path, monkeypatch):
    # Scoped (not autouse) so it only runs for the classifier-scorer tests below that request it explicitly.
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    from agent_eval_harness.store.database import close_db, init_db

    await init_db()
    yield
    await close_db()


async def test_classifier_scorer_known_confusion_matrix(_init_db) -> None:
    """Feed a known dataset, assert accuracy + confusion matrix contents."""
    from unittest.mock import MagicMock

    from agent_eval_harness.metrics.classifier import score_classifier

    dataset_id = "test_guard_stub_v1"
    cases = [
        {"input": {"query": "hello world"}, "expected": {"verdict": "pass"}},
        {"input": {"query": "test query here"}, "expected": {"verdict": "pass"}},
        {"input": {"query": "hi"}, "expected": {"verdict": "reject"}},
        {"input": {"query": "no"}, "expected": {"verdict": "reject"}},
    ]
    await _seed_dataset(dataset_id, cases)

    import agent_eval_harness.metrics.classifier as classifier_module

    original_resolve = classifier_module._resolve_entry_point

    def _mock_resolve(entry_point: str):
        return _StubGuardComponent

    classifier_module._resolve_entry_point = _mock_resolve
    try:
        result = await score_classifier(
            dataset_id=dataset_id,
            component_entry_point="stub:StubGuardComponent",
            llm_client=MagicMock(),
            component_id="guard_rule",
            metric_name="classifier.guard_rule_accuracy",
        )
    finally:
        classifier_module._resolve_entry_point = original_resolve

    assert result.metric_class == "classifier"
    assert result.score is not None
    # Stub is deterministic: "hello world" (11 chars >= 5) → pass ✓
    assert result.score == pytest.approx(1.0)
    assert result.passed is True
    assert result.details["evaluated"] == 4


async def test_classifier_scorer_empty_dataset(_init_db) -> None:
    from unittest.mock import MagicMock

    from agent_eval_harness.metrics.classifier import score_classifier

    result = await score_classifier(
        dataset_id="nonexistent_dataset_xyz",
        component_entry_point="stub:StubGuardComponent",
        llm_client=MagicMock(),
    )
    assert result.score is None
    assert "No cases found" in result.details.get("error", "")


# Metric registry coverage — every registered metric must resolve a real scorer (RED the moment a scorer-less metric is added).
_KNOWN_JUDGE_HANDLERS = {"geval", "ragas_faithfulness", "ragas_answer_relevancy", "tool_correctness"}


def _unresolvable_metrics(registry: dict[str, MetricSpec]) -> list[str]:
    """Metrics with no runnable scorer: assertions must be in ASSERTIONS; llm_judges must carry a dispatch handler sweep recognizes."""
    ensure_assertions_imported()
    unresolvable: list[str] = []
    for name, spec in registry.items():
        if spec.metric_class == "assertion":
            if name not in ASSERTIONS:
                unresolvable.append(name)
        elif spec.metric_class == "llm_judge":
            if spec.dispatch not in _KNOWN_JUDGE_HANDLERS:
                unresolvable.append(name)
        # classifier entries are scored via the dataset+entry_point path, no registry scorer.
    return unresolvable


def test_every_registry_metric_resolves_a_scorer() -> None:
    assert _unresolvable_metrics(METRIC_REGISTRY) == []


def test_gate_goes_red_when_a_scorerless_assertion_is_added() -> None:
    """Mutation proof (teeth): a registered-but-unscored assertion is flagged."""
    mutated = dict(METRIC_REGISTRY)
    mutated["__mutation_no_scorer__"] = MetricSpec(metric_class="assertion")
    assert "__mutation_no_scorer__" in _unresolvable_metrics(mutated)


def test_gate_goes_red_when_a_judge_has_unknown_dispatch() -> None:
    """Mutation proof: an llm_judge whose dispatch handler sweep cannot run is flagged."""
    mutated = dict(METRIC_REGISTRY)
    mutated["__mutation_bad_judge__"] = MetricSpec(
        metric_class="llm_judge", dispatch="not_a_real_handler"
    )
    assert "__mutation_bad_judge__" in _unresolvable_metrics(mutated)


# DeepEval adapter — schema-constrained generation must make a real LLM call and return a validated instance, not a hardcoded mock value.
class _Verdict(BaseModel):
    label: str
    confidence: float


def _adapter(client: FakeLLMClient):
    pytest.importorskip("deepeval")
    from agent_eval_harness.llm.deepeval_adapter import make_deepeval_llm_adapter

    return make_deepeval_llm_adapter(client)


async def test_a_generate_without_schema_returns_raw_text() -> None:
    client = FakeLLMClient(LLMResponse(content="plain answer", model="fake"))
    adapter = _adapter(client)

    result = await adapter.a_generate("say something")

    assert result == "plain answer"
    assert client.calls[-1][-1].content == "say something"


async def test_a_generate_with_schema_calls_real_llm_and_validates() -> None:
    client = FakeLLMClient(
        LLMResponse(content=json.dumps({"label": "positive", "confidence": 0.87}), model="fake")
    )
    adapter = _adapter(client)

    result = await adapter.a_generate("classify this", schema=_Verdict)

    assert isinstance(result, _Verdict)
    assert result.label == "positive"
    assert result.confidence == 0.87
    # the schema's JSON Schema must actually be in the prompt sent to the LLM
    sent_prompt = client.calls[-1][-1].content
    assert "label" in sent_prompt and "confidence" in sent_prompt


async def test_a_generate_with_schema_retries_once_on_bad_json() -> None:
    client = FakeLLMClient(
        [
            LLMResponse(content="not json at all", model="fake"),
            LLMResponse(content=json.dumps({"label": "neutral", "confidence": 0.5}), model="fake"),
        ]
    )
    adapter = _adapter(client)

    result = await adapter.a_generate("classify this", schema=_Verdict)

    assert isinstance(result, _Verdict)
    assert result.label == "neutral"
    assert len(client.calls) == 2


async def test_a_generate_with_schema_raises_after_repeated_failure() -> None:
    client = FakeLLMClient(LLMResponse(content="still not json", model="fake"))
    adapter = _adapter(client)

    with pytest.raises(ValueError, match="_Verdict"):
        await adapter.a_generate("classify this", schema=_Verdict)


async def test_a_generate_with_schema_rejects_json_missing_required_fields() -> None:
    client = FakeLLMClient(
        [
            LLMResponse(content=json.dumps({"label": "positive"}), model="fake"),  # missing confidence
            LLMResponse(content=json.dumps({"label": "positive", "confidence": 0.9}), model="fake"),
        ]
    )
    adapter = _adapter(client)

    result = await adapter.a_generate("classify this", schema=_Verdict)

    assert result.confidence == 0.9
    assert len(client.calls) == 2


def test_generate_sync_delegates_to_a_generate_with_schema() -> None:
    client = FakeLLMClient(
        LLMResponse(content=json.dumps({"label": "negative", "confidence": 0.1}), model="fake")
    )
    adapter = _adapter(client)

    result = adapter.generate("classify this", schema=_Verdict)

    assert isinstance(result, _Verdict)
    assert result.label == "negative"


# DeepEval/RAGAS judges — smoke tests against FakeLLMClient.
def _make_fake_client(response: str = "Good response."):
    from agent_eval_harness.llm.client import LLMResponse
    from agent_eval_harness.llm.fake_client import FakeLLMClient

    return FakeLLMClient(LLMResponse(content=response, model="fake-test"))


def _make_spans(writer_context: str = "Employee gets 15 vacation days per year.") -> list[dict]:
    from datetime import UTC, datetime

    return [
        {
            "id": "span-writer-1",
            "component_id": "writer",
            "span_type": "llm",
            "input_json": json.dumps({"context": writer_context, "query": "vacation policy"}),
            "output_json": json.dumps({"answer": "You get 15 vacation days."}),
            "parent_span_id": None,
            "started_at": datetime.now(UTC).isoformat(),
            "details_json": "{}",
        }
    ]


async def test_geval_runs_end_to_end() -> None:
    pytest.importorskip("deepeval")

    from agent_eval_harness.llm.client import LLMResponse
    from agent_eval_harness.llm.fake_client import FakeLLMClient
    from agent_eval_harness.metrics.judges.deepeval_geval import run_geval

    # GEval.a_measure() makes 2 schema-constrained calls in order: generate steps (schema=Steps),
    # then score against them (schema=ReasonScore) — our adapter has no native
    # a_generate_raw_response, so it falls into the schema-based fallback path.
    steps_response = LLMResponse(
        content=json.dumps({
            "steps": [
                "Check whether the actual output directly answers the input question.",
                "Check whether the actual output is factually consistent with the input.",
            ]
        }),
        model="fake-test",
    )
    # GEval's default score_range is 0-10 (no custom rubric); it normalizes by dividing by the range span, so "8" here becomes 0.8 below.
    score_response = LLMResponse(
        content=json.dumps({
            "score": 8,
            "reason": "The response directly and accurately answers the vacation policy question.",
        }),
        model="fake-test",
    )
    llm_client = FakeLLMClient([steps_response, score_response])

    result = await run_geval(
        metric_name="test_quality",
        rubric_text="Evaluate if the response is helpful and accurate.",
        input_text="What is the vacation policy?",
        actual_output="You get 15 vacation days per year.",
        llm_client=llm_client,
        component_id="writer",
    )

    assert result.metric_class == "llm_judge"
    assert "geval" in result.metric_name
    assert result.score == 0.8
    assert result.passed is True
    assert "vacation policy" in result.details["reason"]


async def test_tool_correctness_deterministic() -> None:
    """Tool Correctness is deterministic — no LLM needed."""
    from agent_eval_harness.metrics.judges.deepeval_geval import run_tool_correctness

    spans = [
        {
            "id": "s1",
            "span_type": "tool_call",
            "component_id": "worker",
            "details_json": json.dumps({"raw_tags": {"aeh.tool.name": "case_law_search"}}),
        }
    ]
    result = await run_tool_correctness(
        spans=spans,
        component_id="worker",
        expected_tool_names=["case_law_search"],
    )
    assert result.metric_class == "llm_judge"
    assert result.passed is True
    assert result.cost_tokens == 0


async def test_tool_correctness_wrong_tool() -> None:
    from agent_eval_harness.metrics.judges.deepeval_geval import run_tool_correctness

    spans = [
        {
            "id": "s1",
            "span_type": "tool_call",
            "component_id": "worker",
            "details_json": json.dumps({"raw_tags": {"aeh.tool.name": "decoy_lookup"}}),
        }
    ]
    result = await run_tool_correctness(
        spans=spans,
        component_id="worker",
        expected_tool_names=["case_law_search"],
    )
    assert result.passed is False


async def test_ragas_faithfulness_runs_end_to_end() -> None:
    pytest.importorskip("ragas")

    from agent_eval_harness.metrics.judges.ragas_judge import run_ragas_faithfulness

    llm_client = _make_fake_client()
    spans = _make_spans()

    result = await run_ragas_faithfulness(
        spans=spans,
        query="How many vacation days do I get?",
        actual_answer="You get 15 vacation days per year.",
        llm_client=llm_client,
        writer_component_id="writer",
        component_id="writer",
    )

    assert result.metric_class == "llm_judge"
    assert result.metric_name == "llm_judge.ragas.faithfulness"
    # retrieved_contexts_source must indicate writer span — not gold or corpus
    assert result.details["retrieved_contexts_source"] == "writer_span_input_json"


async def test_ragas_answer_relevancy_no_embedding_client_degrades() -> None:
    pytest.importorskip("ragas")

    from agent_eval_harness.metrics.judges.ragas_judge import run_ragas_answer_relevancy

    llm_client = _make_fake_client()

    result = await run_ragas_answer_relevancy(
        query="How many vacation days do I get?",
        actual_answer="You get 15 vacation days per year.",
        llm_client=llm_client,
        component_id="writer",
        embedding_client=None,
    )

    assert result.metric_class == "llm_judge"
    assert result.metric_name == "llm_judge.ragas.answer_relevancy"
    assert result.score is None
    assert result.passed is False
    assert "no embedding_client configured" in result.details["error"]


async def test_ragas_answer_relevancy_runs_end_to_end_with_embedding_client() -> None:
    pytest.importorskip("ragas")

    from agent_eval_harness.metrics.judges.ragas_judge import run_ragas_answer_relevancy

    # Ragas AnswerRelevancy generates 3 questions (LLM), embeds them + the original (Embeddings),
    # then cosine-similarity scores them — both clients need mocking.
    llm_client = _make_fake_client(
        response=json.dumps({
            "question": "What is the vacation policy?",
            "noncommittal": 0
        })
    )

    class FakeEmbeddingClient:
        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    fake_embed = FakeEmbeddingClient()

    result = await run_ragas_answer_relevancy(
        query="How many vacation days do I get?",
        actual_answer="You get 15 vacation days per year.",
        llm_client=llm_client,
        component_id="writer",
        embedding_client=fake_embed,  # type: ignore[arg-type]
    )

    assert result.metric_class == "llm_judge"
    assert result.metric_name == "llm_judge.ragas.answer_relevancy"
    assert result.score is not None
    # Since cosine similarity of identical vectors is 1.0, score should be ~1.0
    assert result.score > 0.0


# RAGAS faithfulness — must judge against what the writer actually received, not the gold/corpus context.
def _writer_span(context: list[str]) -> dict:
    return {
        "id": "span-writer",
        "component_id": "writer",
        "span_type": "llm",
        "input_json": json.dumps({"context": context, "query": "test"}),
        "output_json": json.dumps({"answer": "The answer is 42."}),
        "parent_span_id": None,
        "started_at": datetime.now(UTC).isoformat(),
        "details_json": "{}",
    }


WRITER_RECEIVED_CONTEXT = ["Only this context was passed to the writer component."]
GOLD_CONTEXT = ["This is the gold standard context from qa_testset — NOT what writer received."]
CORPUS_CONTEXT = ["This is the raw corpus — definitely NOT what writer received."]


async def test_faithfulness_uses_writer_span_input_not_gold() -> None:
    """retrieved_contexts must equal the writer span's input_json.context, not gold."""
    pytest.importorskip("ragas")

    from agent_eval_harness.metrics.judges.ragas_judge import (
        _extract_writer_contexts,
    )

    spans = [_writer_span(WRITER_RECEIVED_CONTEXT)]

    extracted = _extract_writer_contexts(spans, "writer")
    assert extracted == WRITER_RECEIVED_CONTEXT, (
        f"expected writer's received context {WRITER_RECEIVED_CONTEXT!r}, "
        f"got {extracted!r}"
    )
    assert extracted != GOLD_CONTEXT, "MUST NOT use gold context"
    assert extracted != CORPUS_CONTEXT, "MUST NOT use full corpus"


async def test_faithfulness_different_writer_context_produces_different_extraction() -> None:
    """Changing the writer span's input changes the extracted contexts."""
    pytest.importorskip("ragas")

    from agent_eval_harness.metrics.judges.ragas_judge import _extract_writer_contexts

    context_a = ["Context version A — unique string XYZ123"]
    context_b = ["Context version B — unique string ABC456"]

    spans_a = [_writer_span(context_a)]
    spans_b = [_writer_span(context_b)]

    extracted_a = _extract_writer_contexts(spans_a, "writer")
    extracted_b = _extract_writer_contexts(spans_b, "writer")
    assert extracted_a == context_a
    assert extracted_b == context_b
    assert extracted_a != extracted_b


async def test_faithfulness_empty_when_no_writer_span() -> None:
    """If no writer span exists, returns empty list (graceful degradation)."""
    from agent_eval_harness.metrics.judges.ragas_judge import _extract_writer_contexts

    spans = [
        {
            "id": "span-planner",
            "component_id": "planner",
            "span_type": "llm",
            "input_json": json.dumps({"query": "test"}),
            "output_json": json.dumps({"intents": ["a"]}),
            "parent_span_id": None,
            "started_at": datetime.now(UTC).isoformat(),
            "details_json": "{}",
        }
    ]
    result = _extract_writer_contexts(spans, "writer")
    assert result == []


async def test_faithfulness_result_details_declare_source() -> None:
    """The MetricResult.details must explicitly declare retrieved_contexts_source."""
    pytest.importorskip("ragas")

    from agent_eval_harness.llm.client import LLMResponse
    from agent_eval_harness.llm.fake_client import FakeLLMClient
    from agent_eval_harness.metrics.judges.ragas_judge import run_ragas_faithfulness

    llm_client = FakeLLMClient(LLMResponse(content="faithful answer", model="fake"))
    spans = [_writer_span(WRITER_RECEIVED_CONTEXT)]

    result = await run_ragas_faithfulness(
        spans=spans,
        query="What is the answer?",
        actual_answer="The answer is 42.",
        llm_client=llm_client,
        writer_component_id="writer",
        component_id="writer",
    )

    assert result.details["retrieved_contexts_source"] == "writer_span_input_json"
    assert result.details["writer_component_id"] == "writer"
    assert result.details["context_count"] == len(WRITER_RECEIVED_CONTEXT)
