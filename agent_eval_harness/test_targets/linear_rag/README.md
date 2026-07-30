# T1 — linear_rag

AEH's simplest synthetic test target (CS-261 §6): `retriever -> writer`, a
Haystack `AsyncPipeline` over a small committed corpus (`corpus/`, ~15 neutral
policy snippets). No LLM in the retriever — a pure Python keyword-overlap
ranker, so half the pipeline is genuinely offline-deterministic even without a
stubbed LLM client.

## Usage

```python
from test_targets.linear_rag.pipeline import build_pipeline
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.llm.client import LLMResponse

handle = build_pipeline(FakeLLMClient(LLMResponse(content="...", model="fake-mini")))
```

## Planted defect

- `AEH_DEFECT_WRITER_HALLUCINATE=1` — writer appends a fabricated claim absent
  from the retrieved context, after the LLM call completes (shared with T2).

## Tier-2 (boundary-wrapper) demo

`run_retrieve(query)` / `run_write(query, prior_output)` in `pipeline.py` expose
the SAME underlying logic as the Haystack components above, as plain async
functions — used by CS-261's Tier-2 fallback acceptance test (pretending this
target has no native tracing).

## Golden map

`system_map.yaml` is this target's golden System Map (CS-260 §4c) — the
validation goldens CS-264 will later be graded against structurally.
