# Fix — Embedding Provider Feature: Review Findings from Code-Review Pass

**To whoever picks this up (Gemini)**: this is a contained fix pass over your own recent
implementation of the embedding-provider feature (`repo_atlas_plan/aeh_dataset_embedding_provider_plan.md`)
— not a new ticket, no `CS-XXX` number. It was found by an 8-angle multi-agent code review + a
1-vote verification pass over the full uncommitted diff; every item below was independently
confirmed against the actual current file contents, not just the diff. Line numbers were verified
this session — re-verify before editing since the tree may have moved.

## ⚠️ Operational instructions — same as every prior AEH handover

1. Track every process you spawn; set explicit timeouts; kill everything before finishing.
2. Full AEH pytest suite must stay green (`pytest` in `agent_eval_harness/`) and full backend
   pytest suite must stay green (`pytest` in `backend/`). `npm run typecheck` must stay clean.
3. Do **not** reintroduce a silent dummy/constant-vector embedding fallback anywhere — that was
   the entire point of the feature this review is fixing bugs in. If a code path can't get a real
   embedding client, it must raise or return a clearly-labeled degraded result, never fake data.
4. If stuck or something doesn't match this doc, stop and report rather than guessing.

---

## Must-fix (crashes / silently-ignored user intent)

### 1) `ragas_judge.py` — `ragas.answer_relevancy` metric crashes on every evaluation run

**File**: `agent_eval_harness/agent_eval_harness/metrics/judges/ragas_judge.py:115`

`make_ragas_embeddings()`'s signature was changed elsewhere in this diff to require a mandatory
`embedding_client` parameter with no default. This call site was never updated:

```python
async def run_ragas_answer_relevancy(
    query: str,
    actual_answer: str,
    llm_client: LLMClient,
    *,
    retrieved_contexts: list[str] | None = None,
    component_id: str | None = None,
    trace_id: str | None = None,
) -> MetricResult:
    ...
    llm_adapter = make_ragas_llm_adapter(llm_client)
    embeddings = make_ragas_embeddings()          # <-- TypeError: missing embedding_client
```

This metric is dispatched from `agent_eval_harness/agent_eval_harness/metrics/sweep.py:273-283`
(`_dispatch_judge`), which is called from `_score_judge_entry` (`sweep.py:207-231`), which is
called from `run_sweep` (`sweep.py:28`) — the entry point for every `aeh eval` / rerun. **Any**
suite using the `ragas.answer_relevancy` metric now crashes the whole evaluation run with an
unhandled `TypeError`, not just this one metric.

**There is currently no embedding-provider configuration surface for the evaluation-RUN flow** (only
the Dataset Review screen's Fulfill Datasets flow has one). Building that full surface (a picker in
the run/rerun UI, threading provider/model through `run_sweep`'s callers in `ui/server.py`/`cli.py`)
is explicitly **out of scope for this fix** — do not build it now. Instead, apply the same graceful-
degradation convention this file **already uses** for its sibling metrics
(`run_ragas_faithfulness`/`run_ragas_context_precision` wrap scoring in
`try: ... except Exception: score = None`, returning a `MetricResult` with `score=None` rather than
crashing):

```python
async def run_ragas_answer_relevancy(
    query: str,
    actual_answer: str,
    llm_client: LLMClient,
    *,
    retrieved_contexts: list[str] | None = None,
    component_id: str | None = None,
    trace_id: str | None = None,
    embedding_client: "EmbeddingClient | None" = None,
) -> MetricResult:
    ...
    llm_adapter = make_ragas_llm_adapter(llm_client)
    if embedding_client is None:
        return MetricResult(
            metric_name="llm_judge.ragas.answer_relevancy",
            metric_class="llm_judge",
            score=None,
            passed=False,
            details={"error": "no embedding_client configured for ragas.answer_relevancy — "
                               "this metric needs an embedding provider, not yet wired into "
                               "the evaluation-run flow"},
            component_id=component_id,
            trace_id=trace_id,
            evaluator="ragas.AnswerRelevancy",
            cost_tokens=0,
        )
    embeddings = make_ragas_embeddings(embedding_client)
```

Import `EmbeddingClient` from `agent_eval_harness.llm.embedding_client` (use a `TYPE_CHECKING`
guard or plain import — check how `LLMClient` is already imported at the top of this file and
match that style). Thread `embedding_client: EmbeddingClient | None = None` (purely additive,
default `None`, no existing caller breaks) through:
- `sweep.py`'s `_dispatch_judge(entry, spans, query, llm_client, trace_id)` → add the same optional
  kwarg, pass it into the `run_ragas_answer_relevancy(...)` call at line 277.
- `_score_judge_entry(...)` → same optional kwarg, pass to `_dispatch_judge`.
- `run_sweep(...)` → same optional kwarg, pass to `_score_judge_entry`.

Do not touch `run_ragas_faithfulness` (needs no embeddings, per its own existing comment) or
`run_ragas_context_precision` (also LLM-only).

**Test**: add/update a test in `agent_eval_harness/tests/test_judges_stubbed.py` calling
`run_ragas_answer_relevancy(...)` with no `embedding_client` — assert it returns a `MetricResult`
with `score=None`, `passed=False`, and does **not** raise. Add a second test passing a fake
`EmbeddingClient` (mirror the pattern already used in `test_dataset_qa_testset.py`/
`test_deepeval_adapter.py` for fake LLM clients) and assert scoring actually proceeds.

---

### 2) CLI regression — `aeh dataset generate --kind qa_testset` is now completely broken

**File**: `agent_eval_harness/agent_eval_harness/cli.py:251`

```python
generator_fn = get_generator(args.kind)
cases = await generator_fn(gen_config, llm_client, args.seed)   # <-- 3 positional args only
```

`qa_testset.generate()`'s signature now requires `embedding_client` (raises `ValueError` when
`None`/missing) — `cli.py` never constructs one, so `aeh dataset generate --kind qa_testset ...` now
always fails. Verified: no other `--kind` value's generator needs `embedding_client`, so this is
scoped to `qa_testset` only.

**Fix**: give `cli.py`'s dataset-generate command the same 3 embedding options the Dataset Review
UI has (OpenAI/Gemini/local), as new CLI flags, mirroring how `--provider-id`/`--backend-url` already
work for the LLM client in this same file (see `_build_llm_client`, ~line 163-178). Add:
- `--embedding-provider-id` (str, optional)
- `--embedding-model-id` (str, optional)
- `--use-local-embedding` (flag, default False)

Add a `_build_embedding_client(args, config) -> EmbeddingClient | None` helper next to
`_build_llm_client`, following the exact same resolve-from-args-or-config-then-construct-proxy-
client pattern, using `CodeSpectraEmbeddingProxyClient` from
`agent_eval_harness.llm.embedding_client`. Pass its result into `generator_fn(gen_config, llm_client,
args.seed, embedding_client=embedding_client)` at the `qa_testset` dispatch site — check
`registry.py`'s `get_generator`/dispatch shape first; if generator calls are positional-only there,
you may need `generator_fn(gen_config, llm_client, args.seed, embedding_client)` positionally instead
(match whatever `qa_testset.generate`'s actual parameter order is — do not guess, read the current
signature before writing this call).

If no embedding option is given on the CLI and `--kind qa_testset` is requested, fail with a clear
argparse-level or early error message ("`--kind qa_testset` requires one of --embedding-provider-id,
--use-local-embedding`") rather than letting it reach `qa_testset.generate()`'s generic ValueError —
better UX, same outcome.

**Test**: add a CLI-level test (or extend whatever test file already covers `aeh dataset generate`)
exercising `--kind qa_testset` with each of the 3 new flag combinations, using a fake embedding
client the same way existing tests fake the LLM client.

---

### 3) Local embedding "enabled" Settings toggle is silently ignored

**Files**: `backend/domain/embeddings/local_model.py:73-76`, `backend/api/external.py:256-262`

```python
def local_embedding_available() -> bool:
    """True if the local model has been (or can be) loaded on this machine."""
    gpu_ok, _ = detect_gpu()
    return gpu_ok
```

This only checks GPU/VRAM — it never reads `_LOCAL_EMBEDDING_ENABLED_KEY`, the app_metadata flag
that `backend/api/local_embedding.py`'s `POST /status` route persists when the user toggles
"Enable local embedding model" in Settings. Compare with the reranker's equivalent
(`backend/domain/retrieval/cross_encoder_rerank.py`'s `is_gpu_reranker_enabled()`), which correctly
checks **both** GPU availability **and** the persisted flag before allowing use. Right now a user can
turn the Settings toggle OFF and the local model still loads and runs whenever a request sets
`use_local=True` — the toggle is decorative.

**Fix**: make `local_embedding_available()` async and check the flag too, mirroring
`is_gpu_reranker_enabled()` exactly:

```python
async def local_embedding_available() -> bool:
    """True if the local model is both GPU-usable AND enabled via the Settings toggle."""
    gpu_ok, _ = detect_gpu()
    if not gpu_ok:
        return False

    from infrastructure.db.database import get_db

    db = get_db()
    async with db.execute(
        "SELECT value FROM app_metadata WHERE key = ?", (_LOCAL_EMBEDDING_ENABLED_KEY,)
    ) as cur:
        row = await cur.fetchone()
    return row is not None and row["value"] == "true"
```

Update `backend/api/external.py:258` to `await local_embedding_available()` (the route is already
`async def`). Double-check `backend/api/local_embedding.py`'s own `get_local_embedding_status`
route — it calls `_get_enabled_flag()` (a near-identical private helper) already; consider whether
`local_embedding_available()` should become the **one** shared implementation both routes call,
rather than keeping two copies of the same app_metadata read (nice-to-have, not required for this
fix — the required part is that the actual embed path stops ignoring the toggle).

**Test**: add a test in `backend/tests/test_embedding_adapters.py` (or wherever local-embedding
tests live) that sets the toggle OFF via the repository/DB directly, then asserts
`local_embedding_available()` returns `False` even when `detect_gpu()` is mocked to return `True`.

---

### 4) Local embedding blocks the entire backend event loop

**Files**: `backend/api/external.py:264`, `backend/domain/embeddings/local_model.py:55-70`

```python
# external.py — async def llm_embed(...)
vectors = embed_texts(body.texts)   # <-- plain sync call, no await, no thread offload
```

`embed_texts()` (`local_model.py:55`) is a plain synchronous function: on first call it blocking-
loads a `SentenceTransformer` onto CUDA (its own docstring: "may take several seconds"), then calls
`model.encode(...)`, a blocking GPU/CPU op. Since `llm_embed` is `async def` and this backend runs a
single uvicorn worker/event loop (`backend/main.py`, no `workers=` override), this call blocks the
**one** thread that serves every other concurrent request — chat completions, retrieval, job status
polling — until it finishes. The rest of this codebase already has the correct convention for this
exact situation: `backend/domain/impact/git_signals.py`, `backend/domain/local_repo/service.py`,
`backend/domain/sync_engine/service.py`, `backend/shared/git_utils.py` all wrap their blocking work
in `await asyncio.to_thread(...)` before calling it from an async handler. This is the one place
that doesn't, and it has by far the heaviest blocking call (multi-second GPU load + encode).

**Fix**: in `backend/api/external.py`'s `llm_embed`, change the local branch to:

```python
    if body.use_local:
        from domain.embeddings.local_model import embed_texts, local_embedding_available
        if not await local_embedding_available():  # also picks up fix #3
            raise HTTPException(
                status_code=503,
                detail="Local embedding model unavailable — no usable GPU on this machine",
            )
        try:
            vectors = await asyncio.to_thread(embed_texts, body.texts)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
```

Add `import asyncio` at the top of `external.py` if not already present.

**Test**: hard to unit-test event-loop blocking directly without a real GPU; at minimum confirm the
route still works with a fake/mocked `embed_texts` (patch `domain.embeddings.local_model.embed_texts`
in the test) and that the call goes through `asyncio.to_thread` (assertable via mocking
`asyncio.to_thread` itself and checking it was invoked with `embed_texts`).

---

## Should-fix (build-breaking / confusing-error, lower blast radius)

### 5) OpenAI `embed()` falls back to the chat model instead of a real embedding model

**File**: `backend/domain/model_connector/openai/adapter.py:141`

```python
async def embed(self, request: EmbedRequest) -> EmbedResponse:
    model = request.model_id or self.config.model_id   # <-- self.config.model_id is the CHAT model
```

Compare with `gemini/adapter.py:112`: `model = request.model_id or "gemini-embedding-001"` — a real
embedding model default. If an embed request omits `model_id`, OpenAI's version sends whatever the
provider's configured **chat** model is (e.g. `gpt-4o`) to `/v1/embeddings`, which OpenAI rejects.

**Fix**:
```python
_DEFAULT_EMBED_MODEL = "text-embedding-3-small"
...
async def embed(self, request: EmbedRequest) -> EmbedResponse:
    model = request.model_id or _DEFAULT_EMBED_MODEL
```

---

### 6) `electron.d.ts` — `fulfillDatasets` body type missing the new embedding fields

**File**: `src/renderer/src/types/electron.d.ts:907-916` (search for `fulfillDatasets:` in the `aeh`
namespace)

`DatasetReviewScreen.tsx` passes `embedding_provider_id`, `embedding_model_id`, `use_local_embedding`
in the body object literal, but the declared type wasn't updated. This is a **build-breaking**
excess-property-check error under `npm run typecheck` — the same class of bug already fixed twice
elsewhere this session (`screens/analysis/index.tsx`, `screens/providers/index.tsx`).

**Fix**: add to the `fulfillDatasets` body type:
```typescript
embedding_provider_id?: string | null
embedding_model_id?: string | null
use_local_embedding?: boolean
```

Run `npm run typecheck` after this change and fix anything else it surfaces in files this diff
touched (do not touch pre-existing unrelated typecheck errors in files this diff didn't change).

---

### 7) Invalid `task_type` surfaces as an opaque 500 instead of a clean 400

**File**: `backend/api/external.py:235-243, 279-284`

```python
class LLMEmbedRequest(BaseModel):
    ...
    task_type: str | None = None   # <-- unconstrained
...
    return await _service.embed(
        EmbedRequest(
            ...
            task_type=body.task_type,  # type: ignore[arg-type]
        )
    )
```

`EmbedRequest.task_type` (`backend/domain/model_connector/types.py`) is a
`Literal["retrieval_document","retrieval_query"] | None`. An out-of-range string raises a pydantic
`ValidationError` constructing `EmbedRequest` **inside** the route body — after FastAPI's own request
parsing already succeeded — so it isn't converted to FastAPI's normal 422/`RequestValidationError`
handling, and `backend/main.py` only registers an exception handler for `ProviderError`. Result: a
bare 500.

**Fix**: change `LLMEmbedRequest.task_type` to the same `Literal["retrieval_document",
"retrieval_query"] | None = None` type (import `Literal` from `typing` if not already imported in
this file) so FastAPI's own request-body validation rejects a bad value with a normal 422 before the
route body ever runs. Remove the now-unnecessary `# type: ignore[arg-type]`.

---

## Nice-to-fix (real, lower severity)

### 8) `release_gpu_cache()` runs after every single embed call, not just between batches

**File**: `backend/domain/embeddings/local_model.py:66-70`

```python
def embed_texts(texts: list[str]) -> list[list[float]]:
    ...
    try:
        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [v.tolist() for v in vectors]
    finally:
        release_gpu_cache()
```

`release_gpu_cache()`'s own docstring (`domain/shared/gpu.py:40-44`) says "call between batches, not
within a single inference call" — but this `finally` runs it after **every** `embed_texts()` call. A
single qa_testset run calls this dozens of times (DeepEval's own internal loop, see finding #9),
each time forcing PyTorch to re-request CUDA memory from the driver instead of reusing its cache —
real, avoidable latency.

**Fix**: remove the `finally: release_gpu_cache()` from `embed_texts()` entirely. Instead, call
`release_gpu_cache()` once at the true end of a job — e.g. in
`agent_eval_harness/agent_eval_harness/datasets/fulfillment.py`'s `fulfill_plan()`, after the whole
per-group loop finishes (success or failure), via the embedding_client's `aclose()`/a new explicit
cleanup call, OR simplest: leave cleanup to whatever already tears down the local model process
boundary, since this is a per-request backend process, not a long-lived worker — confirm with the
GPU reranker's actual usage frequency (`rrf_fusion.py`) whether it even needs this at its own call
frequency before deciding where (if anywhere) to keep a call to it for the embedding path.

---

### 9) DeepEval calls the embedder one chunk at a time instead of batching

**File**: `agent_eval_harness/agent_eval_harness/llm/deepeval_adapter.py:119-121`

The installed `deepeval` package's `ContextGenerator._a_get_n_random_contexts_per_source_file`
computes its full chunk list up front, then loops calling `self.embedder.a_embed_text(chunk)` once
per chunk — even though a batched `a_embed_texts(chunks)` call (which this same adapter class
already implements at line 126-127) would do it in one round trip. This is deepeval's own library
behavior, not something wrong in this repo's adapter code — flagging so it's a known, accepted cost
rather than a surprise: for a testset needing N contexts, embedding now costs N sequential
HTTP+GPU round trips instead of 1. **No code change required** unless corpus sizes in practice turn
out large enough that this becomes a real UX problem (e.g. minutes instead of seconds) — if so,
revisit by monkey-patching/wrapping `ContextGenerator` to pre-embed and cache, which is a bigger,
separate change. Just confirmed and documented here, not asking you to fix it now.

---

### 10) `detect_gpu()`'s global cache ignores `min_vram_gb` after the first call (currently latent)

**File**: `backend/domain/shared/gpu.py:17-37`

```python
def detect_gpu(min_vram_gb: float = GPU_MIN_VRAM_GB) -> tuple[bool, float | None]:
    global _gpu_available, _vram_gb
    if _gpu_available is not None:
        return _gpu_available, _vram_gb   # <-- ignores min_vram_gb on every call after the first
```

Both `cross_encoder_rerank.py` and `local_model.py` currently call `detect_gpu()` with no argument
(both rely on the same shared `GPU_MIN_VRAM_GB = 2.0` constant), so this is **not** an active bug
today — confirmed no observable wrong behavior currently. It's a latent design hazard: if either
feature's VRAM threshold is ever tuned independently, whichever module calls `detect_gpu()` first
would silently determine the cached verdict for the other too.

**Fix (low priority, do last)**: key the cache by the `min_vram_gb` value actually requested, e.g.
```python
_gpu_cache: dict[float, tuple[bool, float | None]] = {}

def detect_gpu(min_vram_gb: float = GPU_MIN_VRAM_GB) -> tuple[bool, float | None]:
    if min_vram_gb in _gpu_cache:
        return _gpu_cache[min_vram_gb]
    ...
    _gpu_cache[min_vram_gb] = (gpu_available, vram_gb)
    return _gpu_cache[min_vram_gb]
```
(VRAM total doesn't change at runtime, so caching per-threshold is safe and still avoids repeated
`torch.cuda` probing.) Only do this if time remains after items 1-9 — this is not currently causing
any wrong behavior.

---

## Suggested build order

1. #1 (ragas_judge.py crash) and #2 (CLI regression) — both are outright breakage of previously-
   working functionality, fix first.
2. #3 (Settings toggle bypass) and #4 (blocking event loop) — both are real correctness/availability
   bugs in the new feature itself.
3. #5, #6, #7 — small, independent, low-risk fixes.
4. #8 — small fix, verify it doesn't regress anything (release_gpu_cache still gets called
   *somewhere* reasonable).
5. #9 — no code change, just confirm you agree it's out of scope for now.
6. #10 — only if time remains.

## Verification

1. `pytest` in `agent_eval_harness/` — full suite green, including new/updated tests for #1, #2, #3.
2. `pytest` in `backend/` — full suite green, including new/updated tests for #3, #4, #5, #7.
3. `npm run typecheck` — clean (confirms #6, and that nothing else broke).
4. Manual: run a suite containing a `ragas.answer_relevancy` gate through a real evaluation run —
   confirm it no longer crashes (degrades gracefully per #1's fix).
5. Manual: `aeh dataset generate --kind qa_testset --embedding-provider-id <id> ...` from the CLI —
   confirm it actually generates cases instead of failing immediately.
6. Manual: toggle "Enable local embedding model" OFF in Settings, then trigger a Fulfill Datasets run
   with `use_local_embedding=true` — confirm it now refuses (503) instead of silently running.

## Acceptance

- [ ] `ragas.answer_relevancy` no longer crashes evaluation runs; degrades to `score=None` with a
      clear reason when no embedding_client is available.
- [ ] `aeh dataset generate --kind qa_testset` works again from the CLI with the new embedding flags.
- [ ] The local-embedding Settings toggle actually gates the local model at the real embed route.
- [ ] Local embedding no longer blocks the backend's event loop (offloaded via asyncio.to_thread).
- [ ] OpenAI embed() defaults to a real embedding model, not the provider's chat model.
- [ ] `npm run typecheck` clean.
- [ ] Invalid `task_type` returns a normal 422, not a bare 500.
- [ ] Full AEH + backend pytest suites green.

## Non-goals (do not build these now)

- A full embedding-provider picker for the evaluation-RUN flow (only Dataset Review's Fulfill
  Datasets flow needs one for this fix — #1's minimal graceful-degradation fix is sufficient).
- Any change to DeepEval's own internal batching behavior (#9).
- Persisting/caching embeddings across runs.
- Anthropic/DeepSeek/Ollama/LM Studio embeddings support.
