### Verify Span Capture Before the Full Run — Do Not Skip

Scoring reads spans from `out/eval_log.*.jsonl`. **A run can finish "ok" while producing zero spans, and then it scores nothing** — that silent hole is the exact failure this evaluation exists to catch. Before the full run, run ONE case (`--verify`, or a single agent) and confirm the log gained **≥ 1 span record for that case**. If it did not, stop and adapt as below — never let a 0‑span run pass verify.

**Why it can be zero.** The bundled `tracer.py` implements **Haystack's** tracing protocol (`haystack.tracing.Tracer`) — it does not magically hook other frameworks. Your target's framework is **{{scalar:plan.framework}}**.

- If your target **is** Haystack: the tracer only sees calls that go through Haystack's component‑execution machinery. A bare `agent.run(...)` / direct method call **bypasses it and emits nothing** — you must run the call through a component/pipeline so the tracer fires.
- If your target is **not** Haystack (LangGraph, LangChain, plain Python, …): this tracer captures **nothing automatically**. You must emit the spans yourself.

**Fix it inside your owned dispatch region. Never edit `tracer.py` — it is sha‑locked. Two ways:**

**1) Route through the framework's own tracing.** Reproduce the *minimal* shape the target's real production path uses to run this unit, so the bundled tracer (Haystack) or the framework's native tracer fires. For Haystack that is usually a one‑node `AsyncPipeline` wrapping the agent's component — mirror the target's own pipeline builder and run the case through it instead of calling the agent bare. Use this when you want the framework's rich, nested spans (sub‑components, LLM calls) exactly as production emits them.

**2) Emit one manual span per invocation** (framework‑agnostic, always works — the safe default when in doubt). Time the call and write a single span record; one input→output span per case is all scoring needs:

```python
import time
t0 = time.time()
status = "ok"
try:
    result = await ...        # call this agent's entry unit
except Exception as e:
    status = f"error: {e}"
    raise
finally:
    tracer.write_log_line({
        "record": "span",
        "component_id": "<this agent's id>",   # e.g. the dispatch module's agent id
        "trace_id": case["id"],                # the case id already passed to invoke_agent
        "start_ms": int(t0 * 1000),
        "end_ms": int(time.time() * 1000),
        "status": status,
    })
```

**How to choose.** Look at how the target emits observability in its real run — does it register a tracer, or route everything through a pipeline/graph runner? Mirror that (option 1) if you want fidelity to production. If it is unclear, or the framework has no tracer AEH can hook, use the manual span (option 2) — it reliably gives scoring the one span per case it needs.

**Gate.** After the full run, the span count must equal the case count. If any case yields 0 spans, the dispatch is wrong — fix it and re‑verify before trusting a single score.
