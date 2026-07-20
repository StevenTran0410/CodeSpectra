### Tracer Registration Failure

The tracing SDK (e.g., Haystack) may ship a null-object singleton as the default tracer instance. The generated `run_eval.py` explicitly checks for this and registers a real tracer if needed.

**If you see zero spans in the spanlog after running cases:**
1. Check that the tracer is being registered before the first agent call
2. Verify the isinstance check is catching the SDK's null-tracer type correctly
3. If the SDK's null class name differs, update the isinstance guard in `run_eval.py`
