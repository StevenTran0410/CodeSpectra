### ⚠️ Retrieval Behavior in Evaluation

Agent `{{scalar:agent.id}}` requests retrieval internally. During synthetic case
evaluation you **must** provide a mock retrieval context (stub) instead of a live
retrieval client — otherwise the agent retrieves over the real repository and leaks
real-world content into a synthetic case.

The stub is already generated in `CODE.md` — use it; do not substitute a live
retrieval client.
