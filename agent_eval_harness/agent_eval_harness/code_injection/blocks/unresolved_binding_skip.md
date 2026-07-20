### Case Binding Unresolved

This agent's input case mapping could not be resolved automatically. The generated dispatch code marks this case as `skipped_unsupported` rather than attempting a fallback invocation.

**Reason:** See the REFERENCE.md card for this agent; all input fields should have clear case→kwarg mappings (WIRE table). If a field is missing, it indicates an incomplete contract harvest.

This is not a failure — it means manual review is needed to wire this particular agent's invocation.
