# Self-Retrieve Fixture

Minimal test target for harvesting virtual inputs (evidence from injected retrieval dependencies).

Two agents:

1. **AnnotationTierAgent**: Retrieval service injected with type annotation (`_retriever: SelfRetriever`). Evidence returned by `search()` is typed as `EvidenceBundle`.

2. **UsageTierAgent**: Same injection pattern, but accesses evidence fields via usage-tier patterns (`.get()`, subscript).

Both agents retrieve evidence inside their `run()` method (not via function parameters), requiring virtual-input harvesting to recognize the evidence schema.
