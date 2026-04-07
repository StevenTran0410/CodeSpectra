# Known Limitations

This document tracks current technical limits of the open-source build.

## Language and Code Intelligence Coverage

- Symbol extraction quality depends on parser support per language.
- Unsupported languages fall back to regex-based extraction and may miss structure.
- Cross-file reference resolution is still shallow compared to full code graph engines.
- Dynamic imports/reflection-heavy patterns are only partially visible.

## Graph and Retrieval

- Structural graph is primarily file-level, not full call-graph level.
- Very large repositories can still produce slower graph build and retrieval times.
- Retrieval is tuned for practical relevance, not guaranteed global recall.
- Semantic misses can happen when concepts are expressed with weak lexical overlap.

## LLM Output Reliability

- All section outputs are model-dependent.
- Smaller local models may produce weaker structure or incomplete evidence mapping.
- Strict JSON output and repair logic reduce failures, but malformed responses can still occur.

## Runtime and Performance

- Analysis latency depends on provider speed, model size, and repository size.
- Local models can be significantly slower than cloud models for full runs.
- Native acceleration is optional; missing native modules may reduce throughput.

## Security and Privacy

- In local mode, code stays on-device.
- In cloud mode, selected code context is sent to configured providers.
- Secrets are redacted in logs, but operational hygiene is still required for local DB/log files.

## Packaging and Operations

- Installer/runtime hardening is practical but not equivalent to a formal security audit.
- Automatic update channel is not part of the current OSS baseline.
- Team collaboration and multi-user governance features are not included.

