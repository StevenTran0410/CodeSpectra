# RPA-020 - Design the Abstraction for LLM Providers and the Capability Matrix

* **Epic**: Providers
* **Priority**: P0
* **Relative Estimate**: M
* **Milestone**: Milestone C
* **Dependencies**: RPA-001, RPA-011

## 1) Goal

Create a unified adapter layer so the app does not become vendor-locked in the codebase. This is the backbone for both local and cloud providers.

## 2) Problem this ticket solves

If each provider is called directly from multiple places, the app will turn into spaghetti as soon as streaming, structured output, embeddings, or token estimation are added.

## 3) Detailed scope

* Define the standard provider interface: `validateCredential`, `listModels`, `getCapabilities`, `generateStructured`, `generateText`, `embed`, `estimateTokens`.
* Design the capability matrix: `supports_streaming`, `supports_structured_output`, `supports_embeddings`, `is_openai_compatible`, `max_input_hint`, `privacy_class`.
* Design error normalization so the UI does not need to understand vendor-specific errors.
* Design the provider config model and credential reference model.
* Design an adapter registry so new providers can be added without requiring too many changes to existing code.

## 4) Implementation notes

* Structured generation must be a first-class method because the report schema depends heavily on JSON output.
* The capability matrix must be readable by the UI so it can show the correct warnings and disabled states.
* There should be one generic OpenAI-compatible adapter that can be reused for local servers and DeepSeek-like providers.

## 5) Breakdown subtasks

* Write TypeScript interfaces/types for the provider abstraction.
* Write the error mapping layer.
* Create the provider registry + factory.
* Create a mock adapter for local testing.
* Create model picker UI bindings powered by the capability matrix.

## 6) Acceptance criteria

* A standard interface exists and is shared across both local and cloud providers.
* The UI reads the capability matrix and renders the correct state.
* Code that generates report sections does not need to know whether it is calling OpenAI, Ollama, or Gemini.
* Error messages are normalized to a practically usable level.

## 7) Out of scope

* Prompt tuning per provider.
* Multi-provider ensemble behavior.

## 8) Risks / watchpoints

* If the abstraction is too thin, structured output and embeddings will become patchwork later.
* If the abstraction is too ambitious, it will slow delivery.

## 9) Expected deliverables

* LLM provider interface
* Capability matrix spec
* Error normalization layer
* Provider registry

## 10) Definition of done

* Related code/service/UI work has been merged and runs in the local environment.
* The ticket has at least minimal tests or an appropriate verification checklist.
* Logging and error states are not left blank or unhandled.
* Related docs/settings have been updated.
* No major ambiguity remains open without being clearly documented.

## 11) Suggested QA checklist

* Re-run this ticket against at least 1 public repository or 1 internal sample repository.
* Check empty states, error states, and cancel/retry states where applicable.
* Restart the app and return to the flow to verify that the data remains correct.
* Verify that logs do not expose secrets or tokens.
