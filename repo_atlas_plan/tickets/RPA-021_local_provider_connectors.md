# RPA-021 - Integrate Local Model Connectors: Ollama, LM Studio, and a Generic OpenAI-Compatible Local Server

* **Epic**: Providers
* **Priority**: P0
* **Relative Estimate**: M
* **Milestone**: Milestone C
* **Dependencies**: RPA-020

## 1) Goal

Allow the user to run fully local with Ollama, LM Studio, or any OpenAI-compatible local server without needing a cloud provider.

## 2) Problem this ticket solves

Strict Local Mode only truly exists if the local provider connectors work reliably and are easy to configure.

## 3) Detailed scope

* Integrate an Ollama adapter using the default local API.
* Integrate an LM Studio adapter through its local server and model listing.
* Create a generic OpenAI-compatible local adapter to support other runtimes.
* Create UI for test connection, model listing, generation model selection, and embedding model selection if available.
* Support basic capability discovery from config or manually declared capabilities.

## 4) Implementation notes

* Ollama’s default local API does not require authentication when accessed on localhost, so the UX should be extremely simple.[R9]
* LM Studio supports OpenAI-compatible endpoints, so most of the generic adapter logic can be reused.[R8]
* The UI should clearly display base URL, model ID, timeout, and the local privacy badge.

## 5) Breakdown subtasks

* Implement the Ollama adapter.
* Implement the LM Studio adapter.
* Implement the generic local OpenAI-compatible adapter.
* Create the health-check flow and UI form.
* Create sample settings presets: Ollama default, LM Studio default.

## 6) Acceptance criteria

* The user can test a connection to local Ollama.
* The user can test a connection to local LM Studio.
* A model can be selected to generate the structured report.
* Strict Local Mode shows the correct badge and does not require a cloud API key.

## 7) Out of scope

* Model download/management inside the app.
* Automatic model quality benchmarking.

## 8) Risks / watchpoints

* Capability discovery is not consistent across local runtimes.
* Local runtimes may be slow or may not provide a large enough context window, so the UI must warn the user appropriately.

## 9) Expected deliverables

* Ollama adapter
* LM Studio adapter
* Generic local adapter
* Provider setup UI for local mode

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
