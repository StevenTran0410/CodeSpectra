# RPA-022 - Integrate Cloud Providers: OpenAI, Anthropic, Gemini, and DeepSeek

* **Epic**: Providers
* **Priority**: P1
* **Relative Estimate**: L
* **Milestone**: Milestone C
* **Dependencies**: RPA-020, RPA-011

## 1) Goal

Support BYOK cloud mode for the four primary providers required by the original product idea, while still preserving clear privacy warnings and a clean abstraction layer.

## 2) Problem this ticket solves

Cloud support is a real user need, but if it is added too quickly it will create confusion in secret handling, privacy messaging, and the provider adapter layer.

## 3) Detailed scope

* Implement the OpenAI adapter.
* Implement the Anthropic adapter.
* Implement the Gemini adapter.
* Implement the DeepSeek adapter, preferably through an OpenAI-compatible path where appropriate.
* Allow credential validation and manual model listing/testing if the provider does not offer reliable model listing.
* Display the privacy warning: selected context may leave device.

## 4) Implementation notes

* OpenAI and DeepSeek are good candidates for the generic adapter because they are compatible with OpenAI-style APIs for certain endpoints/base URL patterns.
* Anthropic and Gemini require dedicated adapters because their authentication, headers, and request body shapes differ.
* The cloud mode UI must require explicit consent before the first analysis run.

## 5) Breakdown subtasks

* Implement provider-specific authentication and header configuration.
* Implement normalized structured generation for each provider.
* Implement the consent banner and run-time privacy badge.
* Implement settings and credential validation UI.
* Write integration smoke tests with mocked endpoints.

## 6) Acceptance criteria

* The app can store and test credentials for all 4 provider types.
* Cloud mode shows a clear warning before the first run.
* The generation code does not require scattered provider-specific branching in the UI layer.
* Auth, quota, and network errors are mapped into understandable error groups.

## 7) Out of scope

* Advanced tool use / function calling features specific to each vendor.
* Detailed cost analytics.

## 8) Risks / watchpoints

* Differences in structured output behavior across providers may make JSON generation unstable.
* Users may misunderstand the privacy mode if the warning is too weak.

## 9) Expected deliverables

* 4 cloud adapters
* Consent flow
* Cloud provider settings UI
* Mock integration tests

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
