# 01 — Product Contract

_Frozen after Sprint 1. Changes require explicit team sign-off._

---

## 1. Problem statement

Developers entering a large, unfamiliar codebase — or returning to one they once understood — lack a fast, trustworthy way to orient themselves. Existing tools either require writing code to explore (AI editors) or produce shallow summaries that miss structure and risk. Repo CodeSpectra fills this gap by producing an **evidence-backed onboarding report** without requiring the user to write a single prompt.

---

## 2. Core user story

> As a developer assigned to an unfamiliar repository, I want a structured report that explains what the system does, how it is organized, which files matter most, and where the risks are — so that I can be productive within hours instead of days.

---

## 3. Privacy modes

### 3.1 Strict Local Mode
- All LLM inference runs on the user's machine via Ollama or LM Studio.
- No code, symbol, or text leaves the device.
- No network calls except to localhost.
- Default mode. Shown with a green shield badge in the UI.

### 3.2 BYOK Cloud Mode (Bring Your Own Key)
- User supplies an API key for a supported cloud provider (OpenAI, Anthropic, Google Gemini, DeepSeek).
- Code context is sent to that provider's API during analysis.
- Explicit one-time consent required before any cloud provider can be used.
- API key stored in local SQLite DB, never logged, never returned in API responses (only `has_api_key: true` flag exposed).
- Shown with an amber cloud badge in the UI.

### 3.3 Non-negotiable privacy rules
1. No telemetry, analytics, or crash reporting unless the user opts in explicitly.
2. API keys are stored locally and never transmitted except to their own provider.
3. Cloud mode is opt-in at the provider level — adding a cloud provider requires consent, running analysis with it requires a separate confirmation.

---

## 4. Supported repository sources

| Source | Auth required | Notes |
|---|---|---|
| Local folder | None | git metadata read via subprocess if `.git` present; still works without it |
| GitHub | OAuth / App | RPA-023 |
| Bitbucket | OAuth 2.0 / API token | RPA-024 |

---

## 5. Analysis scope (v1)

The first release produces:

1. **Project Identity Card** — name, language breakdown, primary purpose
2. **Architecture Overview** — top-level structure, boundaries, layers
3. **Onboarding Digest** — where to start reading, key entry points
4. **Functionality-to-File Map** — feature clusters mapped to files
5. **Important Files Radar** — files with disproportionate impact
6. **Glossary** — domain terms extracted from code and docs
7. **Convention Analysis** — naming and structural patterns; anti-pattern flags
8. **Risk / Complexity / Hotspot** — complexity scores, circular imports, test gaps

All claims in the report are backed by evidence references (file + line range + excerpt).

---

## 6. What it is not

- Not an AI code editor
- Not a chat assistant or REPL
- Not a CI/CD tool
- Not a live file-system watcher
- Not a multi-repo dependency manager

---

## 7. Success criteria for v1

- A developer unfamiliar with a ~50k-line repository can read the report and identify the correct entry point within 10 minutes.
- The report contains no hallucinated file paths or function names — every claim references a real location.
- Strict Local Mode produces a complete report without any outbound network call.
- The app runs on macOS and Windows without requiring developer tooling.
