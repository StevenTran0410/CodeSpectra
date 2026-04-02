# 04 — Privacy Wording

Canonical strings used in UI, consent flows, and documentation.

---

## Badges

| Context | Text | Color |
|---|---|---|
| Local provider card | `Strict Local` | Emerald green |
| Local mode during run | `No data leaves this device` | Emerald green |
| Cloud provider card | `BYOK Cloud` | Amber |
| Cloud mode during run | `Code sent to [Provider Name]` | Amber |

---

## Consent banner (cloud provider — first add)

**Title:** Cloud Provider Notice

**Body:**
> When using a cloud provider, your **code and repository context** will be sent to external AI services over the internet.
>
> **What this means:**
> - Source code snippets will leave your device
> - Data is processed under each provider's privacy policy
> - API keys are stored in this app's local database
> - No data is sent until you start an analysis run
>
> **Do not use cloud providers** with confidential, proprietary, or NDA-covered code. Use **Strict Local mode** (Ollama / LM Studio) for sensitive repositories.

**Primary action:** `I understand — proceed`
**Secondary action:** `Cancel`

---

## No-.git folder warning

> No `.git` folder found — this folder will be indexed without commit history. Branch selection and snapshot pinning are unavailable.

Shown as amber inline notice. Does **not** block the import.

---

## Size warning

> **Large folder:** [reason]. Initial scan may take longer than usual.

Shown as soft amber notice. Does **not** block the import.

---

## Analysis run — cloud mode confirmation (per run)

> **Heads up:** This analysis will send code context to **[Provider display name]**.
> Make sure your repository does not contain confidential or proprietary information.

**Primary action:** `Run analysis`
**Secondary action:** `Cancel`

---

## Log / error sanitization rules

1. API keys MUST NOT appear in any log output.
2. File paths MAY appear in logs (they are local paths, not secret).
3. Error messages sent to the frontend MUST NOT include raw stack traces in production builds.
4. Provider error messages from external APIs MAY be surfaced to the user after sanitization (remove any reflected key fragments).
