# 08 — Storage Schema and Secrets Design

---

## Database

**Engine:** SQLite via `aiosqlite` (async), WAL mode enabled, `foreign_keys = ON`.

**Location:** `$CODESPECTRA_DATA_DIR/codespectra.db` (env var set by Electron main on launch; defaults to Electron `userData` path).

---

## Schema migrations

Migrations are applied sequentially at startup. Each migration is append-only — existing migrations are never modified.

| Version | Description |
|---|---|
| 0 | `app_metadata` + `workspaces` tables |
| 1 | `provider_configs` table |
| 2 | `local_repos` table |
| 3 | `ALTER TABLE local_repos ADD COLUMN selected_branch` |

---

## Tables

### `app_metadata`
Key-value store for app-level flags.

| Column | Type | Notes |
|---|---|---|
| `key` | TEXT PK | e.g. `schema_version`, `cloud_consent_given`, `first_launched_at` |
| `value` | TEXT | |

### `workspaces`

| Column | Type |
|---|---|
| `id` | TEXT PK (UUID) |
| `name` | TEXT UNIQUE NOT NULL |
| `created_at` | TEXT (ISO timestamp) |
| `updated_at` | TEXT |
| `settings` | TEXT (JSON, default `{}`) |

### `provider_configs`

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK |
| `kind` | TEXT | `ollama` / `lmstudio` / `openai` / `anthropic` / `gemini` / `deepseek` |
| `display_name` | TEXT UNIQUE NOT NULL | |
| `base_url` | TEXT NOT NULL | |
| `model_id` | TEXT NOT NULL | |
| `capabilities` | TEXT (JSON) | `{streaming, embeddings, max_context_tokens, supports_system_prompt}` |
| `extra` | TEXT (JSON) | Contains `api_key` for cloud providers — **never returned in API responses** |
| `created_at` | TEXT | |
| `updated_at` | TEXT | |

### `local_repos`

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK |
| `path` | TEXT UNIQUE NOT NULL | Absolute OS path |
| `name` | TEXT NOT NULL | Folder basename |
| `source_type` | TEXT | `local_folder` (future: `github`, `bitbucket`) |
| `is_git_repo` | INTEGER | 0 / 1 |
| `git_branch` | TEXT | HEAD branch at last validation |
| `git_head_hash` | TEXT | 12-char short hash |
| `git_remote_url` | TEXT | `origin` remote URL if present |
| `has_size_warning` | INTEGER | 0 / 1 |
| `selected_branch` | TEXT | User-chosen analysis branch (NULL = use HEAD) |
| `added_at` | TEXT | |
| `last_validated_at` | TEXT | |

---

## Secrets handling

### v1 approach (current)

Cloud provider API keys are stored in `provider_configs.extra` as `{"api_key": "sk-..."}`.

**Protections in place:**
- The `extra` field is never returned in API responses — only `has_api_key: true` is exposed.
- Keys are never written to any log file.
- The Python backend reads the key only when making a live API call; it is not held in memory between requests.

### v2 target (RPA-051)

Migrate to OS keychain via Electron's `safeStorage` API:
- Store `api_key` in `safeStorage.encryptString()` and persist the encrypted blob.
- Reference the stored key by a `provider_id`-keyed secret name in the system keychain.
- Remove `api_key` from the SQLite `extra` field entirely.

---

## Data directory layout

```
$CODESPECTRA_DATA_DIR/
├── codespectra.db          ← main SQLite database
├── logs/
│   └── codespectra.log     ← rotating log file (backend)
└── repos/                  ← (future) cloned repository snapshots
    └── <repo_id>/
        └── HEAD/
```
