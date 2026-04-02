# 02 — Domain Model

---

## Core entities

```
Workspace
├── id: UUID
├── name: string
├── created_at / updated_at: ISO timestamp
└── settings: JSON

LocalRepo  (source_type = "local_folder" | "github" | "bitbucket")
├── id: UUID
├── path: string (absolute, on-disk)
├── name: string (folder name)
├── is_git_repo: bool
├── git_branch: string | null          ← actual HEAD at last validation
├── git_head_hash: string | null       ← short hash (12 chars)
├── git_remote_url: string | null
├── selected_branch: string | null     ← user-chosen analysis branch
├── has_size_warning: bool
├── added_at / last_validated_at: ISO timestamp
└── source_type: RepoSourceType

ProviderConfig  (kind = "ollama" | "lmstudio" | "openai" | "anthropic" | "gemini" | "deepseek")
├── id: UUID
├── kind: ProviderKind
├── display_name: string
├── base_url: string
├── model_id: string
├── capabilities: ProviderCapabilities
├── extra: JSON  ← api_key stored here for cloud providers (never returned in API)
├── created_at / updated_at: ISO timestamp
└── [extra.has_api_key: bool in responses]

AnalysisJob  (future: RPA-012 / RPA-040+)
├── id: UUID
├── repo_id: UUID → LocalRepo
├── provider_id: UUID → ProviderConfig
├── status: "pending" | "running" | "done" | "failed" | "cancelled"
├── scan_mode: "quick" | "full"
├── privacy_mode: "strict_local" | "byok_cloud"
├── created_at / started_at / finished_at: ISO timestamp
└── result_ref: UUID → ReportArtifact | null

ReportArtifact  (future: RPA-002 schema)
├── id: UUID
├── job_id: UUID → AnalysisJob
├── schema_version: string
├── sections: SectionArtifact[]
└── generated_at: ISO timestamp

SectionArtifact
├── section_id: string  (e.g. "architecture_overview")
├── title: string
├── content: string (Markdown)
├── evidence: EvidenceItem[]
└── confidence: ConfidenceModel

EvidenceItem
├── file_path: string
├── line_start / line_end: int
├── excerpt: string
└── relevance_note: string

ConfidenceModel
├── score: float  (0.0–1.0)
├── basis: "symbol_count" | "import_graph" | "llm_generated" | "heuristic"
└── caveats: string[]
```

---

## Aggregate boundaries

| Aggregate | Owns | Does NOT own |
|---|---|---|
| Workspace | its own metadata | repos, providers, jobs |
| LocalRepo | path, git metadata, selected_branch | workspace assignment (future) |
| ProviderConfig | connection config, api_key (internal) | which jobs used it |
| AnalysisJob | lifecycle, scan config | file content, report body |
| ReportArtifact | all section content, evidence | job lifecycle |

---

## Invariants

1. A `ProviderConfig` of a cloud kind MUST have a non-empty `api_key` in `extra` before `test_connection` can succeed.
2. A `LocalRepo` with `is_git_repo = false` cannot have a `selected_branch`.
3. `selected_branch`, when set, MUST exist in the local branch list at the time of setting.
4. A `ReportArtifact` MUST reference a completed `AnalysisJob`.
5. All `EvidenceItem.file_path` values MUST be paths that exist (or existed at snapshot time) in the indexed repo.
