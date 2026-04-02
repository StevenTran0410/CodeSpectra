# 06 — Report Schema / Evidence Model / Confidence Model

---

## Top-level: `ReportArtifact`

```json
{
  "schema_version": "1.0.0",
  "id": "<uuid>",
  "job_id": "<uuid>",
  "repo_name": "my-service",
  "repo_path": "/home/user/projects/my-service",
  "generated_at": "2026-04-02T10:00:00Z",
  "scan_mode": "full",
  "privacy_mode": "strict_local",
  "provider_kind": "ollama",
  "model_id": "llama3.2:8b",
  "sections": [ /* SectionArtifact[] */ ]
}
```

---

## `SectionArtifact`

```json
{
  "section_id": "architecture_overview",
  "title": "Architecture Overview",
  "content": "## Architecture Overview\n\nThis service follows a layered architecture...",
  "evidence": [ /* EvidenceItem[] */ ],
  "confidence": { /* ConfidenceModel */ },
  "generated_at": "2026-04-02T10:01:23Z"
}
```

### Defined section IDs

| `section_id` | Title | Ticket |
|---|---|---|
| `project_identity` | Project Identity Card | RPA-040 |
| `architecture_overview` | Architecture Overview | RPA-040 |
| `onboarding_digest` | Onboarding Digest | RPA-040 |
| `feature_map` | Functionality-to-File Map | RPA-042 |
| `important_files` | Important Files Radar | RPA-042 |
| `glossary` | Glossary | RPA-042 |
| `conventions` | Convention & Style Analysis | RPA-041 |
| `anti_patterns` | Anti-Pattern Discovery | RPA-041 |
| `risk_hotspots` | Risk / Complexity / Hotspot | RPA-044 |

---

## `EvidenceItem`

```json
{
  "file_path": "src/api/auth.py",
  "line_start": 42,
  "line_end": 67,
  "excerpt": "class AuthService:\n    def verify_token(self, token: str) -> User:",
  "relevance_note": "Primary authentication entry point — called by all protected routes"
}
```

**Rules:**
- `file_path` MUST be a path that exists (or existed at snapshot time) in the indexed repo.
- `excerpt` MUST be verbatim — no paraphrasing.
- `relevance_note` is LLM-generated but sanitized.

---

## `ConfidenceModel`

```json
{
  "score": 0.87,
  "basis": "symbol_count",
  "caveats": [
    "Import graph coverage: 94% (6% files had parse errors)"
  ]
}
```

| `basis` | Meaning |
|---|---|
| `symbol_count` | Derived from parsed symbol index — high reliability |
| `import_graph` | Derived from import analysis — medium-high reliability |
| `llm_generated` | LLM output without structural verification — medium reliability |
| `heuristic` | Rule-based without LLM — varies |

**Score interpretation:**

| Range | Label |
|---|---|
| 0.85 – 1.0 | High |
| 0.65 – 0.84 | Medium |
| 0.0 – 0.64 | Low — shown with caveat notice in UI |

---

## Versioning

- `schema_version` follows semver.
- Minor bumps: backward compatible (new optional fields).
- Major bumps: require a migration step in the report viewer.
- Old reports stored in DB are never retroactively upgraded — viewer must handle all known versions.
