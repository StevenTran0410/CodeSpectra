# 07 — Markdown Rendering Spec

Defines how `SectionArtifact.content` is rendered in the report viewer (RPA-043).

---

## Supported elements

| Element | Rendered as | Notes |
|---|---|---|
| `## Heading 2` | Section sub-heading | H1 reserved for section title |
| `### Heading 3` | Sub-section heading | |
| `**bold**` | Bold | |
| `*italic*` | Italic | |
| `` `inline code` `` | Monospace inline | |
| ` ```lang\n...\n``` ` | Syntax-highlighted code block | Uses highlight.js; falls back to plain if language unknown |
| `- item` / `* item` | Unordered list | |
| `1. item` | Ordered list | |
| `> quote` | Blockquote | Used for notable quotes from comments/docs |
| `[text](path)` | Evidence link | Opens evidence drawer if path matches a known file |
| `---` | Horizontal rule | Section separator |

---

## Evidence links

When a Markdown link target matches a pattern like `src/auth.py#L42-L67`, the renderer:
1. Renders it as a styled "evidence chip" (file icon + filename + line range).
2. On click, opens the **Evidence Drawer** with the full `EvidenceItem` context.

Pattern: `relative/path/to/file.ext#L<start>-L<end>`

---

## Code block language tags

Supported highlight aliases:

`python`, `typescript`, `javascript`, `tsx`, `jsx`, `go`, `rust`, `java`, `kotlin`, `swift`, `ruby`, `php`, `c`, `cpp`, `csharp`, `bash`, `sh`, `json`, `yaml`, `toml`, `sql`, `markdown`, `plaintext`

---

## Unsupported / sanitized elements

The following are **stripped** before rendering to prevent XSS or layout issues:

- Raw HTML (`<script>`, `<style>`, `<iframe>`, etc.)
- Images (`![...]()`) — no image assets in reports
- Footnotes
- HTML entities beyond basic `&amp;`, `&lt;`, `&gt;`, `&quot;`

---

## Rendering pipeline

```
SectionArtifact.content (raw Markdown string)
        │
        ▼
  sanitize()          ← strip disallowed elements
        │
        ▼
  parseMarkdown()     ← unified/remark parser
        │
        ▼
  transformLinks()    ← convert file links to evidence chips
        │
        ▼
  React component     ← rendered with Tailwind prose classes
```
