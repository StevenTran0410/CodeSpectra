"""QA Agent — answers codebase questions with evidence-backed citations."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

# Type alias for an async progress callback; receives a dict with type/phase/detail keys.
_ProgressCb = Callable[[dict], Awaitable[None]]

from infrastructure.db.database import get_db
from domain.model_connector.service import ProviderConfigService
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import RetrievalEvidence, RetrievalMode, RetrievalSection

from shared.logger import logger

from domain.analysis.agents._graph_plan import plan_queries, retrieve_multi
from domain.analysis.agents.base import BaseTypedAgent
from domain.analysis.prompts import render_bundle

from .types import Citation, QAResponse, QASectionCache
from .section_cache_service import QASectionCacheService
from .intent_classifier import classify_intent as _local_classify_intent


# Step 1: stream a plain-text markdown answer (no JSON wrapper)
_QA_ANSWER_SYSTEM = """\
You are a senior engineer answering questions about a codebase.

You receive two types of context:
- CODEBASE ANALYSIS: Pre-analyzed summaries produced by a static analysis pipeline. \
Use this directly for high-level questions about architecture, tech stack, features, \
conventions, important files, glossary, and risks.
- CODE EVIDENCE: Raw code chunks retrieved for this specific question. \
Use these for implementation-level details, specific logic, and code citations.

Rules:
- Prefer CODEBASE ANALYSIS for high-level questions — it is authoritative and curated.
- Prefer CODE EVIDENCE for specific implementation questions. Cite with [file:line].
- Synthesize both sources when both are relevant.
- If you cannot determine something from either source, state it clearly.
- Set claim_polarity to 'negative' if your answer states a feature or behaviour does NOT exist or is NOT implemented.

FORMATTING — MANDATORY:
- Write in rich Markdown so it renders beautifully in a chat UI.
- Use ### headings to organize sections.
- Use **bold** for key terms, file names, and function names.
- Use bullet lists (- item) or numbered lists for steps and enumerations.
- Use `backticks` for inline code, symbols, and file paths.
- Use ``` fenced code blocks for multi-line code snippets.
- Use --- horizontal rules to separate major sections.
- Do NOT write a flat wall of text — structure every answer.

Respond with ONLY the markdown answer. No JSON, no preamble, no explanation of format.
"""

# Step 2: fast non-streaming metadata (citations, confidence, deep_research hint)
_QA_META_SYSTEM = """\
You are a code assistant extracting structured metadata from a Q&A answer.
Given the question, the evidence, and the answer already written, output ONLY a JSON object:
{
  "citations": [{"file": "string", "line_start": 0, "line_end": 0, "snippet": "string"}],
  "confidence": "high|medium|low",
  "unknowns": ["string"],
  "suggested_files": ["string"],
  "deep_research_recommended": false
}

deep_research_recommended = true when the question involves tracing call chains,
data flow, execution paths across multiple files, or blast-radius analysis.
Set false for simple lookup questions.

Output ONLY the JSON object. No markdown fences, no prose.
"""

_QA_META_SCHEMA = (
    '{"citations": [{"file": "string", "line_start": 0, "line_end": 0, "snippet": "string"}], '
    '"confidence": "high|medium|low", "unknowns": ["string"], '
    '"suggested_files": ["string"], "deep_research_recommended": false}'
)

_QA_RETRIEVAL_BUDGET = 10_000        # reduced from 12K to make room for analysis sections
_QA_ANALYSIS_BUDGET = 8_000          # max tokens for matched analysis sections
_QA_HINT_INJECTION_BUDGET = 3_000    # reserved within retrieval budget for hint files
_SECTION_MATCH_THRESHOLD = 0.3       # minimum BM25 score to include a section

# Same tokenizer as BM25Scorer / retrieval service — must stay in sync
_WORD = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase-tokenize text using the same regex as BM25Scorer."""
    return [w.lower() for w in _WORD.findall(text) if len(w) > 1]


class QAAgent(BaseTypedAgent):
    """Answers free-form codebase questions using smart-matched analysis context + retrieval + LLM."""

    def __init__(
        self,
        provider_service: ProviderConfigService,
        retrieval_service: RetrievalService,
    ) -> None:
        super().__init__(provider_service)
        self._retrieval = retrieval_service
        self._cache_svc = QASectionCacheService(provider_service)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        provider_id: str,
        model_id: str,
        snapshot_id: str,
        question: str,
        include_debug: bool = False,
        progress_cb: _ProgressCb | None = None,
    ) -> dict[str, Any]:
        t0 = time.monotonic()

        async def _emit(phase: str, detail: str = "") -> None:
            if progress_cb:
                await progress_cb({"type": "status", "phase": phase, "detail": detail})

        # 0. Load section caches (lightweight metadata only)
        section_caches = await self._cache_svc.load_caches(snapshot_id)

        # 1. Match question against section caches → section IDs to load
        matched_section_ids: list[str] = []
        section_scores: dict[str, float] = {}
        if section_caches:
            matched_section_ids, section_scores = self._match_sections(question, section_caches)
        logger.info(
            "[QAAgent] question=%r matched_sections=%s",
            question[:80], matched_section_ids,
        )

        # 2. Load full analysis report only if sections matched
        analysis_context = ""
        hint_files: list[str] = []
        if matched_section_ids:
            analysis_report = await self._load_analysis_report(snapshot_id)
            if analysis_report:
                analysis_context = self._render_matched_sections(
                    analysis_report, matched_section_ids, section_scores, _QA_ANALYSIS_BUDGET
                )
                hint_files = self._extract_hint_files_from_sections(
                    analysis_report, matched_section_ids
                )

        # 3. Decompose question into sub-queries
        await _emit("retrieving", "Searching codebase...")
        sub_queries = await plan_queries(
            goal=question,
            provider_service=self._providers,
            provider_id=provider_id,
            model_id=model_id,
            fallback=[question],
        )
        logger.info("[QAAgent] sub_queries=%s", sub_queries)

        # 4. Retrieve code evidence via two-stage pipeline
        bundle = await retrieve_multi(
            self._retrieval,
            snapshot_id,
            sub_queries,
            RetrievalSection.QA,
            RetrievalMode.HYBRID,
            20,
        )

        # Log retrieval quality
        if bundle.quality:
            logger.info(
                "[QAAgent] retrieval_quality flags=%s label=%s",
                bundle.quality.flags, bundle.quality.quality_label
            )

        # 5. Force-inject important files missed by retrieval
        if hint_files:
            existing_paths = {ev.rel_path for ev in bundle.evidences}
            budget_remaining = max(
                0, _QA_RETRIEVAL_BUDGET - bundle.used_tokens - _QA_HINT_INJECTION_BUDGET
            )
            extra = await self._inject_hint_chunks(
                snapshot_id, hint_files, existing_paths, budget_remaining
            )
            if extra:
                updated_evidences = bundle.evidences + extra
                bundle = bundle.model_copy(update={
                    "evidences": updated_evidences,
                    "used_tokens": bundle.used_tokens + sum(e.token_estimate for e in extra),
                })
                logger.info("[QAAgent] injected %d hint-file chunks", len(extra))

        # 6. Build shared prompt parts
        evidence_block = render_bundle(bundle, limit=25, excerpt_chars=1500)
        prompt_parts = [f"QUESTION: {question}"]
        if analysis_context:
            prompt_parts.append(f"CODEBASE ANALYSIS:\n{analysis_context}")
        prompt_parts.append(f"CODE EVIDENCE:\n{evidence_block}")

        if bundle.quality and bundle.quality.quality_label == "weak":
            flags_str = ", ".join(bundle.quality.flags)
            prompt_parts.append(f"NOTE: Retrieval quality is weak ({flags_str}). If the evidence does not directly answer the question, say so rather than guessing.")

        user_prompt = "\n\n".join(prompt_parts)

        # 7a. Stream the answer as plain markdown (tokens → progress_cb)
        await _emit("thinking", "Analyzing evidence...")

        answer_tokens: list[str] = []

        async def _on_token(tok: str) -> None:
            answer_tokens.append(tok)
            if progress_cb:
                await progress_cb({"type": "token", "text": tok})

        answer_text = await self._call_stream(
            provider_id,
            model_id,
            _QA_ANSWER_SYSTEM,
            user_prompt,
            max_completion_tokens=2000,
            on_token=_on_token,
        )

        # 7b. Fast non-streaming metadata call
        meta_user = (
            f"QUESTION: {question}\n\n"
            f"ANSWER:\n{answer_text}\n\n"
            f"CODE EVIDENCE (abbreviated):\n{evidence_block[:3000]}"
        )
        meta = await self._chat_json_typed(
            provider_id,
            model_id,
            _QA_META_SYSTEM,
            meta_user,
            _QA_META_SCHEMA,
            max_completion_tokens=600,
        )

        # 8. Normalize output — merge streaming answer with metadata
        result: dict = {}
        result["answer"] = answer_text
        result["citations"] = [
            self._normalize_citation(c) for c in (meta.get("citations") or [])
        ]
        result["confidence"] = str(meta.get("confidence", "medium")).lower()
        if result["confidence"] not in ("high", "medium", "low"):
            result["confidence"] = "medium"
        result["unknowns"] = meta.get("unknowns") or []
        result["suggested_files"] = meta.get("suggested_files") or []
        result["deep_research_recommended"] = bool(meta.get("deep_research_recommended", False))
        result["retrieval_quality_label"] = bundle.quality.quality_label if bundle.quality else "strong"

        if include_debug:
            result["retrieval_debug"] = {
                "sub_queries": sub_queries,
                "chunks_used": len(bundle.evidences),
                "hint_files_injected": [
                    e.rel_path for e in bundle.evidences
                    if "analysis-hint-injection" in e.reason_codes
                ],
                "has_analysis_context": bool(analysis_context),
                "matched_sections": matched_section_ids,
                "section_scores": {k: round(v, 4) for k, v in section_scores.items()},
            }
        else:
            result["retrieval_debug"] = None

        ms = int((time.monotonic() - t0) * 1000)
        logger.info("[QAAgent] answered in %dms confidence=%s", ms, result["confidence"])
        return result

    # ------------------------------------------------------------------
    # Section cache: load + match
    # ------------------------------------------------------------------

    async def _load_analysis_report(self, snapshot_id: str) -> dict | None:
        """Load the latest analysis report for a snapshot. Returns None if none exists."""
        db = get_db()
        async with db.execute(
            "SELECT report_json FROM analysis_reports WHERE snapshot_id=? ORDER BY created_at DESC LIMIT 1",
            (snapshot_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["report_json"])
        except Exception:
            logger.warning("[QAAgent] failed to parse analysis report for snapshot=%s", snapshot_id)
            return None

    def _match_sections(
        self,
        question: str,
        caches: list[QASectionCache],
    ) -> tuple[list[str], dict[str, float]]:
        """Score each section cache against the question using BM25 + keyword overlap.

        Algorithm:
          For each section cache, construct a synthetic "document" by concatenating:
            - topic_summary (treated as text content)
            - all answerable_questions joined with spaces (treated as text content)
            - key_terms (treated as the path string for path-hit bonus in BM25Scorer)

          Use an inline BM25 approximation because:
            - The corpus is tiny (max 12 docs) so pre-computed IDF from retrieval_bm25_stats
              does not apply here.
            - We build IDF on-the-fly from the 12 section documents.

          Steps:
          1. Tokenize the question → query_terms (lowercased, len > 1, deduped)
          2. Build per-document token lists from (topic_summary + answerable_questions)
          3. Compute IDF using BM25 formula: log((N - df + 0.5) / (df + 0.5) + 1)
             where N = number of sections, df = sections containing the term
          4. For each section, compute BM25 score against query_terms using avgdl over docs
          5. Add keyword overlap bonus: for each query_term that appears exactly in key_terms,
             add 0.5 bonus (caps at 2.0 total bonus)
          6. Return sections where final_score > _SECTION_MATCH_THRESHOLD,
             sorted by score descending.

        Returns:
          (matched_section_ids ordered by score desc, score_map for all sections)
        """
        import math

        if not caches or not question.strip():
            return [], {}

        # Step 1: tokenize question
        query_terms = _tokenize(question)
        if not query_terms:
            return [], {}
        query_terms_set = set(query_terms)

        # Step 2: tokenize each document (topic_summary + answerable_questions)
        def _doc_text(cache: QASectionCache) -> str:
            return cache.topic_summary + " " + " ".join(cache.answerable_questions)

        doc_tokens: list[list[str]] = [_tokenize(_doc_text(c)) for c in caches]
        N = len(caches)

        # Step 3: compute IDF
        doc_freq: dict[str, int] = {}
        for tokens in doc_tokens:
            for tok in set(tokens):
                doc_freq[tok] = doc_freq.get(tok, 0) + 1

        def idf(term: str) -> float:
            df = doc_freq.get(term, 0)
            if df == 0:
                return 0.0
            return math.log((N - df + 0.5) / (df + 0.5) + 1)

        # BM25 parameters (matching BM25Scorer defaults)
        k1 = 2.0
        b = 0.75
        lengths = [len(toks) for toks in doc_tokens]
        avgdl = sum(lengths) / N if N > 0 else 1.0

        scores: dict[str, float] = {}

        for idx, cache in enumerate(caches):
            tokens = doc_tokens[idx]
            dl = len(tokens)
            if dl == 0:
                scores[cache.section_id] = 0.0
                continue

            # Build TF map
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1

            length_norm = 1 - b + b * dl / (avgdl if avgdl > 0 else 1.0)

            bm25_score = 0.0
            for term in query_terms:
                idf_val = idf(term)
                if idf_val <= 0.0:
                    continue
                f = tf.get(term, 0)
                bm25_score += idf_val * (f * (k1 + 1)) / (f + k1 * length_norm)

            # Step 5: keyword overlap bonus on key_terms
            # Each exact match between a query term and key_terms adds 0.5, capped at 2.0
            key_terms_set = set(t.lower() for t in cache.key_terms)
            overlap_hits = len(query_terms_set & key_terms_set)
            overlap_bonus = min(2.0, overlap_hits * 0.5)

            scores[cache.section_id] = bm25_score + overlap_bonus

        # Step 6: filter by threshold, sort by score desc
        matched = [
            sid for sid, score in scores.items()
            if score > _SECTION_MATCH_THRESHOLD
        ]
        matched.sort(key=lambda sid: scores[sid], reverse=True)

        return matched, scores

    # ------------------------------------------------------------------
    # Render matched sections
    # ------------------------------------------------------------------

    def _render_matched_sections(
        self,
        report: dict,
        section_ids: list[str],
        section_scores: dict[str, float],
        token_budget: int,
    ) -> str:
        """Render matched sections as structured text, score-descending, dropping sections once the token_budget (est. 4 chars/token) is exceeded."""
        sections = report.get("sections", {})
        if not sections:
            return ""

        # Sort section_ids by score descending (already sorted by caller but guard here)
        ordered = sorted(section_ids, key=lambda s: section_scores.get(s, 0.0), reverse=True)

        parts: list[str] = []
        used_chars = 0

        for sid in ordered:
            block = self._render_single_section(sid, sections.get(sid) or {})
            if not block:
                continue
            est_chars = len(block)
            est_tokens = est_chars // 4
            if used_chars // 4 + est_tokens > token_budget:
                logger.debug(
                    "[QAAgent] token budget exceeded at section=%s, dropping remaining", sid
                )
                break
            parts.append(block)
            used_chars += est_chars

        return "\n\n---\n\n".join(parts)

    def _render_single_section(self, section_id: str, data: dict) -> str:
        """Render one section as structured text. Returns empty string if no content."""
        if not data or "error" in data:
            return ""

        sid = section_id.upper()

        if sid == "L":
            lines = []
            if v := data.get("executive_summary"):
                lines.append(f"### Executive Summary\n{v}")
            if v := data.get("architecture_narrative"):
                lines.append(f"### Architecture Narrative\n{v}")
            if v := data.get("tech_stack_snapshot"):
                lines.append(f"### Tech Stack Snapshot\n{v}")
            if v := data.get("conventions_digest"):
                lines.append(f"### Conventions Digest\n{v}")
            if v := data.get("risk_highlights"):
                lines.append(f"### Risk Highlights\n{v}")
            return "\n\n".join(lines)

        if sid == "A":
            lines = []
            if v := data.get("domain"):
                lines.append(f"**Domain:** {v}")
            if v := data.get("purpose"):
                lines.append(f"**Purpose:** {v}")
            if v := data.get("runtime_type"):
                lines.append(f"**Runtime:** {v}")
            if ts := data.get("tech_stack"):
                lines.append(f"**Tech Stack:** {', '.join(ts)}")
            if v := data.get("business_context"):
                lines.append(f"**Business Context:** {v}")
            return "### Project Identity\n" + "\n".join(lines) if lines else ""

        if sid == "B":
            lines = []
            if v := data.get("main_layers"):
                lines.append(f"**Layers:** {', '.join(v)}")
            if v := data.get("frameworks"):
                lines.append(f"**Frameworks:** {', '.join(v)}")
            if v := data.get("entrypoints"):
                lines.append(f"**Entrypoints:** {', '.join(v)}")
            if svcs := data.get("main_services"):
                svc_lines = [
                    f"  - **{s.get('name','?')}** (`{s.get('path','')}`) — {s.get('role','')}"
                    for s in svcs[:10]
                ]
                lines.append("**Main Services:**\n" + "\n".join(svc_lines))
            if v := data.get("external_integrations"):
                lines.append(f"**External Integrations:** {', '.join(v)}")
            if v := data.get("database_hints"):
                lines.append(f"**Databases:** {', '.join(v)}")
            if v := data.get("config_sources"):
                lines.append(f"**Config Sources:** {', '.join(v)}")
            return "### Architecture\n" + "\n".join(lines) if lines else ""

        if sid == "C":
            folders = data.get("folders") or []
            summary = data.get("summary", "")
            if not folders and not summary:
                return ""
            folder_lines = [
                f"  - `{f.get('path','')}` ({f.get('role','')}) — {f.get('description','')}"
                for f in folders[:15]
            ]
            header = f"### Folder Structure\n{summary}\n" if summary else "### Folder Structure\n"
            return header + "\n".join(folder_lines)

        if sid == "D":
            conv_keys = [
                ("naming_style", "Naming Style"),
                ("error_handling", "Error Handling"),
                ("async_style", "Async Style"),
                ("di_style", "DI Style"),
                ("class_vs_functional", "Class vs Functional"),
                ("test_style", "Test Style"),
            ]
            lines = []
            for key, label in conv_keys:
                if aspect := data.get(key):
                    if desc := aspect.get("description"):
                        lines.append(f"  - **{label}:** {desc}")
            if signals := data.get("signals"):
                for sig in signals[:5]:
                    lines.append(
                        f"  - [{sig.get('category','')}] {sig.get('description','')} (evidence: `{sig.get('evidence','')}`)"
                    )
            return "### Conventions\n" + "\n".join(lines) if lines else ""

        if sid == "E":
            e_lines = []
            if rules := data.get("rules"):
                rule_lines = [
                    f"  - [{r.get('severity','?')}] {r.get('rule','')} (inferred from `{r.get('inferred_from','')}`)"
                    for r in rules[:10]
                ]
                e_lines.append("**Rules:**\n" + "\n".join(rule_lines))
            if violations := data.get("violations_found"):
                viol_lines = [
                    f"  - [{v.get('severity','?')}] `{v.get('file','')}` — {v.get('description','')}"
                    for v in violations[:10]
                ]
                e_lines.append("**Violations:**\n" + "\n".join(viol_lines))
            return "### Rules & Violations\n" + "\n".join(e_lines) if e_lines else ""

        if sid == "F":
            features = data.get("features") or []
            if not features:
                return ""
            feat_lines = []
            for feat in features[:12]:
                name = feat.get("name", "")
                desc = feat.get("description", "")
                ep = feat.get("entrypoint", "")
                kf = feat.get("key_files", [])
                feat_lines.append(f"  - **{name}** — {desc}")
                if ep:
                    feat_lines.append(f"    Entrypoint: `{ep}`")
                if kf:
                    feat_lines.append(f"    Key files: {', '.join(f'`{p}`' for p in kf[:5])}")
                if dp := feat.get("data_path"):
                    feat_lines.append(f"    Data path: {dp}")
            return "### Feature Map\n" + "\n".join(feat_lines)

        if sid == "G":
            g_lines = []
            for key, label in [
                ("entrypoint", "Entrypoint"),
                ("backbone", "Backbone"),
                ("critical_config", "Critical Config"),
                ("highest_centrality", "Highest Centrality"),
                ("read_first", "Read First"),
            ]:
                ref = data.get(key, {})
                if isinstance(ref, dict) and ref.get("file"):
                    g_lines.append(f"  - **{label}:** `{ref['file']}` — {ref.get('reason','')}")
            for item in data.get("other_important", [])[:8]:
                if isinstance(item, dict) and item.get("file"):
                    g_lines.append(f"  - `{item['file']}` — {item.get('reason','')}")
            return "### Important Files\n" + "\n".join(g_lines) if g_lines else ""

        if sid == "H":
            steps = data.get("steps") or []
            if not steps:
                return ""
            step_lines = [
                f"  {s.get('order','?')}. `{s.get('file','')}` — {s.get('goal','')} → {s.get('outcome','')}"
                for s in steps[:8]
            ]
            return "### Onboarding Path\n" + "\n".join(step_lines)

        if sid == "I":
            terms = data.get("terms") or []
            if not terms:
                return ""
            term_lines = [
                f"  - **{t.get('term','')}:** {t.get('definition','')}"
                for t in terms[:20]
            ]
            return "### Glossary\n" + "\n".join(term_lines)

        if sid == "J":
            parts = []
            if summary := data.get("summary"):
                parts.append(f"### Risk Summary\n{summary}")
            if findings := data.get("findings"):
                risk_lines = [
                    f"  - [{f.get('severity','?')}] **{f.get('title','')}** ({f.get('category','')}) — {f.get('rationale','')[:150]}"
                    for f in findings[:8]
                ]
                parts.append("### Risk Findings\n" + "\n".join(risk_lines))
            return "\n\n".join(parts)

        # K (audit) and any unknown section: skip (not useful for QA)
        return ""

    # ------------------------------------------------------------------
    # Hint file extraction (section-scoped)
    # ------------------------------------------------------------------

    def _extract_hint_files_from_sections(
        self, report: dict, matched_section_ids: list[str]
    ) -> list[str]:
        """Extract hint file paths from matched sections G and F only."""
        if not report or not matched_section_ids:
            return []
        sections = report.get("sections", {})
        hints: list[str] = []
        matched_set = set(matched_section_ids)

        if "G" in matched_set:
            if g := sections.get("G"):
                for key in ("entrypoint", "backbone", "critical_config", "highest_centrality", "read_first"):
                    ref = g.get(key, {})
                    if isinstance(ref, dict) and (f := ref.get("file")):
                        hints.append(f)
                for item in g.get("other_important", []):
                    if isinstance(item, dict) and (f := item.get("file")):
                        hints.append(f)

        if "F" in matched_set:
            if f_sec := sections.get("F"):
                for feat in f_sec.get("features", []):
                    if ep := feat.get("entrypoint"):
                        hints.append(ep)
                    hints.extend(feat.get("key_files", []))

        seen: set[str] = set()
        out: list[str] = []
        for h in hints:
            if h and h not in seen:
                seen.add(h)
                out.append(h)
        return out

    # ------------------------------------------------------------------
    # Hint chunk injection
    # ------------------------------------------------------------------

    async def _inject_hint_chunks(
        self,
        snapshot_id: str,
        hint_files: list[str],
        existing_paths: set[str],
        budget: int,
    ) -> list[RetrievalEvidence]:
        """Fetch first chunk from each important file not already in retrieved evidence."""
        if budget <= 0:
            return []

        missing = [f for f in hint_files if f not in existing_paths][:10]
        if not missing:
            return []

        db = get_db()
        extra: list[RetrievalEvidence] = []
        used = 0

        for path in missing:
            if used >= budget:
                break
            async with db.execute(
                """SELECT id, rel_path, chunk_index, content, token_estimate
                   FROM retrieval_chunks
                   WHERE snapshot_id=? AND rel_path=?
                   ORDER BY chunk_index ASC LIMIT 1""",
                (snapshot_id, path),
            ) as cur:
                row = await cur.fetchone()
            if not row or not row["content"]:
                continue
            tok = int(row["token_estimate"] or max(1, len(row["content"]) // 4))
            if used + tok > budget:
                continue
            extra.append(RetrievalEvidence(
                chunk_id=row["id"],
                rel_path=row["rel_path"],
                chunk_index=row["chunk_index"],
                reason_codes=["analysis-hint-injection"],
                score=0.0,
                token_estimate=tok,
                excerpt=row["content"],
            ))
            used += tok

        return extra

    # ------------------------------------------------------------------
    # Citation normalization
    # ------------------------------------------------------------------

    @staticmethod
    def classify_intent(question: str) -> bool:
        """Return True if the question warrants deep research.

        Uses a local character n-gram Naive Bayes classifier trained at startup.
        Zero API calls, <1 ms per prediction, works in 30+ languages.
        """
        is_deep, score = _local_classify_intent(question)
        logger.debug("[QAAgent] classify_intent score=%.3f deep=%s", score, is_deep)
        return is_deep

    @staticmethod
    def _normalize_citation(c: Any) -> dict:
        """Normalize a citation to the expected dict format.

        LLM sometimes returns citations as plain strings ("file.ts:10") instead of
        the schema dict. This handles both formats so Pydantic validation never fails.
        """
        if isinstance(c, dict):
            return c
        if isinstance(c, str):
            # Try to parse "path/to/file.ts:line" format
            if ":" in c:
                parts = c.rsplit(":", 1)
                line_part = parts[1].strip() if len(parts) == 2 else ""
                line = int(line_part) if line_part.isdigit() else 0
                return {
                    "file": parts[0].strip(),
                    "line_start": line,
                    "line_end": line,
                    "snippet": "",
                }
            return {"file": c.strip(), "line_start": 0, "line_end": 0, "snippet": ""}
        return {"file": str(c), "line_start": 0, "line_end": 0, "snippet": ""}
