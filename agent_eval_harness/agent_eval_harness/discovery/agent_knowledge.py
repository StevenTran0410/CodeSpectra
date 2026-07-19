"""Agent knowledge model: semantic profile enriched from source code and LLM analysis."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_eval_harness.planning.contract import OutputContract

logger = logging.getLogger("agent_eval_harness.discovery.agent_knowledge")

_MD_TITLE_TRUNCATE_LEN = 50


class Citation(BaseModel):
    """A reference to a location in source code. file/line optional — the LLM doesn't
    always have one to cite; verify_citations treats a missing one as unclaimed, not phantom."""
    file: str | None = None
    line: int | None = None
    symbol: str = ''

    @field_validator('file')
    @classmethod
    def validate_file(cls, v: str | None) -> str | None:
        if v == '':
            raise ValueError('file cannot be empty')
        return v

    @field_validator('line')
    @classmethod
    def validate_line(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError('line cannot be negative')
        return v


class LocationInfo(BaseModel):
    """Where the agent's code entry point is located."""
    file: str
    line_start: int
    line_end: int
    entry_method: str
    entry_line: int


class ComponentRef(BaseModel):
    """Reference to a component this agent owns."""
    id: str
    role: str
    file: str
    line: int


class ContractArg(BaseModel):
    """Input parameter to this agent."""
    kwarg: str
    source_kind: str = ''
    type_hint: str = ''
    example: str = ''


class PromptSiteRef(BaseModel):
    """Location of a prompt or prompt-related constant."""
    file: str
    line: int
    kind: str
    snippet: str


class ComponentRoleVerdict(BaseModel):
    """One component's role verdict from Stage 2.5 enrichment, post structural hard-gate.
    Raw LLM-derived output — NEVER read for gating; see Component.role on the persisted
    system_map, which is the authoritative value this same verdict was written onto."""
    id: str
    role: str = 'unknown'
    confidence: float = 0.0
    reasoning: str = ''

    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        """A malformed cached sidecar must never crash the cache path — an empty/falsy
        role coerces to 'unknown' rather than raising."""
        return v if v else 'unknown'


class ContextBuilderRef(BaseModel):
    """A helper that builds context for this agent. file/line optional — see Citation."""
    name: str
    file: str | None = None
    line: int | None = None
    builds_kwarg: str = ''


class ConsumerRef(BaseModel):
    """A downstream service that consumes output from this agent. file/line optional —
    see Citation."""
    name: str
    file: str | None = None
    line: int | None = None


class FailureModeRef(BaseModel):
    """A documented failure mode of this agent. file/line optional — see Citation."""
    description: str
    file: str | None = None
    line: int | None = None


class AgentKnowledge(BaseModel):
    """Enriched semantic profile of an agent, combining static harvest with LLM analysis."""
    model_config = ConfigDict(extra='ignore')

    # Structural fields (sourced from static harvest, LLM never overrides)
    location: LocationInfo | None = None
    components: list[ComponentRef] = Field(default_factory=list)
    input_contract: list[ContractArg] = Field(default_factory=list)
    output_contract: OutputContract | None = None
    prompt_sites: list[PromptSiteRef] = Field(default_factory=list)

    # Role verdicts — first conclusion of the LLM round, post hard-gate. NEVER authoritative;
    # the same gated values are written onto Component.role on the system_map.
    component_roles: list[ComponentRoleVerdict] = Field(default_factory=list)

    # Semantic fields (LLM-derived, all with citations)
    functionality: str = ''
    functionality_citations: list[Citation] = Field(default_factory=list)
    context_builders: list[ContextBuilderRef] = Field(default_factory=list)
    upstream_consumers: list[ConsumerRef] = Field(default_factory=list)
    downstream_consumers: list[ConsumerRef] = Field(default_factory=list)
    failure_modes: list[FailureModeRef] = Field(default_factory=list)
    output_described_in_prompt: str = ''
    special_traits: list = Field(default_factory=list)
    constraints: list = Field(default_factory=list)  # hard rules the prompt imposes → Stage 3 gate fuel
    method_steps: list = Field(default_factory=list)  # ordered procedure the prompt tells the agent to follow

    # Flags and metadata
    degraded: bool = False
    confidence: Literal['low', 'medium', 'high'] = 'low'
    degraded_reason: str | None = None
    needs_human: list[str] = Field(default_factory=list)
    evidence_hash: str = ''
    query_count: int = 0
    generated_at: str = ''

    def to_md(self) -> str:
        """Human-readable markdown dossier."""
        lines = []
        lines.append(f"# Agent Knowledge: {self.functionality[:_MD_TITLE_TRUNCATE_LEN]}")

        if self.degraded:
            lines.append(f"⚠️ **Degraded**: {self.degraded_reason}")

        lines.append(f"**Confidence**: {self.confidence}")
        lines.append(f"**Query Count**: {self.query_count}")

        if self.location:
            lines.append("## Location")
            lines.append(f"- File: {self.location.file}:{self.location.line_start}-{self.location.line_end}")
            lines.append(f"- Entry: {self.location.entry_method}:{self.location.entry_line}")

        if self.components:
            lines.append("## Components")
            for c in self.components:
                lines.append(f"- {c.id} ({c.role}): {c.file}:{c.line}")

        if self.output_contract and self.output_contract.json_schema:
            lines.append("## Output Contract")
            if self.output_contract.schema_source:
                lines.append(f"*Source: {self.output_contract.schema_source}*")
            lines.append("```")
            lines.append(json.dumps(self.output_contract.json_schema, indent=2))
            lines.append("```")

        if self.component_roles:
            lines.append("## Role (non-authoritative — see system_map for the persisted value)")
            for r in self.component_roles:
                lines.append(f"- {r.id}: {r.role} (confidence={r.confidence:.2f}) — {r.reasoning}")

        if self.functionality:
            lines.append("## Functionality")
            lines.append(self.functionality)

        if self.context_builders:
            lines.append("## Context Builders")
            for cb in self.context_builders:
                loc = f" ({cb.file}:{cb.line})" if cb.file and cb.line else ""
                lines.append(f"- {cb.name} → builds `{cb.builds_kwarg}`{loc}")

        if self.upstream_consumers:
            lines.append("## Upstream Consumers")
            for consumer in self.upstream_consumers:
                loc = f" ({consumer.file}:{consumer.line})" if consumer.file and consumer.line else ""
                lines.append(f"- {consumer.name}{loc}")

        if self.downstream_consumers:
            lines.append("## Downstream Consumers")
            for consumer in self.downstream_consumers:
                loc = f" ({consumer.file}:{consumer.line})" if consumer.file and consumer.line else ""
                lines.append(f"- {consumer.name}{loc}")

        if self.failure_modes:
            lines.append("## Known Failure Modes")
            for mode in self.failure_modes:
                loc = f" ({mode.file}:{mode.line})" if mode.file and mode.line else ""
                lines.append(f"- {mode.description}{loc}")

        if self.output_described_in_prompt:
            lines.append("## Output (described in prompt)")
            lines.append(self.output_described_in_prompt)

        if self.method_steps:
            lines.append("## Method Steps")
            for i, step in enumerate(self.method_steps, 1):
                lines.append(f"{i}. {step}")

        if self.constraints:
            lines.append("## Constraints (from prompt)")
            for c in self.constraints:
                lines.append(f"- {c}")

        if self.special_traits:
            lines.append("## Special Traits")
            for t in self.special_traits:
                lines.append(f"- {t}")

        if self.functionality_citations:
            lines.append("## Functionality Citations")
            for cit in self.functionality_citations:
                if cit.file and cit.line:
                    lines.append(f"- {cit.file}:{cit.line} ({cit.symbol})")

        if self.needs_human:
            lines.append("## Needs Human Review")
            for item in self.needs_human:
                lines.append(f"- {item}")

        return "\n".join(lines)

    def to_json(self) -> dict:
        """Serializable dictionary representation."""
        return json.loads(self.model_dump_json())

    @classmethod
    def from_json(cls, data: dict) -> AgentKnowledge:
        """Deserialize from JSON dict, degrading gracefully on validation error."""
        try:
            return cls.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to validate AgentKnowledge: {e}")
            return cls(
                degraded=True,
                confidence='low',
                degraded_reason=f"Validation error: {str(e)}"
            )


class ClaimVerification(BaseModel):
    """Verification result for a single semantic claim."""
    status: Literal['verified', 'unverified', 'phantom']
    citation: Citation
    reason: str


class VerificationReport(BaseModel):
    """Results of verifying all citations in an AgentKnowledge."""
    agent_id: str
    claims: list[ClaimVerification] = Field(default_factory=list)


_CITATION_LINE_TOLERANCE = 3  # a cite just above a symbol (decorator/blank line) still resolves to it


def verify_citations(
    knowledge: AgentKnowledge,
    repo_root: Path,
    symbols_by_file: dict[str, list[dict]] | None = None,
) -> VerificationReport:
    """Verify semantic-field citations against the real repo (structural fields are
    static-wins, never re-verified). A citation is valid if it can be resolved to
    readable context — the symbol text is on the line, OR the line falls inside/just
    above an indexed symbol span (downstream reads the whole function/class anyway), OR
    it coincides with a detected prompt site. Only citations that resolve to nothing are
    flagged. symbols_by_file (from code_symbols) enables the span check; absent it, only
    the line-text and prompt-site checks run."""
    report = VerificationReport(agent_id='', claims=[])
    symbols_by_file = symbols_by_file or {}
    prompt_site_locs = {
        (p.file, p.line) for p in knowledge.prompt_sites if p.file and p.line
    }

    def _resolves_to_symbol(file: str, line: int) -> bool:
        for sym in symbols_by_file.get(file) or []:
            ls, le = sym.get('line_start'), sym.get('line_end')
            if ls and le and ls <= line <= le:
                return True
            if ls and 0 <= ls - line <= _CITATION_LINE_TOLERANCE:
                return True
        return False

    semantic_citations: list[tuple[str, int, str]] = []

    # Sources missing file/line are skipped (unclaimed), not flagged as phantom.
    for named_source in (
        knowledge.context_builders, knowledge.upstream_consumers, knowledge.downstream_consumers
    ):
        for item in named_source:
            if item.file and item.line and item.line > 0:
                semantic_citations.append((item.file, item.line, item.name))

    for mode in knowledge.failure_modes:
        if mode.file and mode.line and mode.line > 0:
            semantic_citations.append((mode.file, mode.line, ''))

    for cit in knowledge.functionality_citations:
        if cit.file and cit.line and cit.line > 0:
            semantic_citations.append((cit.file, cit.line, cit.symbol))

    for file, line, symbol in semantic_citations:
        file_path = repo_root / file

        if not file_path.exists():
            claim = ClaimVerification(
                status='phantom',
                citation=Citation(file=file, line=line, symbol=symbol),
                reason=f"File does not exist: {file}"
            )
            report.claims.append(claim)
            knowledge.needs_human.append(f"Phantom citation: {file}:{line}:{symbol}")
            continue

        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')

            # 1-indexed line numbers
            if line < 1 or line > len(lines):
                claim = ClaimVerification(
                    status='unverified',
                    citation=Citation(file=file, line=line, symbol=symbol),
                    reason=f"Line {line} out of range (file has {len(lines)} lines)"
                )
                report.claims.append(claim)
                knowledge.needs_human.append(f"Unverified citation: {file}:{line}:{symbol}")
                continue

            actual_line = lines[line - 1]
            # A citation resolves to readable context if the symbol text is on the line, OR the line
            # falls inside/just above an indexed symbol span, OR it lands on a detected prompt site.
            resolved = (
                (not symbol or symbol in actual_line)
                or _resolves_to_symbol(file, line)
                or (file, line) in prompt_site_locs
            )
            if not resolved:
                claim = ClaimVerification(
                    status='unverified',
                    citation=Citation(file=file, line=line, symbol=symbol),
                    reason=f"Line {line} resolves to no symbol span or prompt site"
                )
                report.claims.append(claim)
                knowledge.needs_human.append(f"Unverified citation: {file}:{line}:{symbol}")
                continue

            claim = ClaimVerification(
                status='verified',
                citation=Citation(file=file, line=line, symbol=symbol),
                reason=f"Symbol found on line {line}"
            )
            report.claims.append(claim)

        except Exception as e:
            logger.debug(f"Citation verification could not read {file}:{line}: {e}")
            claim = ClaimVerification(
                status='unverified',
                citation=Citation(file=file, line=line, symbol=symbol),
                reason=f"Error reading file: {str(e)}"
            )
            report.claims.append(claim)
            knowledge.needs_human.append(f"Unverified citation: {file}:{line}:{symbol}")

    return report
