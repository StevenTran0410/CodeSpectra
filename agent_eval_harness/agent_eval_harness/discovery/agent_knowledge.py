"""Agent knowledge model: semantic profile enriched from source code and LLM analysis."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger("agent_eval_harness.discovery.agent_knowledge")


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
    source_kind: str
    type_hint: str
    example: str


class OutputContract(BaseModel):
    """Output specification for this agent."""
    json_schema: str
    example: str


class PromptSiteRef(BaseModel):
    """Location of a prompt or prompt-related constant."""
    file: str
    line: int
    kind: str
    snippet: str


class ComponentRoleVerdict(BaseModel):
    """One component's role verdict from Stage 2.5 (CS-300), post structural hard-gate.
    Raw LLM-derived output — NEVER read for gating; see Component.role on the persisted
    system_map, which is the authoritative value this same verdict was written onto."""
    id: str
    role: str = 'unknown'
    confidence: float = 0.0
    reasoning: str = ''

    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        """A pre-CS-300 or otherwise malformed cached sidecar must never crash the cache
        path — an empty/falsy role coerces to 'unknown' rather than raising."""
        return v if v else 'unknown'


class ContextBuilderRef(BaseModel):
    """A helper that builds context for this agent. file/line optional — see Citation."""
    name: str
    file: str | None = None
    line: int | None = None
    builds_kwarg: str = ''


class ConsumerRef(BaseModel):
    """A downstream service that consumes output from this agent. file/line optional — see Citation."""
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

    # Role verdicts (CS-300) — first conclusion of the LLM round, post hard-gate. NEVER
    # authoritative; the same gated values are written onto Component.role on the system_map.
    component_roles: list[ComponentRoleVerdict] = Field(default_factory=list)

    # Semantic fields (LLM-derived, all with citations)
    functionality: str = ''
    functionality_citations: list[Citation] = Field(default_factory=list)
    context_builders: list[ContextBuilderRef] = Field(default_factory=list)
    upstream_consumers: list[ConsumerRef] = Field(default_factory=list)
    downstream_consumers: list[ConsumerRef] = Field(default_factory=list)
    failure_modes: list[FailureModeRef] = Field(default_factory=list)

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
        lines.append(f"# Agent Knowledge: {self.functionality[:50]}")

        if self.degraded:
            lines.append(f"⚠️ **Degraded**: {self.degraded_reason}")

        lines.append(f"**Confidence**: {self.confidence}")
        lines.append(f"**Query Count**: {self.query_count}")

        if self.location:
            lines.append(f"## Location")
            lines.append(f"- File: {self.location.file}:{self.location.line_start}-{self.location.line_end}")
            lines.append(f"- Entry: {self.location.entry_method}:{self.location.entry_line}")

        if self.components:
            lines.append(f"## Components")
            for c in self.components:
                lines.append(f"- {c.id} ({c.role}): {c.file}:{c.line}")

        if self.component_roles:
            lines.append(f"## Role (non-authoritative — see system_map for the persisted value)")
            for r in self.component_roles:
                lines.append(f"- {r.id}: {r.role} (confidence={r.confidence:.2f}) — {r.reasoning}")

        if self.functionality:
            lines.append(f"## Functionality")
            lines.append(self.functionality)

        if self.context_builders:
            lines.append(f"## Context Builders")
            for cb in self.context_builders:
                loc = f" ({cb.file}:{cb.line})" if cb.file and cb.line else ""
                lines.append(f"- {cb.name} → builds `{cb.builds_kwarg}`{loc}")

        if self.upstream_consumers:
            lines.append(f"## Upstream Consumers")
            for consumer in self.upstream_consumers:
                loc = f" ({consumer.file}:{consumer.line})" if consumer.file and consumer.line else ""
                lines.append(f"- {consumer.name}{loc}")

        if self.downstream_consumers:
            lines.append(f"## Downstream Consumers")
            for consumer in self.downstream_consumers:
                loc = f" ({consumer.file}:{consumer.line})" if consumer.file and consumer.line else ""
                lines.append(f"- {consumer.name}{loc}")

        if self.failure_modes:
            lines.append(f"## Known Failure Modes")
            for mode in self.failure_modes:
                loc = f" ({mode.file}:{mode.line})" if mode.file and mode.line else ""
                lines.append(f"- {mode.description}{loc}")

        if self.functionality_citations:
            lines.append(f"## Functionality Citations")
            for cit in self.functionality_citations:
                if cit.file and cit.line:
                    lines.append(f"- {cit.file}:{cit.line} ({cit.symbol})")

        if self.needs_human:
            lines.append(f"## Needs Human Review")
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


def verify_citations(knowledge: AgentKnowledge, repo_root: Path) -> VerificationReport:
    """Verify semantic-field citations against the real repo (structural fields are
    static-wins, never re-verified). Invalid citations are added to needs_human."""
    report = VerificationReport(agent_id='', claims=[])
    semantic_citations: list[tuple[str, int, str]] = []

    # Sources missing file/line are skipped (unclaimed), not flagged as phantom.
    for builder in knowledge.context_builders:
        if builder.file and builder.line and builder.line > 0:
            semantic_citations.append((builder.file, builder.line, builder.name))

    for consumer in knowledge.upstream_consumers:
        if consumer.file and consumer.line and consumer.line > 0:
            semantic_citations.append((consumer.file, consumer.line, consumer.name))

    for consumer in knowledge.downstream_consumers:
        if consumer.file and consumer.line and consumer.line > 0:
            semantic_citations.append((consumer.file, consumer.line, consumer.name))

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
            if symbol and symbol not in actual_line:
                claim = ClaimVerification(
                    status='unverified',
                    citation=Citation(file=file, line=line, symbol=symbol),
                    reason=f"Symbol '{symbol}' not found on line {line}"
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
            claim = ClaimVerification(
                status='unverified',
                citation=Citation(file=file, line=line, symbol=symbol),
                reason=f"Error reading file: {str(e)}"
            )
            report.claims.append(claim)
            knowledge.needs_human.append(f"Unverified citation: {file}:{line}:{symbol}")

    return report
