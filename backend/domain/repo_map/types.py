"""Repo map and symbol extraction types (RPA-032)."""
from enum import Enum

from pydantic import BaseModel


class SymbolKind(str, Enum):
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    INTERFACE = "interface"
    ENUM = "enum"
    TYPE = "type"
    VARIABLE = "variable"
    MODULE = "module"


class ExtractSource(str, Enum):
    AST = "ast"
    LEXICAL = "lexical"


class ExtractMode(str, Enum):
    LEXICAL = "lexical"
    HYBRID = "hybrid"


class BuildRepoMapRequest(BaseModel):
    snapshot_id: str
    force_rebuild: bool = True


class SymbolRecord(BaseModel):
    id: str
    snapshot_id: str
    rel_path: str
    language: str | None
    name: str
    kind: SymbolKind
    line_start: int
    line_end: int
    signature: str | None
    parent_name: str | None = None
    extract_source: ExtractSource = ExtractSource.LEXICAL


class RepoMapSummary(BaseModel):
    snapshot_id: str
    total_symbols: int
    files_indexed: int
    parse_failures: int
    extract_mode: ExtractMode
    language_breakdown: dict[str, int]
    kind_breakdown: dict[str, int]
    generated_at: str


class BuildRepoMapResponse(BaseModel):
    summary: RepoMapSummary


class SymbolsResponse(BaseModel):
    snapshot_id: str
    symbols: list[SymbolRecord]


class RepoMapCsvResponse(BaseModel):
    snapshot_id: str
    row_count: int
    csv: str
